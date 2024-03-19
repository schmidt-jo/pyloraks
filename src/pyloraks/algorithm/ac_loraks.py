from .base import Base
from . import fns_ops
from ..utils import plotting
import logging
import torch
import tqdm
import pathlib as plib

log_module = logging.getLogger(__name__)


class ACLoraks(Base):
    def __init__(
            self, k_space_input: torch.tensor, mask_indices_input: torch.tensor,
            mode: str, radius: int, rank: int, lam: float, max_num_iter: int, conv_tol: float,
            fft_algorithm: bool = False,
            device: torch.device = torch.device("cpu"), fig_path: plib.Path = None, visualize: bool = True):
        log_module.debug(f"Setup Base class")
        # initialize base constants
        super().__init__(
            k_space_input=k_space_input, mask_indices_input=mask_indices_input,
            mode=mode, radius=radius, rank=rank, lam=lam, max_num_iter=max_num_iter, conv_tol=conv_tol,
            device=device, fig_path=fig_path, visualize=visualize
        )
        log_module.debug(f"Setup AC LORAKS specifics")
        self.fft_algorithm: bool = fft_algorithm
        # compute aha - stretch along channels
        fhf = self.fhf[:, None, :].expand((-1, self.dim_channels, -1))
        self.aha = torch.flatten(fhf + self.lam * self.p_star_p[:, None, None]).to(self.device)
        # recon takes place per slice for combined echo and channel dimensions.
        # we can thus find the AC region and save it to not redo the computation each slice
        # we save the indices of mapped vectors to op matrix after finding which neighborhoods are
        # completely mapped into the operator matrix
        self.idxs_ac: torch.tensor = self._find_ac_indices()

    def _find_ac_indices(self):
        # we can use the sampling mask, rebuild it in 2d, sampling pattern equal per coil but different per echo
        mask = torch.reshape(self.fhf, (self.dim_read, self.dim_phase, self.dim_echoes))
        # momentarily the implementation is aiming at joint echo recon. we use all echoes,
        # but might need to batch the channel dimensions. in the nullspace approx we use all data
        # get indices for centered origin k-space
        p_nb_nx_ny_idxs, m_nb_nx_ny_idxs = self.op_x.point_sym_nb_coos
        # we want to find indices for which the whole neighborhood of point symmetric coordinates is contained
        a = mask[m_nb_nx_ny_idxs[:, :, 0], m_nb_nx_ny_idxs[:, :, 1]]
        b = mask[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]]
        # lets look for the ACS which is constant across the joint dimension and the neighborhood
        # we check the mask for it, irrespective of the combined dimension, its usually sampled different
        # for different echoes, but equally for the different channels. We can deduce the dimensions from the mask.
        c = torch.sum(a, dim=(1, 2)) + torch.sum(b, dim=(1, 2))
        idxs_ac = torch.tile((c == a.shape[1] * a.shape[2] * 2), dims=(2,))
        # plot extracted region if flag set
        if self.visualize:
            # plot the found acs for reference
            dim = self.dim_nb * self.dim_t_ch
            if self.mode in ["s", "S"]:
                dim *= 2
            acs = torch.zeros((idxs_ac.shape[0], dim))
            acs[idxs_ac] = 1
            acs = self.op_x.operator_adjoint(acs)
            acs = torch.reshape(acs, (self.dim_read, self.dim_phase, -1))
            plotting.plot_slice(acs[:, :, 0], f"extracted_AC_region", self.fig_path)
        # return the found indices
        return idxs_ac

    def get_acs_v(self, idx_slice: int):
        """
        Compute the V matrix for solving the AC LORAKS formulation from autocalibration data.
        We need to find the ACS data first and then evaluate the nullspace subspace.
        """
        if self.mode in ["c", "C"]:
            # want submatrix of respecitve loraks matrix with fully populated rows
            c_idx = self.op_x.operator(k_space=self.fhf)
            idxs = torch.nonzero(torch.sum(c_idx, dim=1) == c_idx.shape[1])
            # build X matrix from k-space
            m_ac = self.op_x.operator(k_space=self.k_space_input[idx_slice])[idxs]
        elif self.mode in ["s", "S"]:
            # now we build the s_matrix with the 0 filled data for the current slice
            s_mac_k_in = torch.reshape(self.fhd[idx_slice], (self.dim_s, self.dim_t_ch))
            s_matrix = self.op_x.operator(k_space=s_mac_k_in)
            # and keep only the fully populated rows - we calculated which those are already for all
            # echo sampling patterns
            m_ac = s_matrix[self.idxs_ac]
        else:
            err = f"ACS calibration not implemented (yet) for mode {self.mode}. Choose either c or s"
            log_module.error(err)
            raise ValueError(err)
        if m_ac.shape[0] < self.rank:
            err = f"detected AC region is too small for chosen rank"
            log_module.error(err)
            raise ValueError(err)
        if torch.cuda.is_available():
            m_ac = m_ac.to(torch.device("cuda:0"))

        # via eigh
        eig_vals, eig_vecs = torch.linalg.eigh(torch.matmul(m_ac.T, m_ac))
        m_ac_rank = eig_vals.shape[0]
        # get subspaces from svd of subspace matrix
        eig_vals, idxs = torch.sort(torch.abs(eig_vals), descending=True)
        # eig_vecs_r = eig_vecs[idxs]
        eig_vecs = eig_vecs[:, idxs]
        # v_sub_r = eig_vecs_r[:self.rank].to(self.device)
        v_sub = eig_vecs[:, :self.rank].to(self.device)

        if m_ac_rank < self.rank:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)

        if idx_slice == 0 and self.visualize:
            plotting.plot_slice(v_sub, name="v_sub", outpath=self.fig_path)

        return v_sub

    def _get_m_1_diag_vector(self, f: torch.tensor, v: torch.tensor) -> torch.tensor:
        """
        We define the M_1 matrix from A^H A f and the loraks operator P_x(f) V V^H,
        after extracting V from ACS data and getting the P_x based on the Loraks mode used.
        """
        m1_fhf = self.aha * f
        m1_v = torch.flatten(
            self.op_x.operator_adjoint(
                torch.matmul(
                    self.op_x.operator(torch.reshape(f, (self.dim_s, -1))),
                    v
                )
            )
        )
        return m1_fhf - self.lam * m1_v

    def _solve_fft(self):
        """
        observing L_i(f) = P_x(f)v_i for a column v_i of V matrix. Because of convolutional structure of the
        Loraks Matrices, this operator can be computed via FFTs:
        L_i(f) = T F^-1 ( F(Z_2 v_i) * F(Z_1 f) )
        Where F are FFT operations, Z are zero padding ops, its a element wise multiplication and
        T extracts relevant samples of respective neighborhood.

        It follows we can construct M
        M = A^H A f + lam Z_1^H F^-1 (N * F(Z_1 f) )
        """
        pass

    def _cgd(self, b, v):
        n2b = torch.linalg.norm(b)

        x = torch.zeros_like(b)
        p = 1
        xmin = x
        iimin = 0
        tolb = self.conv_tol * n2b

        r = b - self._get_m_1_diag_vector(x, v)

        normr = torch.linalg.norm(r)
        normr_act = normr

        if normr < tolb:
            log_module.info("convergence before loop")

        res_vec = torch.zeros(self.max_num_iter)
        normrmin = normr

        rho = 1

        for ii in tqdm.trange(self.max_num_iter):
            z = r
            rho1 = rho
            rho = torch.abs(torch.sum(r.conj() * r))
            if ii == 0:
                p = z
            else:
                beta = rho / rho1
                p = z + beta * p
            q = self._get_m_1_diag_vector(p, v)
            pq = torch.sum(p.conj() * q)
            alpha = rho / pq

            x = x + alpha * p
            r = r - alpha * q

            normr = torch.linalg.norm(r)
            normr_act = normr
            res_vec[ii] = normr

            if normr <= tolb:
                r = b - self._get_m_1_diag_vector(x, v)
                normr_act = torch.linalg.norm(r)
                res_vec[ii] = normr_act
                log_module.info(f"reached convergence at step {ii + 1}")
                break

            if normr_act < normrmin:
                normrmin = normr_act
                xmin = x
                iimin = ii
                log_module.debug(f"min residual {normrmin:.2f}, at {iimin + 1}")

        return xmin, res_vec, {"norm_res_min": normrmin, "iteration": iimin}

    def _recon(self):
        """
        AC LORAKS specific reconstruction method.
        """
        # loraks_recon = torch.zeros(self.k_space_dims, dtype=torch.complex128)
        # compute combined echo and channel information slice wise
        # essentially here we are minimizing data consistency for the sampled points against
        # low rank constraints of the LORAKS matrices
        for idx_slice in range(self.dim_slice):
            log_module.info(f"Reconstruction::Processing::Slice {1 + idx_slice} / {self.dim_slice}")
            log_module.debug("estimate nullspace")
            v_sub = self.get_acs_v(idx_slice=idx_slice)

            vvh = torch.matmul(v_sub, v_sub.conj().T)

            log_module.debug("start pcg solve")

            in_fhd = torch.flatten(self.fhd[idx_slice]).to(self.device)

            f_slice, residual_abs_sum, stats = self._cgd(in_fhd, vvh)
            self.k_iter_current[idx_slice] = torch.reshape(
                f_slice,
                (-1, self.dim_channels, self.dim_echoes)
            )
            self.iter_residuals[idx_slice, :len(residual_abs_sum)] = residual_abs_sum.clone().cpu()
            self.stats = stats
            # ToDo implement different algorithmic choices here:
            #  multiplicative Majorize-Minimize Approach - FFT version, start with fft

