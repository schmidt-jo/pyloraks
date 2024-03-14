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
        self.aha = torch.flatten(torch.repeat_interleave(self.fhf, self.dim_channels, dim=-1)
                                 + self.lam * self.p_star_p[:, None]).to(self.device)

    def get_acs_v(self, idx_slice: int):
        """
        Compute the V matrix for solving the AC LORAKS formulation from autocalibration data.
        We need to find the ACS data first and then evaluate the nullspace subspace.
        """
        if self.mode == "c" or self.mode == "C":
            # want submatrix of respecitve loraks matrix with fully populated rows
            c_idx = self.op_x.operator(k_space=self.fhf)
            idxs = torch.nonzero(torch.sum(c_idx, dim=1) == c_idx.shape[1])
            # build X matrix from k-space
            m_ac = self.op_x.operator(k_space=self.k_space_input[idx_slice])[idxs]
        elif self.mode == "s" or self.mode == "S":
            # we can use the sampling mask, rebuild it in 2d
            mask = torch.reshape(self.fhf, (self.dim_read, self.dim_phase, self.dim_echoes))
            # momentarily the implementation is aiming at joint echo recon. we use all echoes,
            # but might need to batch the channel dimensions. in the nullspace approx we use all data
            # get indices for centered origin k-space
            p_nb_nx_ny_idxs, m_nb_nx_ny_idxs = self.op_x.get_point_symmetric_neighborhoods()
            # we want to find indices for which the whole neighborhood of point symmetric coordinates is contained
            a = mask[m_nb_nx_ny_idxs[:, :, 0], m_nb_nx_ny_idxs[:, :, 1]]
            b = mask[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]]
            # lets look for the ACS which is constant across the joint dimension and the neighborhood
            c = torch.sum(a, dim=(1, 2)) + torch.sum(b, dim=(1, 2))
            idxs_ac = (c == self.dim_nb * self.dim_t_ch * 2)
            # now we build the s_matrix with the 0 filled data
            s_mac_k_in = torch.reshape(self.fhd[idx_slice], (self.dim_s, self.dim_t_ch))
            s_matrix = self.op_x.operator(k_space=s_mac_k_in)
            # and keep only the fully populated rows
            m_ac = s_matrix[torch.tile(idxs_ac, dims=(2,))]
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
        # evaluate nullspace
        u, s, v = torch.linalg.svd(m_ac, full_matrices=False)
        m_ac_rank = v.shape[1]
        if m_ac_rank < self.rank:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)
        # get subspaces from svd of subspace matrix
        v_null = v[self.rank:].to(self.device)
        v_sub = v[:self.rank].to(self.device)
        # u_sub = u[:self.rank].to(self.device)
        # u_null = u[self.rank:].to(self.device)
        # get into complex shape
        # dim_nb = fns_ops.get_k_radius_grid_points(self.radius).shape[0]

        # if idx_slice == 0 and self.visualize:
        #     # nullspace: reshape and split values
        #     v_null_split_cplx = v_null
        #     dim_ns = v_null_split_cplx.shape[1]
        #     v_null_split_cplx = torch.reshape(v_null_split_cplx, (dim_ns, self.dim_nb, -1))
        #     v_null_split_cplx = v_null_split_cplx[:, :, ::2] + 1j * v_null_split_cplx[:, :, 1::2]
        #     v_null_cplx = torch.reshape(v_null_split_cplx, (dim_ns, -1)).conj().T
        #     plotting.plot_slice(v_null_cplx, name="cplx_nullspace_slice_0", outpath=self.fig_path)
        #     # subspace: reshape and split values
        #     v_sub_split_cplx = v_sub
        #     dim_ns = v_sub_split_cplx.shape[0]
        #     v_sub_split_cplx = torch.reshape(v_sub_split_cplx, (dim_ns, self.dim_nb, -1))
        #     v_sub_split_cplx = v_sub_split_cplx[:, :, ::2] + 1j * v_sub_split_cplx[:, :, 1::2]
        #     v_null_cplx = torch.reshape(v_sub_split_cplx, (dim_ns, -1)).conj().T
        #     plotting.plot_slice(v_null_cplx, name="cplx_subspace_slice_0", outpath=self.fig_path)

        return v_null.T, v_sub.T

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
        return m1_fhf + self.lam * m1_v

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

    def _cgd(self, fhd, v):
        residual_abs_sum = []
        # get starting value of f
        f = fhd.clone()
        # starting value of rhs is also fhd
        f_rand = torch.rand_like(f) * 1e-2
        f[torch.abs(f) < 1e-6] = f_rand[torch.abs(f) < 1e-6]
        # get matrix (is only a vector, matrix would be a diagonal matrix, hence we save memory here)

        # get residual
        r = fhd - self._get_m_1_diag_vector(f, v)
        # get vars
        z = r.clone()

        for i in tqdm.trange(self.max_num_iter, desc="Processing::Slice::Iteration"):
            rr = torch.dot(r, r)
            residual_abs_sum.append(torch.abs(rr))
            if torch.abs(rr) < self.conv_tol:
                log_module.info("Convergence reached!")
                break

            az = self._get_m_1_diag_vector(z, v)
            zaz = torch.dot(z, az)
            a = torch.div(rr, zaz)
            f = f + a * z
            r_new = r - a * az
            rr_new = torch.dot(r_new, r_new)
            beta = rr_new / rr
            z = r_new + beta * z
            r = r_new
        return f, residual_abs_sum

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
            v_null, v_sub = self.get_acs_v(idx_slice=idx_slice)

            vvh = torch.matmul(v_sub, v_sub.conj().T)

            log_module.debug("start pcg solve")

            in_fhd = torch.flatten(self.fhd[idx_slice]).to(self.device)
            # in_fhd = torch.flatten(self.fhd[slice_idx]).numpy()

            f_slice, residual_abs_sum = self._cgd(in_fhd, vvh)
            self.k_iter_current[idx_slice] = torch.reshape(
                f_slice,
                (-1, self.dim_channels, self.dim_echoes)
            )
            self.iter_residuals[idx_slice, :len(residual_abs_sum)] = torch.tensor(residual_abs_sum)

            # ToDo implement different algorithmic choices here:
            #  multiplicative Majorize-Minimize Approach - FFT version, start with fft
            # if self.fft_algorithm:
            #     self._solve_fft()
            # else
            #     self._solve()
            # try:
            # loraks_solve = ssl.cg(
            #     M1, in_fhd, x0=in_fhd, tol=1e-9, atol=1e-5,
            #     maxiter=self.opts.max_num_iter, callback=cg_counter
            # )
            # f = torch.rand_like(in_fhd)
            # for _ in tqdm.trange(self.max_num_iter, desc="loraks solve iteration"):
            #     f_last = f
            #     xs = self.cgd(
            #         input_f=f, input_fhd=in_fhd, vvh=vvh
            #     )
            #     convergence = torch.linalg.norm(xs[1:] - xs[:-1], dim=-1)
            #     log_module.info(f"convergence: {convergence}")
            #     f = xs[-1]
            #
            # loraks_solve = torch.from_numpy(loraks_solve[0])
            # # except SmallEnoughException:
            # #     loraks_solve = cg_counter.last_iterate
            # self.k_iter_current[idx_slice] = torch.reshape(
            #     loraks_solve,
            #     (self.dim_s, self.dim_channels, self.dim_echoes)
            # )
