""" implement different loraks flavors.
We define k to be the k-space representation to be evaluated,
F is the matrix that converts full k-space sampled data to the undersampled representation
d is the undersampled noisy measured data.

______________________
J.Schmidt 07.08.2023
"""
from typing import Type

import torch
import logging
from pyloraks import options, fns_ops, plotting
import tqdm
from collections import OrderedDict
import pathlib as plib
from scipy.sparse import linalg as ssl
import abc
log_module = logging.getLogger(__name__)


class LoraksFlavor:
    def __init__(self, k_space_input: torch.tensor, mask_indices_input: torch.tensor,
                 opts: options.Config, device: torch.device = torch.device("cpu")):
        log_module.info(f"config loraks flavor")
        # config
        self.opts: options.Config = opts
        self.device: torch.device = device
        self.fig_path = plib.Path(opts.output_path).absolute().joinpath("plots")
        self.fig_path.mkdir(parents=True, exist_ok=True)

        log_module.info(f"initialize | set input matrices | set operators")
        # input
        self.mask_indices_input: torch.tensor = mask_indices_input
        # save dimensions - input assumed [x, y, z, ch, t]
        self.dim_read, self.dim_phase, self.dim_slice, self.dim_channels, self.dim_echoes = k_space_input.shape
        # combined xy dim
        self.dim_s = self.dim_read * self.dim_phase
        # combined ch - t dim
        self.dim_t_ch = self.dim_echoes * self.dim_channels
        # neighborhood dim
        self.dim_nb = fns_ops.get_k_radius_grid_points(self.opts.radius).shape[0]
        # want to set z to first dimension
        self.k_space_dims = k_space_input.shape
        k_space_input = torch.moveaxis(k_space_input, 2, 0)
        # want to combine xy dims
        self.k_space_input: torch.tensor = torch.reshape(
            k_space_input,
            (self.dim_slice, self.dim_s, self.dim_channels, self.dim_echoes)
        )
        # build fHf
        # f maps k space data of length L to us data of length S,
        # fHf is the vector with 1 at positions of undersampled points and 0 otherwise,
        # basically mapping a vector of length L to us subspace S and back to L.
        self.fhf = torch.zeros((self.dim_s, self.dim_echoes), device=self.device)
        self.fhf[self.mask_indices_input[:, 0], self.mask_indices_input[:, 1]] = 1
        # we can use this to build us data from fs or just populating the us data (fHd),
        # hence fHd basically is the us input
        # store k-space in fhd vector - this would create us data from fs input data provided a mask
        # dims fhf [xy, t], k-space-in [z, xy, ch, t]
        self.fhd: torch.tensor = self.fhf[None, :, None, :] * self.k_space_input

        # set iterate - if we would initialize k space with 0 -> loraks terms would give all 0s,
        # and first iterations would just regress to fhd, hence we can init fhd from the get go
        self.k_iter_current: torch.tensor = self.fhd.clone().detach()
        self.k_iter_last: torch.tensor = self.fhd.clone().detach()

        # get operator
        self.op_x: fns_ops.S | fns_ops.G | fns_ops.C = fns_ops.get_operator_from_mode(
            mode=self.opts.mode, k_space_dims=self.k_space_dims, radius=self.opts.radius)

        # p*p is the same matrix irrespective of channel / time sampling information,
        # we can compute it for single slice, single channel, single echo data
        self.p_star_p: torch.tensor = torch.abs(
            self.op_x.p_star_p(k_vector=torch.ones(self.dim_s, dtype=torch.complex128))
        )

        if opts.visualize:
            logging.debug(f"look at undersampled data")
            plot_d = torch.reshape(self.fhd[int(self.dim_slice/2), :, 0, 0], (self.dim_read, self.dim_phase))
            plotting.plot_img(img_tensor=plot_d.clone().detach().cpu(), log_mag=True,
                              out_path=self.fig_path, name=f"us_k_space")
            d_sl_us_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_d)))
            plotting.plot_img(img_tensor=d_sl_us_img_recon.clone().detach().cpu(),
                              out_path=self.fig_path, name="naive_us_recon")
            plotting.plot_img(img_tensor=torch.reshape(self.fhf[:, 0], (self.dim_read, self.dim_phase)),
                              out_path=self.fig_path, name="fhf")

    def reconstruct(self):
        log_module.info(f"start processing")
        self._recon()
        # move slice dim back, dims [z, xy, ch, t]
        k_space_recon = torch.moveaxis(self.k_iter_current, 0, 1)
        # reshape to original input dims [xy, z, ch, t]
        k_space_recon = torch.reshape(k_space_recon, self.k_space_dims)
        return k_space_recon

    @abc.abstractmethod
    def _recon(self):
        """ to be set in the individual flavors, aka abstract method """
        return NotImplementedError


class Loraks(LoraksFlavor):
    def __init__(self, k_space_input: torch.tensor, mask_indices_input: torch.tensor, opts: options.Config):
        super().__init__(k_space_input, mask_indices_input, opts)
        # reminder: dimensions of k-space / fhd [ z, xy, ch, t]

        log_module.info(f"Setup standard loraks")
        # standard loraks flavor specific

        log_module.debug(f"set slice and deduce unchanged matrices")
        # compute phi dagger
        # equal for all slices and all channels! Compute without those dims ->
        # caution fhf dims [xy, t], p*p dims [xy] -> phi dagger is dims [xy, t]
        self.phi_dagger: torch.tensor = torch.nan_to_num(
            torch.div(
                1.0,
                self.opts.lambda_data * self.fhf +
                (1.0 - self.opts.lambda_data) * self.opts.lam * self.p_star_p[:, None]
            ),
            nan=0.0, posinf=0.0
        )

        # progress bar
        self.iter_bar = tqdm.trange(self.opts.max_num_iter)
        self.iter_bar.set_description(f"loraks iteration steps")

        # if we set k vector to 0 initially, the loraks contributions will give 0, just set init frob norm really high
        fro_x = torch.tensor(1e8)
        # l2 = torch.linalg.norm(torch.matmul(f, loraks_k_vector) - d_sl_us)
        l2 = torch.zeros(1)
        # compute loss
        loss = l2 + self.opts.lam * fro_x
        postfix_dict = {
            "loss": loss.item(), "l2": l2.item(), "loraks": fro_x.item(), "convergence": 1e8,
        }
        self.iter_bar.set_postfix(ordered_dict=OrderedDict(postfix_dict))

    def _recon(self):
        with self.iter_bar as iter_bar:
            for iter_idx in iter_bar:
                fro_x = torch.zeros(1)
                self.k_iter_last = self.k_iter_current.clone().detach()
                for idx_slice in range(self.dim_slice):
                    postfix_dict.__setitem__("slice", f"{idx_slice + 1}/{self.dim_slice}")
                    iter_bar.set_postfix(ordered_dict=OrderedDict(postfix_dict))
                    loraks_k_slice, fro_x_slice = self.loraks_process_slice(
                        slice_idx=idx_slice, t=iter_bar
                    )
                    self.k_iter_current[idx_slice] = loraks_k_slice
                    fro_x += fro_x_slice
                # compute loss for whole volume
                l2 = torch.linalg.norm(
                    torch.reshape(
                        self.k_iter_current[:, self.mask_indices_input[:, 0], :, self.mask_indices_input[:, 1]],
                        self.fhd.shape
                    ) - self.fhd
                )
                # compute loss
                loss = l2 + self.opts.lam * fro_x
                # compute convergence
                convergence = torch.linalg.norm(self.k_iter_last - self.k_iter_current) / torch.linalg.norm(self.k_iter_current)
                postfix_dict = {
                    "loss": loss.item(), "l2": l2.item(), "loraks": fro_x.item(), "convergence": convergence.item(),
                }
                iter_bar.set_postfix(ordered_dict=OrderedDict(postfix_dict))
                if convergence < self.opts.conv_tol and iter_idx > 3:
                    log_module.info(f"reached set convergence tolerance!")
                    break

    def loraks_process_slice(self, slice_idx: int, t: tqdm) -> (torch.tensor, torch.tensor):
        fro_x = torch.zeros(1)
        # we want to use joint reconstruction, i.e. mix channel and echo information,
        # but due to matrix size we cant compute the whole slice matrix at once.
        # we need to take care of the sampling pattern which is equal for channels but different for echoes
        # try: using all echoes but batch of randomly permuted channels
        ch_perm_idx = torch.randperm(self.dim_channels)
        batch_ch_size = self.opts.batch_size
        ch_perm_idx = torch.split(ch_perm_idx, batch_ch_size)
        # store k-space-recon slice information, since the slice data is batched we need to allocate
        k_iter_slice = torch.zeros_like(self.k_iter_current[slice_idx])
        batch_num_steps = len(ch_perm_idx)
        for batch_idx in range(batch_num_steps):
            # update bar
            postfix_list = [a.split(sep="=") for a in t.postfix.split(sep=", ")]
            postfix_dict = {}
            for a in postfix_list:
                key, val = a
                if key != "batch" and key != "slice":
                    postfix_dict.__setitem__(key, float(val))
                else:
                    postfix_dict.__setitem__(key, val)
            postfix_dict.__setitem__("batch", f"{batch_idx + 1}/{batch_num_steps}")
            t.set_postfix(OrderedDict(postfix_dict))
            # compute
            # chose data - dims per slice [xy, ch, t] - combine t-ch
            batch = torch.reshape(self.k_iter_current[:, ch_perm_idx[batch_idx], :], (self.dim_s, -1))
            # build X matrix from k-space
            x_matrix = self.op_x.operator(k_space=batch)
            # compute low rank approximation of x
            x = fns_ops.construct_low_rank_matrix_from_svd(matrix_tensor=x_matrix, rank=self.opts.rank)
            # compute norm
            fro_x += torch.linalg.norm(x_matrix - x, ord="fro")
            # compute k from x -> dims [xy, t*batch_size] cast to [xy, batch_size, t]
            k_x = torch.reshape(
                self.op_x.operator_adjoint(x_matrix=x),
                (self.dim_s, ch_perm_idx[batch_idx].shape[0], k_iter_slice.shape[-1])
            )
            # compute new k iterate ->
            # k_iter dims [xy, ch, t], phi_dagger dims [xy, t], fhd dims [xy, ch, t], k_x dims [xy, batch_size, t]
            k_iter_slice[:, ch_perm_idx[batch_idx], :] = self.phi_dagger[:, None, :] * (
                    self.opts.lambda_data * self.fhd[slice_idx, :, ch_perm_idx[batch_idx], :] +
                    (1.0 - self.opts.lambda_data) * self.opts.lam * k_x
            )
        return k_iter_slice, fro_x


class AcLoraks(LoraksFlavor):
    def __init__(self, k_space_input: torch.tensor, mask_indices_input: torch.tensor, opts: options.Config):
        super().__init__(k_space_input, mask_indices_input, opts)
        # reminder: dimensions of k-space / fhd [ z, xy, ch, t]
        log_module.info(f"setup ac loraks")

        log_module.debug(f"set slice and deduce unchanged matrices")
        # compute aha
        self.aha = torch.flatten(torch.repeat_interleave(self.fhf, self.dim_channels, dim=-1)
                                 + self.opts.lam * self.p_star_p[:, None])

        self.slice_bar_dict_default = OrderedDict({
            "x-tract v": "[ ]", "pcg solve": "[ ]"
        })

    def _recon(self):
        with tqdm.trange(self.dim_slice) as slice_bar:
            slice_bar.set_description(
                f"loraks iteration - joint-recon using compressed channels and all echoes - slice")
            # reset bar descriptions
            slice_bar.set_postfix(self.slice_bar_dict_default)

            for slice_idx in slice_bar:
                slice_bar_dict = self.slice_bar_dict_default.copy()
                # extract ac region - supposed to be equal positions for all slices
                # but obviously k-space data is different

                log_module.debug("estimate nullspace")
                slice_bar_dict.__setitem__("x-tract v", "[x]")
                slice_bar.set_postfix(slice_bar_dict)
                v_null, v_sub = self.get_acs_v(slice_idx=slice_idx)

                vvh = torch.matmul(v_sub, v_sub.conj().T)

                # M1 = self.get_lin_operator(vvh=vvh)
                # define NxN matrix for solver as lin operation on vector (aka diagonal matrix)
                # - dont want to spawn a huge matrix with only diagonal entries

                log_module.debug("start pcg solve")
                # solve
                slice_bar_dict.__setitem__("pcg solve", "[x]")
                slice_bar.set_postfix(slice_bar_dict)
                # cg_counter = IterCountBar(
                #     slice_num=slice_idx, bar=slice_bar, add_info_tensor=v_sub, loraks_flavor_obj=self
                # )
                in_fhd = torch.flatten(self.fhd[slice_idx])
                # in_fhd = torch.flatten(self.fhd[slice_idx]).numpy()

                # try:
                    # loraks_solve = ssl.cg(
                    #     M1, in_fhd, x0=in_fhd, tol=1e-9, atol=1e-5,
                    #     maxiter=self.opts.max_num_iter, callback=cg_counter
                    # )
                f = torch.rand_like(in_fhd)
                for _ in tqdm.trange(self.opts.max_num_iter, desc="loraks solve iteration"):
                    f_last = f
                    xs = self.cgd(
                        input_f=f, input_fhd=in_fhd, vvh=vvh
                    )
                    convergence = torch.linalg.norm(xs[1:] - xs[:-1], dim=-1)
                    log_module.info(f"convergence: {convergence}")
                    f = xs[-1]

                loraks_solve = torch.from_numpy(loraks_solve[0])
                # except SmallEnoughException:
                #     loraks_solve = cg_counter.last_iterate
                self.k_iter_current[slice_idx] = torch.reshape(
                    loraks_solve,
                    (self.dim_s, self.dim_channels, self.dim_echoes)
                )

    def m_1_matrix(self, f: torch.tensor, vvh: torch.tensor):
        m1_fhf = self.aha * f
        m1_v = torch.flatten(
            self.op_x.operator_adjoint(
                torch.matmul(
                    self.op_x.operator(torch.reshape(f, (self.dim_s, -1))), vvh
                )
            )
        )
        return m1_fhf + self.opts.lam * m1_v

    def get_lin_operator(self, vvh: torch.tensor):
        def m_1_matrix(f):
            return self.m_1_matrix(f=torch.from_numpy(f), vvh=vvh).numpy()
        flat_shape = self.dim_s * self.dim_t_ch
        return ssl.LinearOperator(dtype=complex, shape=(flat_shape, flat_shape), matvec=m_1_matrix)

    def cgd(self, input_f: torch.tensor, input_fhd: torch.tensor, vvh: torch.tensor):
        f = input_f.clone()
        m1_f = self.m_1_matrix(f=f, vvh=vvh)
        dim = m1_f.shape[0]
        # get residual, we want to do this vectorized in 1d, the matrix a is assumed to be diagonal
        # and represented as vector
        r = input_fhd - m1_f
        d = r
        rr = torch.dot(r, r)
        xs = f.clone()[None]

        for _ in tqdm.trange(1, 5):
            # matrix multiplication, since m1(d) is a vector we can simply multiply pointwise
            Ad = self.m_1_matrix(d, vvh=vvh)
            alpha = rr / torch.dot(d, Ad)
            f += alpha * d
            r -= alpha * Ad
            rr_new = torch.dot(r, r)
            beta = rr_new / rr
            d = r + beta * d
            rr = rr_new
            print(torch.max(torch.abs(r)))
            xs = torch.concat((xs, f[None]), dim=0)

        return xs

    def get_acs_v(self, slice_idx: int):
        if self.opts.mode == "c" or self.opts.mode == "C":
            # want submatrix of respecitve loraks matrix with fully populated rows
            c_idx = self.op_x.operator(k_space=self.fhf)
            idxs = torch.nonzero(torch.sum(c_idx, dim=1) == c_idx.shape[1])
            # build X matrix from k-space
            m_ac = self.op_x.operator(k_space=self.k_space_input[slice_idx])[idxs]
        elif self.opts.mode == "s" or self.opts.mode == "S":
            # we can use the sampling mask, rebuild it in 2d
            mask = torch.reshape(self.fhf, (self.dim_read, self.dim_phase, self.dim_echoes))
            # save neighborhood size
            kn_pts = fns_ops.get_k_radius_grid_points(self.opts.radius)

            # momentarily the implementation is aiming at joint echo recon. we use all echoes,
            # but might need to batch the channel dimensions. in the nullspace approx we use all data
            # get indices for centered origin k-space
            m_nx_ny_idxs, p_nx_ny_idxs = self.op_x.get_half_space_k_dims()
            p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
            m_nb_nx_ny_idxs = m_nx_ny_idxs[:, None] + kn_pts[None, :]
            # get indices for which whole neighborhood is contained for both locations

            # for all echoes we check which k-space parts are included as a whole
            # first get all masking points which are accessed in s_matrix creation
            upper_half_mask = mask[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1], :]
            lower_half_mask = mask[m_nb_nx_ny_idxs[:, :, 0], m_nb_nx_ny_idxs[:, :, 1], :]
            # get indices if whole neighborhood is included
            # (i.e. each point in neighborhood has mask value 1, for all echoes)
            idxs_ac = torch.squeeze(torch.nonzero(
                (torch.sum(upper_half_mask, dim=(1, 2)) == self.dim_echoes * self.dim_nb) &
                (torch.sum(lower_half_mask, dim=(1, 2)) == self.dim_echoes * self.dim_nb)
            )
            )
            # now we build the s_matrix with the 0 filled data
            s_mac_k_in = torch.reshape(self.fhd[slice_idx], (self.dim_s, self.dim_t_ch))
            s_matrix = self.op_x.operator(k_space=s_mac_k_in)
            # and keep only the fully populated rows
            m_ac = s_matrix[idxs_ac]
        else:
            err = f"ACS calibration not implemented (yet) for mode {self.opts.mode}. Choose either c or s"
            log_module.error(err)
            raise ValueError(err)
        if m_ac.shape[0] < self.opts.rank:
            err = f"detected AC region is too small for chosen rank"
            log_module.error(err)
            raise ValueError(err)
        if torch.cuda.is_available():
            m_ac = m_ac.to(torch.device("cuda:0"))
        # evaluate nullspace
        u, s, v = torch.linalg.svd(m_ac, full_matrices=False)
        m_ac_rank = v.shape[1]
        if m_ac_rank < self.opts.rank:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)
        # get subspaces from svd of subspace matrix
        v_null = v[self.opts.rank:].to(self.device)
        v_sub = v[:self.opts.rank].to(self.device)
        return v_null.conj().T, v_sub.conj().T


# set exception to use for convergence
class SmallEnoughException(Exception):
    pass


# define counter class
class IterCountBar:
    def __init__(self, slice_num: int, add_info_tensor: torch.tensor,
                 loraks_flavor_obj: Loraks | AcLoraks, bar: tqdm):
        self.counter: int = 0
        self.loraks_obj: Loraks | AcLoraks = loraks_flavor_obj
        self.opts: options.Config = self.loraks_obj.opts
        self.add_info_tensor: torch.tensor = add_info_tensor
        self.fro: float = 0.0
        self.l2: float = 0.0
        self.bar: tqdm = bar
        self.convergence: float = 1.0
        self.slice_num: int = slice_num
        self.last_iterate: torch.tensor = loraks_flavor_obj.fhd[slice_num].clone().detach()

    def loss(self):
        return self.fro + self.l2

    def __call__(self, xk):
        self.counter += 1
        # get xk to dims [xy, ch, t]
        xk = torch.reshape(
            torch.from_numpy(xk),
            (self.loraks_obj.dim_s, self.loraks_obj.dim_channels, self.loraks_obj.dim_echoes)
        )
        # compute l2 with candidate
        self.l2 = torch.linalg.norm(
            torch.flatten(
                self.loraks_obj.fhf[:, None, :] * xk - self.loraks_obj.fhd[self.slice_num]
            ), 2
        ).item()
        # compute frob norm with candidate
        if self.opts.flavour == "AC-Loraks":
            self.fro = (self.opts.lam * torch.linalg.norm(
                torch.matmul(
                    self.loraks_obj.op_x.operator(
                        k_space=torch.reshape(
                            xk,
                            (self.loraks_obj.dim_s, -1)
                        )
                    ),
                    self.add_info_tensor
                ), 'fro'
            )).item()
        else:
            # ToDo generalize counter class
            self.fro = self.add_info_tensor
        # compute convergence with candidate and last iterate
        self.convergence = (
                torch.linalg.norm(
                    torch.flatten(
                        xk -
                        self.last_iterate
                    )
                ) /
                torch.linalg.norm(
                    torch.flatten(
                        xk
                    )
                )
        ).item()
        log_module.debug(f"loss: {self.loss():.4f}, l2: {self.l2:.4f}, loraks: {self.fro:.4f},"
                         f"convergence: {self.convergence:.4f}")
        self.update_bar()
        d_us_k_space = torch.reshape(
            xk[:, 0, 0],
            (self.loraks_obj.dim_read, self.loraks_obj.dim_phase)
        )
        d_us_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(d_us_k_space)))
        plotting.plot_img(img_tensor=d_us_img_recon.clone().detach().cpu(), supress_log=True,
                          out_path=self.loraks_obj.fig_path,
                          name=f"loraks_us_recon_slice_{self.slice_num+1}_iter_{self.counter}")
        # save current state as last state
        self.last_iterate = xk
        if self.convergence < self.opts.conv_tol:
            log_module.debug(f"convergence criterion met")
            raise SmallEnoughException

    def update_bar(self):
        # update bar
        postfix_list = [a.split(sep="=") for a in self.bar.postfix.split(sep=", ")]
        postfix_dict = OrderedDict({})
        for a in postfix_list:
            k, v = a
            postfix_dict.__setitem__(k, v)
        postfix_dict.__setitem__("iteration", f"{self.counter}")
        postfix_dict.__setitem__("l2", f"{self.l2:.5f}")
        postfix_dict.__setitem__("fro", f"{self.fro:.5f}")
        postfix_dict.__setitem__("loss", f"{self.loss():.5f}")
        postfix_dict.__setitem__("convergence", f"{self.convergence:.5f}")
        self.bar.set_postfix(postfix_dict)


def get_flavor_from_opts(opts: options.Config) -> Type[AcLoraks | Loraks]:
    d_available = {
        "AC-Loraks": AcLoraks,
        "Loraks": Loraks
    }
    flavor = d_available.get(opts.flavour)
    if flavor is None:
        err = f"Set Loraks flavor ({opts.flavour}) not available! Chose one of {d_available.keys()}"
        log_module.error(err)
        raise ValueError(err)
    return flavor



