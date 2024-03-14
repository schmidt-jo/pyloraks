"""
To build Loraks as efficiently as possible we dont want to recompute constant vectors and matrices
we implement a base class which computes and stores those objects.

Jochen Schmidt, 12.03.2024
"""
import abc
import torch
import logging
from . import fns_ops
from .. utils import plotting
import pathlib as plib
log_module = logging.getLogger(__name__)


class Base:
    def __init__(self, k_space_input: torch.tensor, mask_indices_input: torch.tensor,
                 mode: str, radius: int, rank: int, lam: float, max_num_iter: int, conv_tol: float,
                 device: torch.device = torch.device("cpu"), fig_path: plib.Path = None, visualize: bool = True):
        log_module.info(f"config loraks flavor")
        # config
        self.device: torch.device = device
        # self.fig_path = plib.Path(opts.output_path).absolute().joinpath("plots")
        # self.fig_path.mkdir(parents=True, exist_ok=True)

        log_module.info(f"initialize | set input matrices | set operators")
        # input
        self.mask_indices_input: torch.tensor = mask_indices_input
        # save dimensions - input assumed [x, y, z, ch, t]
        self.dim_read, self.dim_phase, self.dim_slice, self.dim_channels, self.dim_echoes = k_space_input.shape
        # combined xy dim
        self.dim_s = self.dim_read * self.dim_phase
        # combined ch - t dim
        self.dim_t_ch = self.dim_echoes * self.dim_channels
        # loraks params
        self.radius: int = radius
        self.rank: int = rank
        self.mode: str = mode
        self.lam: float = lam
        self.max_num_iter: int = max_num_iter
        self.conv_tol: float = conv_tol
        # neighborhood dim
        self.dim_nb = fns_ops.get_k_radius_grid_points(self.radius).shape[0]
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
        self.fhf = torch.zeros((self.dim_s, self.dim_echoes))
        self.fhf[self.mask_indices_input[:, 0], self.mask_indices_input[:, 1]] = 1
        # we can use this to build us data from fs or just populating the us data (fHd),
        # hence fHd basically is the us input
        # store k-space in fhd vector - this would create us data from fs input data provided a mask
        # dims fhf [xy, t], k-space-in [z, xy, ch, t]
        self.fhd: torch.tensor = self.fhf[None, :, None, :] * self.k_space_input

        # set iterate - if we would initialize k space with 0 -> loraks terms would give all 0s,
        # and first iterations would just regress to fhd, hence we can init fhd from the get go
        self.k_iter_current: torch.tensor = self.fhd.clone().detach()
        self.iter_residuals: torch.tensor = torch.zeros((self.dim_slice, self.max_num_iter))

        # get operator
        self.op_x: fns_ops.S | fns_ops.G | fns_ops.C = fns_ops.get_operator_from_mode(
            mode=self.mode, k_space_dims=self.k_space_dims, radius=self.radius)

        # p*p is the same matrix irrespective of channel / time sampling information,
        # we can compute it for single slice, single channel, single echo data
        self.p_star_p: torch.tensor = torch.abs(
            self.op_x.p_star_p(k_vector=torch.ones(self.dim_s, dtype=torch.complex128))
        )
        self.visualize: bool = visualize
        self.fig_path: plib.Path = fig_path

        if self.visualize:
            plotting.plot_slice(
                torch.reshape(self.p_star_p, (self.dim_phase, self.dim_phase)),
                name="p_star_p", outpath=self.fig_path,
            )

        #     logging.debug(f"look at undersampled data")
        #     plot_d = torch.reshape(self.fhd[int(self.dim_slice/2), :, 0, 0], (self.dim_read, self.dim_phase))
        #     plotting.plot_img(img_tensor=plot_d.clone().detach().cpu(), log_mag=True,
        #                       out_path=self.fig_path, name=f"us_k_space")
        #     d_sl_us_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_d)))
        #     plotting.plot_img(img_tensor=d_sl_us_img_recon.clone().detach().cpu(),
        #                       out_path=self.fig_path, name="naive_us_recon")
        #     plotting.plot_img(img_tensor=torch.reshape(self.fhf[:, 0], (self.dim_read, self.dim_phase)),
        #                       out_path=self.fig_path, name="fhf")

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
