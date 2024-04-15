"""
using
Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR.
A software channel compression technique for faster reconstruction with many channels.
Magn Reson Imaging 2008; 26: 133–141

and gcc -> idea that coil sensitivities are varying dependent on spatial position, but are assumed to be unknown.
We benefit here from using a coil compression in each slice.

Tao Zhang, John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
Coil  Compression for Accelerated Imaging with Cartesian Sampling
Magn Reson Med. 2013 69

We do not need to compute inverse fourier transform along slice dimension as we have a slice wise acquisition.
Do PCA based channel compression for faster computation of LORAKS
"""

import torch
import logging
import tqdm
import plotly.graph_objects as go

log_module = logging.getLogger(__name__)


def compress_channels(input_k_space: torch.tensor, sampling_pattern: torch.tensor,
                      num_compressed_channels: int) -> torch.tensor:
    """ k-space is assumed to be provided in dims [x, y, z, ch, t (optional)]"""
    # set path for plotting
    # out_path = plib.Path(opts.output_path).absolute()
    # fig_path = out_path.joinpath("plots/")
    # check input
    if input_k_space.shape.__len__() < 5:
        input_k_space = torch.unsqueeze(input_k_space, -1)
    if sampling_pattern.shape.__len__() < 5:
        sampling_pattern = torch.unsqueeze(sampling_pattern, 2)
    # check if we are actually provided fewer channels
    num_in_ch = input_k_space.shape[-2]
    if num_in_ch < num_compressed_channels:
        msg = (f"input data has fewer channels ({num_in_ch}) than set for compression ({num_compressed_channels})! "
               f"No compression done.")
        log_module.info(msg)
        return input_k_space
    # slice wise processing
    log_module.info(f"extract ac region from sampling mask")
    # find ac data - we look for fully contained neighborhoods starting from middle position of the sampling mask
    nx, ny = sampling_pattern.shape[:2]
    mid_read = int(nx / 2)
    mid_pe = int(ny / 2)
    # make sure sampling pattern is bool
    sampling_pattern = sampling_pattern.to(torch.bool)
    # move from middle out
    l = mid_read
    r = mid_read
    b = mid_pe
    t = mid_pe
    for idx_pos in torch.arange(1, max(mid_read, mid_pe)):
        if mid_read - idx_pos > 0:
            if sampling_pattern[mid_read - idx_pos, mid_pe, 0, 0, 0]:
                l -= 1
            if sampling_pattern[mid_read + idx_pos, mid_pe, 0, 0, 0]:
                r += 1
        if mid_pe - idx_pos > 0:
            if sampling_pattern[mid_read, mid_pe - idx_pos, 0, 0, 0]:
                b -= 1
            if sampling_pattern[mid_read, mid_pe + idx_pos, 0, 0, 0]:
                t += 1
    # extract ac region
    ac_mask = torch.zeros_like(sampling_pattern[:, :, 0, 0, 0])
    ac_mask[l:r, b:t] = True
    # set input data
    ac_data = input_k_space[ac_mask]
    # allocate output_data
    compressed_data = torch.zeros(
        (*input_k_space.shape[:-2], num_compressed_channels, input_k_space.shape[-1]),
        dtype=input_k_space.dtype
    )
    log_module.info(f"start pca -> building compression matrix from calibration data")
    for idx_slice in tqdm.trange(input_k_space.shape[2]):
        # if opts.visualize and opts.debug:
        #     plotting.plot_img(ac_mask.to(torch.int), out_path=fig_path, name="cc_ac_region")
        sli_data = ac_data[:, idx_slice]
        # set channel dimension first and rearrange rest
        sli_data = torch.moveaxis(sli_data, -2, 0)
        sli_data = torch.reshape(sli_data, (num_in_ch, -1))
        # substract mean from each channel vector
        sli_data = sli_data - torch.mean(sli_data, dim=1, keepdim=True)
        # calculate covariance matrix
        cov = torch.cov(sli_data)
        cov_eig_val, cov_eig_vec = torch.linalg.eig(cov)
        # get coil compression matrix
        a_l_matrix = cov_eig_vec[:num_compressed_channels]
        log_module.debug(f"compressing data channels from {num_in_ch} to {num_compressed_channels}")
        # compress data -> coil dimension over a_l
        compressed_data[:, :, idx_slice] = torch.einsum(
            "ikmn, om -> ikon",
            input_k_space[:, :, idx_slice],
            a_l_matrix
        )
    # if opts.visualize and opts.debug:
    #     sli, ch, t = (torch.tensor([*compressed_data.shape[2:]]) / 2).to(torch.int)
    #     plot_k = compressed_data[:, :, sli, ch, 0]
    #     plotting.plot_img(img_tensor=plot_k.clone().detach().cpu(), log_mag=True,
    #                       out_path=fig_path, name=f"fs_k_space_compressed")
    #     fs_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_k)))
    #     plotting.plot_img(img_tensor=fs_img_recon.clone().detach().cpu(),
    #                       out_path=fig_path, name="fs_recon_compressed")
    return compressed_data
