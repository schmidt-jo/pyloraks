"""
using
Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR.
A software channel compression technique for faster reconstruction with many channels.
Magn Reson Imaging 2008; 26: 133–141

To do PCA based channel compression for faster computation of LORAKS
"""

import torch
import logging

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
    log_module.info(f"extract ac region from sampling mask")
    # find ac data - this is implemented for the commonly used (in jstmc) sampling scheme of acquiring middle pe lines
    # use 0th echo, ac should be equal for all slices, all channels and all echoes
    # start search in middle of pe dim
    mid_read, mid_pe, _, _, _ = (torch.tensor(sampling_pattern.shape) / 2).to(torch.int)
    lower_edge = mid_pe
    upper_edge = mid_pe
    # make sure sampling pattern is bool
    sampling_pattern = sampling_pattern.to(torch.bool)
    for idx_pe in torch.arange(1, mid_pe):
        # looking for sampled line (actually just looking for the read middle point)
        # if previous line was also sampled were still in a fully sampled region
        # assuming AC region same for all echoes
        next_up_line_sampled = sampling_pattern[mid_read, mid_pe + idx_pe, 0, 0, 0]
        next_low_line_sampled = sampling_pattern[mid_read, mid_pe - idx_pe, 0, 0, 0]
        prev_up_line_sampled = sampling_pattern[mid_read, mid_pe + idx_pe - 1, 0, 0, 0]
        prev_low_line_sampled = sampling_pattern[mid_read, mid_pe - idx_pe + 1, 0, 0, 0]
        if next_up_line_sampled and prev_up_line_sampled:
            upper_edge = mid_pe + idx_pe
        if next_low_line_sampled and prev_low_line_sampled:
            lower_edge = mid_pe - idx_pe
        if not next_up_line_sampled or not next_low_line_sampled:
            # get out if we hit first unsampled line
            break
    # extract ac region
    ac_mask = torch.zeros_like(sampling_pattern[:, :, 0, 0, 0])
    ac_mask[:, lower_edge:upper_edge] = True
    # if opts.visualize and opts.debug:
    #     plotting.plot_img(ac_mask.to(torch.int), out_path=fig_path, name="cc_ac_region")
    log_module.info(f"start pca -> building compression matrix from calibration data")
    # set input data
    pca_data = input_k_space[ac_mask]
    # set channel dimension first and rearrange rest
    pca_data = torch.moveaxis(pca_data, -2, 0)
    pca_data = torch.reshape(pca_data, (num_in_ch, -1))
    # substract mean from each channel vector
    pca_data = pca_data - torch.mean(pca_data, dim=1, keepdim=True)
    # calculate covariance matrix
    cov = torch.cov(pca_data)
    cov_eig_val, cov_eig_vec = torch.linalg.eig(cov)
    # get coil compression matrix
    a_l_matrix = cov_eig_vec[:num_compressed_channels]
    log_module.info(f"compressing data channels from {num_in_ch} to {num_compressed_channels}")
    # compress data -> coil dimension over a_l
    compressed_data = torch.einsum("iklmn, om -> iklon", input_k_space, a_l_matrix)
    # if opts.visualize and opts.debug:
    #     sli, ch, t = (torch.tensor([*compressed_data.shape[2:]]) / 2).to(torch.int)
    #     plot_k = compressed_data[:, :, sli, ch, 0]
    #     plotting.plot_img(img_tensor=plot_k.clone().detach().cpu(), log_mag=True,
    #                       out_path=fig_path, name=f"fs_k_space_compressed")
    #     fs_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_k)))
    #     plotting.plot_img(img_tensor=fs_img_recon.clone().detach().cpu(),
    #                       out_path=fig_path, name="fs_recon_compressed")
    return compressed_data
