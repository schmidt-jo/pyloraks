import sys
import pathlib as plib

p_wd = plib.Path(__file__).absolute().parent.parent
sys.path.append(p_wd.as_posix())

import logging
import torch
from pyloraks import plotting, options, in_out, flavors, compression, fns_ops
from lprd import recon_fns
import pathlib as plib
import wandb


def main(opts: options.Config):
    logging.info(f"___ Loraks Reconstruction ___")
    logging.info(f"{opts.flavour}; Rank - {opts.rank}; Radius - {opts.radius}; "
                 f"Lambda - {opts.lam}; mode - {opts.mode}; coil compression - {opts.coil_compression}")
    out_path = plib.Path(opts.output_path).absolute()
    fig_path = out_path.joinpath("plots/")
    if opts.visualize:
        fig_path.mkdir(parents=True, exist_ok=True)

    # set up device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    torch.manual_seed(0)
    # device = torch.device("cpu")
    # if input path empty we use phantom
    if opts.input_k == "":
        k_space, _ = in_out.get_shepp_logan_phantom(250, 250, snr=20)
        k_space_dims = k_space.shape
        k = torch.flatten(k_space)
        # set sampling mask - dim l - assume 2 times acceleration and random sampling
        l = int(k.shape[0] / 2)
        f_indexes = torch.randint(k.shape[0], size=(l,)).unique(dim=0)  # rows of S x S identity matrix to keep
        sli = 1
        affine = torch.eye(4)
    else:
        if opts.input_k.endswith(".pt"):
            k_space = in_out.load_k_data(opts.input_k)
            affine = in_out.load_affine(opts.input_k_extra)
        else:
            k_space_real, affine = in_out.load_nii_data(opts.input_k)
            k_space_phase, _ = in_out.load_k_data(opts.input_k_extra)
            k_space = k_space_real * torch.exp(1j * k_space_phase)
        while k_space.shape.__len__() < 5:
            k_space = torch.unsqueeze(k_space, -1)

        # choose only middles slice if slice toggle set
        if opts.process_slice:
            mid_slice = int(k_space.shape[2] / 2)
            logging.info(f"single slice processing: pick slice {mid_slice + 1}")
            k_space = k_space[:, :, mid_slice, None]
        # get sampling pattern
        sampling_pattern = in_out.load_sampling_pattern(opts.input_sampling_pattern)
        if opts.debug:
            # for debuging reduce dims
            sampling_pattern = sampling_pattern[:, :, :3]
            k_space = k_space[:, :, :2, :12, :3]
        if sampling_pattern.shape.__len__() < 3:
            # sampling pattern supposed to be xy, t
            sampling_pattern = sampling_pattern[:, :, None]

        # moveaxis to have read in first dim
        k_space = fns_ops.shift_read_dir(k_space, read_dir=opts.read_dir)
        sampling_pattern = fns_ops.shift_read_dir(sampling_pattern, read_dir=opts.read_dir)

        # look at data
        if opts.visualize:
            logging.debug(f"look at fs data")
            sli, ch, t = (torch.tensor([*k_space.shape[2:]]) / 2).to(torch.int)
            plot_k = k_space[:, :, sli, ch, 0]
            plotting.plot_img(img_tensor=plot_k.clone().detach().cpu(), log_mag=True,
                              out_path=fig_path, name=f"fs_k_space")
            fs_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_k)))
            plotting.plot_img(img_tensor=fs_img_recon.clone().detach().cpu(),
                              out_path=fig_path, name="fs_recon")
            if opts.debug:
                plotting.plot_img(sampling_pattern[:, :, 0].to(torch.int), out_path=fig_path, name="sampling_pattern")

        if opts.coil_compression is not None:
            if opts.coil_compression < k_space.shape[-2]:
                # compress channels
                k_space = compression.compress_channels(
                    input_k_space=k_space,
                    sampling_pattern=sampling_pattern,
                    opts=opts
                )
        read, phase, sli, ch, t = k_space.shape
        # flatten xy dims
        s_xy = sampling_pattern.shape[0] * sampling_pattern.shape[1]
        if abs(s_xy - read * phase) > 1e-3:
            err = f"sampling pattern dimensions do not match input k-space data"
            logging.error(err)
            raise ValueError(err)
        # concatenate ch dimensions to sampling mask
        # sampling_pattern = torch.repeat_interleave(sampling_pattern, ch, dim=-1)
        # get indices for sampling -> first dim vector in xy direction sampled 1 or not 0,
        # second dim echo index -> different sampling k-space dependent on echo
        f_indexes = torch.squeeze(
            torch.nonzero(
                torch.reshape(
                    sampling_pattern.to(torch.int),
                    (s_xy, -1)
                )
            )
        )
    # set on device
    k = k_space.to(device)

    loraks_flavor = flavors.get_flavor_from_opts(opts=opts)(
        k_space_input=k, mask_indices_input=f_indexes, opts=opts
    )
    loraks_k_vector = loraks_flavor.reconstruct()

    if opts.debug:
        # save recon k-space
        nii_name = f"loraks_k_space_recon_r-{opts.radius}_l-{opts.lam}_rank-{opts.rank}"
        in_out.save_data(out_path=out_path, name=nii_name, data=loraks_k_vector, affine=affine)

    # recon sos and phase coil combination
    np_loraks_k_space_recon = loraks_k_vector.numpy(force=True)
    np_loraks_mag, np_loraks_phase, _ = recon_fns.fft_mag_sos_phase_aspire_recon(
        k_data_xyz_ch_t=np_loraks_k_space_recon, se_indices=opts.aspire_echo_indexes
    )
    loraks_img = torch.from_numpy(np_loraks_mag) * torch.exp(1j * torch.from_numpy(np_loraks_phase))

    # move back first axis to where read was
    loraks_img = fns_ops.shift_read_dir(loraks_img, read_dir=opts.read_dir, forward=False)
    # rearranging x, y, z, t
    nii_name = f"loraks_image_recon_r-{opts.radius}_l-{opts.lam}_rank-{opts.rank}"
    in_out.save_data(out_path=out_path, name=nii_name, data=loraks_img, affine=affine)

    if opts.visualize:
        logging.debug(f"reshape to k-space and plot")
        # choose some slice
        sli_id = int(sli / 2)
        # dims [x, y, ]
        plotting.plot_img(loraks_img[:, :, sli_id, 0], out_path=fig_path,
                          name=f"loraks_img_recon_r-{opts.radius}_l-{opts.lam}_rank_reduced-{opts.rank}")


if __name__ == '__main__':
    parser = options.creat_cli()
    args = parser.parse_args()
    # load opts
    config = options.Config.from_cli(args=args)
    # set up logging
    if config.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
        if config.wandb:
            # setup wandb
            wandb.init()
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=level)
    try:
        main(opts=config)
    except Exception as e:
        logging.error(e)
        parser.print_usage()
        exit(-1)
