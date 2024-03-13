import logging
import wandb
from pyloraks import options, utils, algorithm
import torch
import pathlib as plib
import plotly.graph_objects as go


# want a fucntion to wrap the program if not used from CLI but in a pipeline
def reconstruction(
        config_file: str, output_path: str, input_k: str,
        input_extra: str = None, input_sampling_mask: str= None,
        coil_compression: int = 8, read_dir: int = 0,
        aspire_echo_indexes: tuple = (0, 1),
        debug: bool = False, process_slice: bool = False,
        visualize: bool = True, mode: str = "s",
        flavour: str = "AC-LORAKS",
        rank: int = 100, radius: int = 3, lam: float = 0.1,
        lambda_data: float = 0.5, conv_tol: float = 1e-3, max_num_iter: int = 10,
        batch_size: int = 4, wandb: bool = False):
    """
    :param config_file: str
    Provide configuration file in which all relevant options are given.
    Can be overwritten by additional cmd-line inputs

    :param output_path: str
    (Optional) Specify output path. If left blank, input path is used.

    :param input_k: str
    Specify input k-space data path. If no sampling mask is provided unsampled data needs to be 0 padded.
    Takes .nii (.gz), .npy and .pt files. The latter two can be complex data.

    :param input_extra: str
    If input is not .nii format, we need the affine transformation as additional input to output .nii.
    If input is .nii format, we can use this to input additional phase data.

    :param input_sampling_mask: str
    Specify input k-space-sampling pattern path.
    If no sampling mask is provided it will be deduced from input data.

    :param coil_compression: int (default=8)
    Specify coil compression for multi channel data. Default working mode is
    Joint-Echo-Channel reconstruction, which can lead to memory problems.
    Compression can help in those cases.

    :param read_dir: int (default=0)
    specify read direction if not in x. Necessary for AC LORAKS to deduce AC region.

    :param aspire_echo_indexes: tuple (default=(0, 1))
    Input echo indexes to use for phase coil combination with aspire (m=1).
    Usually the indices of first two SE data in the echo train

    :param debug: bool (default=False)
    If set provides additional debugging information. Reduces input data for quicker processing.

    :param process_slice: bool (default=False)
    toggle processing of middle slize and not whole volume, eg. for testing LORAKS parameters.

    :param visualize: bool (default=True)
    If set enables visualization & plots of intermediate data.

    :param mode: str (choices=["s", "S", "c", "C", "g", "G"], default="s")
    LORAKS mode.

    :param flavour: str (default="AC-LORAKS")
    LORAKS flavour. Implementation of different LORAKS variations

    :param rank: int (default=100)
    LORAKS rank.

    :params radius: int (default=3)
    LORAKS neighborhood radius to construct sparse matrix.

    :params lam: float (default=0.1)
    LORAKS lambda.

    :params lambda_data: float (default=0.5)
    LORAKS lambda for data consistency term.

    :params conv_tol: float (default=1e-3)
    LORAKS convergence tolerance.

    :params batch_size: int (default=4)
    Batch-size for batched implementation if echoes*coils to big for memory. Not yet Implemented!

    :params wandb: bool (default=False)
    Wandb logging.
    """

    opts = options.Config(
        config_file=config_file, output_path=output_path,
        input_k=input_k, input_extra=input_extra, input_sampling_mask=input_sampling_mask,
        coil_compression=coil_compression, read_dir=read_dir,    # aspire_echo_indexes=aspire_echo_indexes,
        debug=debug, process_slice=process_slice, visualize=visualize,
        mode=mode, flavour=flavour,
        rank=rank, radius=radius, lam=lam, lambda_data=lambda_data, conv_tol=conv_tol, max_num_iter=max_num_iter,
        batch_size=batch_size, wandb=wandb
    )
    main(opts)


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
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    logging.debug("Load data")
    k_space, affine, sampling_pattern = utils.load_data(
        input_k_space=opts.input_k,
        input_sampling_pattern=opts.input_sampling_mask,
        input_extra=opts.input_extra
    )
    if opts.read_dir > 0:
        k_space = torch.swapdims(k_space, 0, 1)
        sampling_pattern = torch.swapdims(sampling_pattern, 0, 1)

    logging.debug(f"Check single slice toggle set")
    if opts.process_slice:
        mid_slice = int(k_space.shape[2] / 2)
        logging.info(f"single slice processing: pick slice {mid_slice + 1}")
        k_space = k_space[:, :, mid_slice, None]
    logging.debug(f"Check Debug toggle set & if, reduce dims")
    if opts.debug:
        # for debuging take one coil
        k_space = k_space[:, :, :, 0, None, :]
    logging.debug(f"Check sampling pattern shape")
    if sampling_pattern.shape.__len__() < 3:
        # sampling pattern supposed to be x, y, t
        sampling_pattern = sampling_pattern[:, :, None]
    logging.debug(f"Visualize if toggle set")
    # if opts.visualize:
    #     logging.debug(f"look at fs data")
    #     sli, ch, t = (torch.tensor([*k_space.shape[2:]]) / 2).to(torch.int)
    #     plot_k = k_space[:, :, sli, ch, 0]
    #     plotting.plot_img(img_tensor=plot_k.clone().detach().cpu(), log_mag=True,
    #                       out_path=fig_path, name=f"fs_k_space")
    #     fs_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_k)))
    #     plotting.plot_img(img_tensor=fs_img_recon.clone().detach().cpu(),
    #                       out_path=fig_path, name="fs_recon")
    #     if opts.debug:
    #         plotting.plot_img(sampling_pattern[:, :, 0].to(torch.int), out_path=fig_path, name="sampling_pattern")

    if opts.coil_compression is not None:
        k_space = utils.compress_channels(
            input_k_space=k_space,
            sampling_pattern=sampling_pattern,
            num_compressed_channels=opts.coil_compression
        )
    read, phase, sli, ch, t = k_space.shape
    # flatten xy dims
    s_xy = sampling_pattern.shape[0] * sampling_pattern.shape[1]
    if abs(s_xy - read * phase) > 1e-3:
        err = f"sampling pattern dimensions do not match input k-space data"
        logging.error(err)
        raise ValueError(err)
    logging.debug(f"Set sampling indices as matrix (fhf)")
    f_indexes = torch.squeeze(
        torch.nonzero(
            torch.reshape(
                sampling_pattern.to(torch.int),
                (s_xy, -1)
            )
        )
    )

    # ToDo: Want to set all matrices used throughout different algorithms / flavors as object
    # ToDo: chose algorithm / flavor, send object and execute

    # recon sos and phase coil combination
    solver = algorithm.ACLoraks(
        k_space_input=k_space, mask_indices_input=f_indexes,
        mode=opts.mode, radius=opts.radius, rank=opts.rank, lam=opts.lam,
        max_num_iter=opts.max_num_iter, conv_tol=opts.conv_tol,
        fft_algorithm=False, device=device, fig_path=fig_path
    )
    loraks_recon = solver.reconstruct()

    # ToDo implement aspire phase reconstruction
    # np_loraks_k_space_recon = loraks_k_vector.numpy(force=True)
    # np_loraks_mag, np_loraks_phase, _ = recon_fns.fft_mag_sos_phase_aspire_recon(
    #     k_data_xyz_ch_t=np_loraks_k_space_recon, se_indices=opts.aspire_echo_indexes
    # )
    # loraks_img = torch.from_numpy(np_loraks_mag) * torch.exp(1j * torch.from_numpy(np_loraks_phase))

    # move back first axis to where read was
    # loraks_img = fns_ops.shift_read_dir(loraks_img, read_dir=opts.read_dir, forward=False)
    # rearranging x, y, z, t

    # switch back
    if opts.read_dir > 0:
        loraks_recon = torch.swapdims(loraks_recon, 0, 1)


    loraks_phase = torch.angle(loraks_recon)
    loraks_phase = torch.mean(loraks_phase, dim=-2)

    loraks_recon_mag = torch.sqrt(
        torch.sum(
            torch.square(loraks_recon),
            dim=-2
        )
    )
    loraks_recon = loraks_recon_mag * torch.exp(1j * loraks_phase)
    if opts.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    nii_name = f"loraks_image_recon_r-{opts.radius}_l-{opts.lam}_rank-{opts.rank}"
    utils.save_data(out_path=out_path, name=nii_name, data=loraks_recon, affine=affine)

    for idx_e in range(3):
        utils.plot_slice(loraks_recon[:, :, 0, idx_e], name=f"recon_echo-{idx_e+1}", outpath=fig_path)

    # quick plot of residual sum
    fig = go.Figure()
    for idx_slice in range(solver.dim_slice):
        fig.add_trace(
            go.Scattergl(y=solver.iter_residuals[idx_slice], name=f"slice: {idx_slice}")
        )
    fig_name = out_path.joinpath(f"residuals.html")
    logging.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # if opts.visualize:
    #     logging.debug(f"reshape to k-space and plot")
    #     # choose some slice
    #     sli_id = int(sli / 2)
    #     # dims [x, y, ]
    #     plotting.plot_img(loraks_img[:, :, sli_id, 0], out_path=fig_path,
    #                       name=f"loraks_img_recon_r-{opts.radius}_l-{opts.lam}_rank_reduced-{opts.rank}")


if __name__ == '__main__':
    # parse input arguments
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

    logging.info(f"_______________________________")
    logging.info(f"___ PyLORAKS reconstruction ___")
    logging.info(f"_______________________________")
    try:
        main(opts=config)
    except Exception as e:
        logging.exception(e)
        parser.print_usage()
        exit(-1)
