import logging
from pyloraks import options, utils, algorithm
import torch
import pathlib as plib
import plotly.graph_objects as go
import wandb


# want a fucntion to wrap the program if not used from CLI but in a pipeline
def reconstruction(
        config_file: str, output_path: str, input_k: str,
        input_extra: str = None, input_sampling_mask: str = None,
        coil_compression: int = 8, read_dir: int = 0,
        debug: bool = False, process_slice: bool = False,
        visualize: bool = True, flavour: str = "AC-LORAKS",
        radius: int = 3, conv_tol: float = 1e-3, max_num_iter: int = 10,
        lambda_s: float = 0.1, rank_s: int = 250, lambda_c: float = 0.1, rank_c: int = 150,
        batch_size: int = 4, use_wandb: bool = False):
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

    :param debug: bool (default=False)
    If set provides additional debugging information. Reduces input data for quicker processing.

    :param process_slice: bool (default=False)
    toggle processing of middle slize and not whole volume, eg. for testing LORAKS parameters.

    :param visualize: bool (default=True)
    If set enables visualization & plots of intermediate data.

    :param flavour: str (default="AC-LORAKS")
    LORAKS flavour. Implementation of different LORAKS variations

    :param rank_c: int (default=150)
    LORAKS rank for C Matrix. Applicable only if lambda_c > 0.

    :param lambda_c: float (default=0.1)
    LORAKS regularization parameter for C Matrix rank regularization.

    :param rank_s: int (default=250)
    LORAKS rank for S Matrix. Applicable only if lambda_s > 0.

    :param lambda_s: float (default=0.1)
    LORAKS regularization parameter for S Matrix rank regularization.

    :param radius: int (default=3)
    LORAKS neighborhood radius to construct sparse matrix.

    :param conv_tol: float (default=1e-3)
    LORAKS convergence tolerance.
    
    :param max_num_iter: int (default=10)
    Maximum number of CGD iterations to solve minimization.
    
    :param batch_size: int (default=4)
    Batch-size for batched implementation if echoes*coils to big for memory. Not yet Implemented!

    :param use_wandb: bool (default=False)
    Wandb logging.
    """

    opts = options.Config(
        config_file=config_file, output_path=output_path,
        input_k=input_k, input_extra=input_extra, input_sampling_mask=input_sampling_mask,
        coil_compression=coil_compression, read_dir=read_dir,  # aspire_echo_indexes=aspire_echo_indexes,
        debug=debug, process_slice=process_slice, visualize=visualize,
        flavour=flavour, conv_tol=conv_tol, max_num_iter=max_num_iter, radius=radius,
        rank_c=rank_c, lambda_c=lambda_c, rank_s=rank_s, lambda_s=lambda_s,
        batch_size=batch_size, use_wandb=use_wandb
    )
    main(opts)


def setup_data(opts: options.Config):
    logging.debug("Load data")
    k_space, affine, sampling_pattern = utils.load_data(
        input_k_space=opts.input_k,
        input_sampling_pattern=opts.input_sampling_mask,
        input_extra=opts.input_extra
    )
    if opts.read_dir > 0:
        k_space = torch.swapdims(k_space, 0, 1)
        sampling_pattern = torch.swapdims(sampling_pattern, 0, 1)

    logging.debug(f"Check Debug toggle set & if, reduce dims")
    if opts.debug:
        # for debuging take one coil
        k_space = k_space[:, :, :, 0, None, :]
        # also take one slice. if not set anyway, we set it
        opts.process_slice = True
    logging.debug(f"Check single slice toggle set")
    if opts.process_slice:
        mid_slice = int(k_space.shape[2] / 2)
        logging.info(f"single slice processing: pick slice {mid_slice + 1}")
        k_space = k_space[:, :, mid_slice, None]
    logging.debug(f"Check sampling pattern shape")
    if sampling_pattern.shape.__len__() < 3:
        # sampling pattern supposed to be x, y, t
        sampling_pattern = sampling_pattern[:, :, None]

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
    return k_space, f_indexes, affine


def main(opts: options.Config):
    # setup
    out_path = plib.Path(opts.output_path).absolute()
    fig_path = out_path.joinpath("plots/")
    if opts.visualize:
        fig_path.mkdir(parents=True, exist_ok=True)

    # set up device
    if opts.use_gpu:
        logging.info(f"configuring gpu::  cuda:{opts.gpu_device}")
        device = torch.device(f"cuda:{opts.gpu_device}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(0)

    k_space, f_indexes, affine = setup_data(opts=opts)

    # ToDo: Want to set all matrices used throughout different algorithms / flavors as object
    # ToDo: chose algorithm / flavor, send object and execute

    logging.info(f"___ Loraks Reconstruction ___")
    logging.info(f"{opts.flavour}; Radius - {opts.radius}; Rank C - {opts.rank_c}; Lambda C - {opts.lambda_c}; "
                 f"Rank S - {opts.rank_s}; Lambda S - {opts.lambda_s}; "
                 f"coil compression - {opts.coil_compression}")

    # recon sos and phase coil combination
    solver = algorithm.ACLoraks(
        k_space_input=k_space, mask_indices_input=f_indexes,
        radius=opts.radius,
        rank_c=opts.rank_c, lambda_c=opts.lambda_c,
        rank_s=opts.rank_s, lambda_s=opts.lambda_s,
        max_num_iter=opts.max_num_iter, conv_tol=opts.conv_tol,
        fft_algorithm=False, device=device, fig_path=fig_path
    )
    solver.reconstruct()

    # print stats
    residuals, stats = solver.get_residuals()
    logging.info(f"Minimum residual l2: {stats['norm_res_min']:.3f}")
    logging.info(f"save optimizer loss plot")
    # quick plot of residual sum
    fig = go.Figure()
    for idx_slice in range(solver.dim_slice):
        fig.add_trace(
            go.Scattergl(y=residuals[idx_slice], name=f"slice: {idx_slice}")
        )
    fig_name = out_path.joinpath(f"residuals.html")
    logging.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # get k-space
    loraks_recon = solver.get_k_space()
    # ToDo implement aspire phase reconstruction

    # switch back
    if opts.read_dir > 0:
        loraks_recon = torch.swapdims(loraks_recon, 0, 1)

    if opts.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    logging.info(f"Save k-space reconstruction")
    loraks_name = f"loraks_k_space_recon_r-{opts.radius}"
    if opts.lambda_c > 1e-6:
        loraks_name = f"{loraks_name}_lc-{opts.lambda_c:.3f}_rank-c-{opts.rank_c}"
    if opts.lambda_s > 1e-6:
        loraks_name = f"{loraks_name}_ls-{opts.lambda_s:.3f}_rank-s-{opts.rank_s}"
    loraks_name = loraks_name.replace(".", "p")
    file_name = out_path.joinpath(loraks_name).with_suffix(".pt")
    logging.info(f"write file: {file_name}")
    torch.save(loraks_recon, file_name.as_posix())

    # save recon as nii for now to look at
    # rSoS k-space-data for looking at it
    loraks_recon_mag = torch.sqrt(
        torch.sum(
            torch.square(
                torch.abs(loraks_recon)
            ),
            dim=-2
        )
    )

    loraks_phase = torch.angle(loraks_recon)
    loraks_phase = torch.mean(loraks_phase, dim=-2)

    loraks_recon_k = loraks_recon_mag * torch.exp(1j * loraks_phase)
    if opts.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    utils.save_data(out_path=out_path, name=loraks_name, data=loraks_recon_k, affine=affine)

    logging.info("FFT into image space")
    # fft into real space
    loraks_recon_img = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(
                loraks_recon, dim=(0, 1)
            ),
            dim=(0, 1)
        ),
        dim=(0, 1)
    )

    logging.info("rSoS channels")
    # for nii we rSoS combine channels
    loraks_recon_mag = torch.sqrt(
        torch.sum(
            torch.square(
                torch.abs(loraks_recon_img)
            ),
            dim=-2
        )
    )

    loraks_phase = torch.angle(loraks_recon_img)
    loraks_phase = torch.mean(loraks_phase, dim=-2)

    loraks_recon_img = loraks_recon_mag * torch.exp(1j * loraks_phase)

    nii_name = loraks_name.replace("k_space", "image")
    utils.save_data(out_path=out_path, name=nii_name, data=loraks_recon_img, affine=affine)


def optimize_loraks_params():
    wandb.init()
    # load opts
    opts = options.Config.load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/pulseq_2024-03-07_megesse-mese-acc-test/"
        "processed/pyloraks/wandb/pyloraks_config.json"
    )
    # setup
    out_path = plib.Path(opts.output_path).absolute()
    fig_path = out_path.joinpath("plots/")
    if opts.visualize:
        fig_path.mkdir(parents=True, exist_ok=True)

    # set up device
    if opts.use_gpu:
        logging.info(f"configuring gpu::  cuda:{opts.gpu_device}")
        device = torch.device(f"cuda:{opts.gpu_device}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(0)

    k_space, f_indexes, _ = setup_data(opts=opts)

    # get loraks params from wandb
    loraks_radius = wandb.config["radius"]
    loraks_rank_c = wandb.config["rank_c"]
    loraks_lambda_c = wandb.config["lambda_c"]
    loraks_rank_s = wandb.config["rank_s"]
    loraks_lambda_s = wandb.config["lambda_s"]
    # loraks_radius = 3
    # loraks_rank_c = 100
    # loraks_lambda_c = 0.2
    # loraks_rank_s = 200
    # loraks_lambda_s = 0.2

    # recon sos and phase coil combination
    solver = algorithm.ACLoraks(
        k_space_input=k_space, mask_indices_input=f_indexes,
        radius=loraks_radius, rank_c=loraks_rank_c, lambda_c=loraks_lambda_c,
        rank_s=loraks_rank_s, lambda_s=loraks_lambda_s,
        max_num_iter=opts.max_num_iter, conv_tol=opts.conv_tol,
        fft_algorithm=False, device=device, fig_path=fig_path, visualize=opts.visualize
    )
    solver.reconstruct()

    # get stats and min residual norm
    residual_vector, stats = solver.get_residuals()
    logging.info(f"Finished Run. Min l2 residual norm: {stats['norm_res_min']:3f}")
    wandb.log({"res_min": stats["norm_res_min"]})


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

    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=level)

    logging.info(f"_______________________________")
    logging.info(f"___ PyLORAKS reconstruction ___")
    logging.info(f"_______________________________")
    try:
        if config.use_wandb:
            # setup wandb
            wandb.init()
            optimize_loraks_params()
        else:
            main(opts=config)
    except Exception as e:
        logging.exception(e)
        parser.print_usage()
        exit(-1)
