import simple_parsing as sp
import dataclasses as dc
import pathlib as plib
import logging

log_module = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.Serializable):
    config_file: str = sp.field(
        alias="-c", default="./pyloraks/example_data/config.json",
        help=f"Provide configuration file in which all relevant options are given. "
             f"Can be overwritten by additional cmd-line inputs"
    )
    output_path: str = sp.field(
        alias="-o", default="",
        help=f"(Optional) Specify output path. If left blank, input path is used."
    )
    input_k: str = sp.field(
        alias="-im", default="./pyloraks/example_data/fs_k.pt",
        help=f"Specify input k-space data path. If no sampling mask is provided unsampled data needs to be 0 padded. "
             f"Takes .nii (.gz), .npy and .pt files. The latter two can be complex data."
    )
    input_extra: str = sp.field(
        alias="-ip", default="./pyloraks/example_data/fs_affine.pt",
        help="If input is not .nii format, we need the affine transformation as additional input to output .nii."
             "If input is .nii format, we can use this to input additional phase data."
    )
    input_sampling_mask: str = sp.field(
        alias="-is", default="./pyloraks/example_data/us_sampling_mask.pt",
        help=f"Specify input k-space-sampling pattern path. "
             f"If no sampling mask is provided it will be deduced from input data."
    )
    coil_compression: int = sp.field(
        alias="-cc", default=8,
        help=f"Specify coil compression for multi channel data. Default working mode is "
             f"Joint-Echo-Channel reconstruction, which can lead to memory problems. "
             f"Compression can help in those cases."
    )
    read_dir: int = sp.field(
        alias="-rd", default=0,
        help="specify read direction if not in x. Necessary for AC LORAKS to deduce AC region."
    )
    # aspire_echo_indexes: tuple = sp.field(
    #     alias="-aei", default_factory=(0, 1),
    #     help="Input echo indexes to use for phase coil combination with aspire (m=1). "
    #          "Usually the indices of first two SE data in the echo train"
    # )

    debug: bool = sp.field(
        alias="-d", default=False,
        help="If set provides additional debugging information. Reduces input data for quicker processing."
    )
    process_slice: bool = sp.field(
        alias="-ps", default=False,
        help="toggle processing of middle slize and not whole volume, eg. for testing LORAKS parameters."
    )
    visualize: bool = sp.field(
        alias="-v", default=True,
        help="If set enables visualization & plots of intermediate data."
    )
    flavour: str = sp.field(
        choices=["AC-LORAKS", "LORAKS"], alias="-f", default="AC-LORAKS",
        help=f"LORAKS flavour. Implementation of different LORAKS variations."
    )

    radius: int = sp.field(alias="-r", default=3)
    rank_c: int = sp.field(alias="-rrc", default=150)
    lambda_c: float = sp.field(
        alias="-lc", default=0.1,
        help=f"regularization parameter for Loraks C matrix "
             f"rank minimization. Set 0.0 to disable C regularization."
    )
    rank_s: int = sp.field(alias="-rrs", default=250)
    lambda_s: float = sp.field(
        alias="-ls", default=0.1,
        help=f"regularization parameter for Loraks S matrix "
             f"rank minimization. Set 0.0 to disable S regularization."
    )

    # lambda_data: float = sp.field(alias="-dl", default=0.5)
    conv_tol: float = sp.field(alias="-ct", default=1e-3)
    max_num_iter: int = sp.field(alias="-mni", default=10)

    batch_size: int = sp.field(alias="-b", default=4)
    use_gpu: bool = sp.field(alias="-gpu", default=False)
    gpu_device: int = sp.field(alias="-gpud", default=0)
    use_wandb: bool = sp.field(alias="-wb", default=False)

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        c_instance = cls()
        default_dict = c_instance.__dict__
        config_file = False
        if args.config.config_file != default_dict["config_file"]:
            path = plib.Path(args.config.config_file).absolute()
            if path.is_file():
                log_module.info(f"loading config file: {path.as_posix()}")
                c_instance = cls().load(path.as_posix())
                config_file = True
            else:
                err = f"{path.as_posix()} is not a file!"
                log_module.error(err)
                raise ValueError(err)
        # catch additional non default cli input
        for key, val in args.config.to_dict().items():
            # if item in arguments passed is non default, we assume its given explicitly via cli
            # then we overwrite the entry in the class. if config file was loaded or not
            default_val = default_dict.__getitem__(key)
            if default_val != val:
                if config_file and key != "config_file":
                    log_module.info(f"additional cli value {val} found for {key}. overwrite config file entry")
                c_instance.__setattr__(key, val)
        return c_instance


def creat_cli() -> sp.ArgumentParser:
    parser = sp.ArgumentParser(prog="pyloraks")
    parser.add_arguments(Config, dest="config")
    return parser


if __name__ == '__main__':
    # create config instance
    config = Config()
    # save to example location
    path = plib.Path(__file__).absolute().parent.joinpath("example_data/")
    path = path.joinpath("default_config").with_suffix(".json")
    config.config_file = path.relative_to(plib.Path(__file__).absolute().parent.parent).__str__()
    logging.info(f"write file: {path}")
    config.save_json(path.__str__(), indent=2)
    # check if properly loading
    l_conf = config.load(path.as_posix())
    assert l_conf == config
