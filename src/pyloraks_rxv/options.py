import simple_parsing as sp
import dataclasses as dc
import pathlib as plib
import logging

log_module = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.Serializable):
    config_file: str = sp.field(alias="-c", default="./pyloraks/example_data/config.json")
    output_path: str = sp.field(alias="-o", default="./pyloraks/example_data/")
    input_k: str = sp.field(alias="-im", default="./pyloraks/example_data/fs_k.pt")
    input_k_extra: str = sp.field(alias="-ip", default="./pyloraks/example_data/fs_affine.pt",
                                  help="input additional info needed such as affine or nii img header")
    input_sampling_pattern: str = sp.field(alias="-is",
                                           default="./pyloraks/example_data/us_sampling_mask.pt")
    coil_compression: int = sp.field(alias="-cc", default=8)
    debug: bool = sp.field(alias="-d", default=False)
    process_slice: bool = sp.field(alias="-ps", default=False,
                                   help="toggle processing of middle slize eg. for setting of LORAKS parameters."
                                   )
    visualize: bool = sp.field(alias="-v", default=True)

    mode: str = sp.field(choices=["s", "S", "c", "C", "g", "G"], alias="m", default="s")
    flavour: str = sp.field(choices=["AC-Loraks", "Loraks"], alias="-f", default="AC-Loraks")

    rank: int = sp.field(alias="-rr", default=100)
    radius: int = sp.field(alias="-r", default=3)
    lam: float = sp.field(alias="-l", default=0.1)

    lambda_data: float = sp.field(alias="-dl", default=0.5)
    conv_tol: float = sp.field(alias="-ct", default=1e-3)
    batch_size: int = sp.field(alias="-b", default=4)
    max_num_iter: int = sp.field(alias="-mni", default=10)

    read_dir: int = sp.field(alias="-rd", default=0,
                             help="specify read direction if not in x")
    aspire_echo_indexes: tuple = sp.field(alias="-aei", default=(0, 1),
                                          help="input echo indexes to use for phase coil combination with aspire (m=1)")
    wandb: bool = sp.field(alias="-wb", default=False)

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
