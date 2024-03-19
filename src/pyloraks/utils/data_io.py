import numpy as np
import torch
import skimage as ski
import typing
import pathlib as plib
import logging
import nibabel as nib
import pandas as pd
import tqdm
log_module = logging.getLogger(__name__)


def load_k_data(file_name: typing.Union[str, plib.Path]) -> torch.tensor:
    file_name = plib.Path(file_name).absolute()
    if not file_name.is_file():
        err = f"{file_name.as_posix()} not a file"
        log_module.error(err)
        raise ValueError(err)
    if ".nii" in file_name.suffixes:
        data, _ = load_nii_data(file_name=file_name)
    elif ".pt" in file_name.suffixes:
        log_module.info(f"load file: {file_name.as_posix()}")
        data = torch.load(file_name.as_posix())
    elif ".npy" in file_name.suffixes:
        log_module.info(f"load file: {file_name.as_posix()}")
        data = torch.from_numpy(np.load(file_name.as_posix()))
    else:
        err = f"None of {file_name.suffixes} is a recognized suffixes."
        log_module.error(err)
        raise ValueError
    return data


def load_nii_data(file_name: typing.Union[str, plib.Path]) -> (torch.tensor, torch.tensor):
    file_name = plib.Path(file_name).absolute()
    if file_name.is_file():
        log_module.info(f"load file: {file_name.as_posix()}")
        nib_img = nib.load(filename=file_name.as_posix())
        nii_data = torch.from_numpy(nib_img.get_fdata())
        nii_affine = torch.from_numpy(nib_img.affine)
    else:
        err = f"{file_name} not a file."
        log_module.error(err)
        raise ValueError
    return nii_data, nii_affine


def load_affine(file_name: typing.Union[str, plib.Path]) -> torch.tensor:
    file_name = plib.Path(file_name).absolute()
    if not file_name.is_file():
        err = f"{file_name.as_posix()} not a file"
        log_module.error(err)
        raise ValueError(err)
    if ".nii" in file_name.suffixes:
        _, aff = load_nii_data(file_name=file_name)
    else:
        aff = load_data(file_name=file_name)
    return aff


def load_sampling_pattern(file_name: typing.Union[str, plib.Path],
                          n_read: int = 0, n_phase: int = 0, n_echoes: int = 0) -> torch.tensor:
    file_name = plib.Path(file_name).absolute()
    log_module.info(f"loading sampling pattern: {file_name}")
    if ".pkl" in file_name.suffixes:
        sp = pd.read_pickle(file_name)
        if not min(n_read, n_phase, n_echoes) > 0:
            err = "if giving .pkl file you need to specify read, phase and echo dimensions"
            log_module.error(err)
            raise ValueError(err)
        k_sampling_mask = torch.zeros(
            (n_read, n_phase, n_echoes),
            dtype=torch.bool
        )

        for k in tqdm.trange(len(sp), desc="\t\t\t\t\t - build torch tensor mask from pandas sampling file"):
            row = sp.iloc[k]
            pe_num = int(row["pe_num"])
            echo_num = int(row["echo_num"])
            if not sp.iloc[k]["navigator"]:
                k_sampling_mask[:, pe_num, echo_num] = True
    elif ".pt" in file_name.suffixes:
        k_sampling_mask = torch.load(file_name.as_posix())
    elif ".npy" in file_name.suffixes:
        k_sampling_mask = np.load(file_name.as_posix())
    else:
        err = f"none of {file_name.suffixes} supported to load sampling pattern, provide .pkl or .pt files"
        log_module.error(err)
        raise ValueError(err)
    return k_sampling_mask


def load_data(input_k_space: str, input_extra: str = None, input_sampling_pattern: str = None):
    # if input path empty we use phantom
    if input_k_space == "":
        k_space_fs_phantom, _ = get_shepp_logan_phantom(250, 250, snr=20)
        k_space_dims = k_space_fs_phantom.shape
        # set sampling mask - dim l - assume 2 times acceleration and random sampling
        l = int(k_space_dims[0] / 2)
        f_indexes = torch.randint(k_space_dims[0], size=(l,)).unique(dim=0)  # rows of S x S identity matrix to keep
        k_space_data = torch.zeros_like(k_space_fs_phantom)
        k_space_data[f_indexes] = k_space_fs_phantom[f_indexes]
        k_affine = torch.eye(4)
        # force sampling pattern to be regenerated below
        input_sampling_pattern = None
    else:
        # load input data
        k_space_data = load_k_data(input_k_space)
        if input_k_space.endswith(".nii"):
            # load affine from same data
            k_affine = load_affine(input_k_space)
            # load phase if applicable
            if input_extra is not None:
                k_phase_data = load_k_data(input_extra)
                k_space_data = k_phase_data * torch.exp(1j * k_phase_data)
        else:
            # load affine from extra data
            k_affine = load_k_data(input_extra)
    # check dimensions. want [x, y, z, ch, t]. if not available just append
    while k_space_data.shape.__len__() < 5:
        k_space_data = torch.unsqueeze(k_space_data, -1)

    # get sampling pattern
    if input_sampling_pattern is not None:
        input_sampling_pattern = plib.Path(input_sampling_pattern)
        if not input_sampling_pattern.is_file():
            err = f"{input_sampling_pattern.as_posix()} not a file"
            log_module.error(err)
            raise ValueError(err)
        sampling_pattern = torch.squeeze(load_sampling_pattern(input_sampling_pattern))
        # if shape not match ks_space assume its [x, y, t]
        while sampling_pattern.shape.__len__() < k_space_data.shape.__len__():
            sampling_pattern = sampling_pattern.unsqueeze(2)
    else:
        # assume z and channels all sampled with same pattern, get a 3d thing
        sampling_pattern = torch.ones_like(k_space_data[:, :, 0, 0, :]).to(torch.int)
        mid_slice = int(k_space_data.shape[2] / 2)
        sampling_pattern[torch.abs(k_space_data[:, :, mid_slice, 0, :]) < 1e-8] = 0

    return k_space_data, k_affine, sampling_pattern


def save_data(out_path: typing.Union[str, plib.Path], name: str, data: typing.Union[torch.tensor, np.ndarray],
              affine: typing.Union[torch.tensor, np.ndarray]):
    # strip file ending
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".nii"):
        name = name[:-4]
    # swap dots in name
    name = name.replace(".", "p")
    # build path
    nii_out = out_path.joinpath(name).with_suffix(".nii")
    if out_path.suffixes:
        out_path = out_path.parent
    out_path.mkdir(parents=True, exist_ok=True)
    # check datatype
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(affine):
        affine = affine.numpy()
    # if data complex split into mag and phase
    if np.iscomplexobj(data):
        mag = np.abs(data)
        phase = np.angle(data)
        save_dat = [mag, phase]
        save_labels = ["_mag", "_phase"]
    else:
        save_dat = [data]
        save_labels = [""]
    # save
    for dat_idx in range(save_dat.__len__()):
        f_name = nii_out.with_stem(f"{nii_out.stem}{save_labels[dat_idx]}")
        log_module.info(f"Writing file: {f_name.as_posix()}")
        img = nib.Nifti1Image(save_dat[dat_idx], affine)
        nib.save(img, f_name.as_posix())


def get_shepp_logan_phantom(x_dim: int, y_dim: int, snr: float = 0.0) -> (torch.tensor, torch.tensor):
    # load phantom
    log_module.info("load shepp-logan phantom")
    sl_phantom = ski.data.shepp_logan_phantom()
    sl_phantom = ski.transform.resize(sl_phantom, (x_dim, y_dim))
    sl_phantom = torch.from_numpy(sl_phantom).to(torch.complex128)
    sl_k_space = torch.fft.fftshift(torch.fft.fft2(sl_phantom))
    # snr 0 disables noise sampling
    noise = 0.0
    if snr > 1e-5:
        n_amp = torch.max(torch.abs(sl_k_space)) / snr
        noise = 2 * (torch.rand(size=(x_dim, y_dim)) - 0.5) + 2j * (torch.rand(size=(x_dim, y_dim)) - 0.5)
        noise *= n_amp
    sl_k_space += noise
    # reconsturct from noisy k space
    sl_phantom = torch.fft.ifftshift(torch.fft.ifft2(sl_k_space))
    return sl_k_space, sl_phantom
