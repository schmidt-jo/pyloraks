import typing
import torch
import matplotlib.pyplot as plt

plt.style.use("ggplot")
import logging
import pathlib as plib
import plotly.express as px
import itertools
import pandas as pd

log_module = logging.getLogger(__name__)


def plot_img(img_tensor: torch.tensor, log_mag: bool = False,
             out_path: plib.Path = plib.Path(__file__).absolute().parent, name: str = "", supress_log: bool = False):
    # in case input has no time dimension
    if img_tensor.shape.__len__() < 3:
        plot_data = torch.zeros((2, *img_tensor.shape), dtype=torch.float64)
        # separate complex data - plot magnitude logarithmically eg for k-space data
        plot_data[0] = torch.abs(img_tensor)
        if log_mag:
            plot_data[0] = torch.log(plot_data[0])
        plot_data[1] = torch.angle(img_tensor)
        fig = px.imshow(plot_data, facet_col=0)
    else:
        # assume time dimension is first dimension
        plot_data = torch.zeros((img_tensor.shape[0], 2, *img_tensor.shape[1:]), dtype=torch.float64)
        # separate complex data - plot magnitude logarithmically eg for k-space data
        plot_data[:, 0] = torch.abs(img_tensor)
        if log_mag:
            plot_data[:, 0] = torch.log(plot_data[:, 0])
        plot_data[:, 1] = torch.angle(img_tensor)
        fig = px.imshow(plot_data, facet_col=1, animation_frame=0)
    if log_mag:
        text = ["log mag", "phase"]
    else:
        text = ["mag", "phase"]
    # set facet titles
    for i, txt in enumerate(text):
        fig.layout.annotations[i]['text'] = txt
    # hide ticklabels
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    # update traces to use different coloraxis
    for i, t in enumerate(fig.data):
        t.update(coloraxis=f"coloraxis{i + 1}")

    # update layout --> plot colorbar for each trace, define colorscale
    fig.update_layout(
        coloraxis={
            "colorbar": {
                "x": 0.475,
                "y": 0.5,
                "len": 1,
                'title': 'log mag'
            },
            "colorscale": "Inferno"
        },
        coloraxis2={
            "colorbar": {
                "x": 1,
                "y": 0.5,
                "len": 1,
                'title': 'Phase'

            },
            "colorscale": 'Viridis'
        }
    )

    name = name.replace(".", "p")
    file_name = out_path.joinpath(f"plot_{name}").with_suffix(".html")
    if supress_log:
        log_module.debug(f"\t\t-writing file: {file_name}")
    else:
        log_module.info(f"\t\t-writing file: {file_name}")
    fig.write_html(file_name.as_posix())


def plot_k2c(k_space: torch.tensor, c: torch.tensor, s_mat: torch.tensor,
             f_path: typing.Union[str, plib.Path], name: str = ""):
    # make sure device is right
    k_space = k_space.clone().detach().cpu()
    c = c.clone().detach().cpu()
    s_mat = s_mat.clone().detach().cpu()
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 0.1])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("c")
    img = ax.imshow(torch.abs(c), aspect=2e-3, clim=(0, 5e-4), interpolation="None")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("k_space reco")
    ax.imshow(torch.log(torch.abs(k_space)), interpolation="None")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("k_space pt count")
    ax.imshow(torch.abs(s_mat), interpolation="None")

    cb_ax = fig.add_subplot(gs[0, -1])
    plt.colorbar(img, cax=cb_ax)

    f_path = plib.Path(f_path.absolute())
    file_name = f_path.joinpath(f"sl_C2K_{name}").with_suffix(".png")
    logging.info(f"\t\t- writing plot file: {file_name}")
    plt.savefig(file_name.as_posix(), bbox_inches="tight")


def plot_k_img_recon(k_space_recon: torch.tensor, fig_path: typing.Union[str, plib.Path], name: str = "",
                     static_image: bool = False):
    img_recon = torch.fft.fft2(torch.fft.fftshift(k_space_recon))

    # normalizing for plot
    plot_rd_k_mag = torch.log(torch.abs(k_space_recon))
    plot_rd_k_mag /= torch.max(torch.abs(plot_rd_k_mag))
    plot_rd_k_phase = torch.abs(torch.angle(k_space_recon)) / torch.pi
    plot_rd_img_mag = torch.abs(img_recon) / torch.max(torch.abs(img_recon))
    plot_rd_img_phase = torch.abs(torch.angle(img_recon)) / torch.pi

    rd_plot_data = torch.row_stack(
        [plot_rd_k_mag[None, :], plot_rd_k_phase[None, :],
         plot_rd_img_mag[None, :], plot_rd_img_phase[None, :]]
    )

    fig = px.imshow(rd_plot_data, facet_col=0)
    fig.update_yaxes(matches=None)
    fig_path = plib.Path(fig_path).absolute()
    if static_image:
        file_name = fig_path.joinpath(f"rd_recon_{name}").with_suffix(".png")
        fig.write_image(file_name)
    else:
        file_name = fig_path.joinpath(f"rd_recon_{name}").with_suffix(".html")
        fig.write_html(file_name)
    logging.info(f"\t\t- writing plot file: {file_name}")


def plot_svd_lines(svds: list, fig_path: typing.Union[str, plib.Path], static_image: bool = False):
    in_r = [3] * svds[0].shape[0] + [4] * svds[1].shape[0] + [5] * svds[2].shape[0]
    in_idxs = list(itertools.chain(*[torch.arange(svds[k].shape[0]).tolist() for k in range(3)]))
    in_data = list(itertools.chain(*svds))

    df = pd.DataFrame({
        "R": in_r, "index": in_idxs, "SVD value": in_data
    })

    fig = px.line(df, x="index", y="SVD value", color="R")

    if static_image:
        file_name = fig_path.joinpath("svd_c").with_suffix(".png")
        fig.write_image(file_name)
    else:
        file_name = fig_path.joinpath("svd_c").with_suffix(".html")
        fig.write_html(file_name)
    logging.info(f"\t\t- writing plot file: {file_name}")


def plot_nullspace_rep(k: torch.tensor, fig_path: typing.Union[str, plib.Path],
                       name: str = "", static_image: bool = False):
    # need to sort as k-space image
    # k_space = torch.zeros((nv.shape[0], *k_space_dims), dtype=torch.complex128)
    # m_idxs = torch.arange(torch.prod(torch.tensor(k_space_dims) - 2 * radius).item())
    # x = radius + torch.floor(m_idxs / (k_space_dims[0] - 2 * radius)).to(torch.int)
    # y = radius + (m_idxs % (k_space_dims[1] - 2 * radius)).to(torch.int)
    # for k in range(8):
    #     k_space[k, x, y] = nv[k]
    log_module.info(f"plot nullspace img recon")
    # reconstruct image
    img = torch.fft.ifft2(k, norm="ortho")
    # plot
    plot_data = torch.abs(img)
    max_val = torch.max(plot_data).item() * 0.7
    fig = px.imshow(plot_data, zmax=max_val, facet_col=0, facet_col_wrap=4, width=1600, height=900)

    fig_path = plib.Path(fig_path).absolute()
    if static_image:
        file_name = fig_path.joinpath(f"nullspace_recon_{name}").with_suffix(".png")
        fig.write_image(file_name)
    else:
        file_name = fig_path.joinpath(f"nullspace_recon_{name}").with_suffix(".html")
        fig.write_html(file_name)
    logging.info(f"\t\t- writing plot file: {file_name}")
