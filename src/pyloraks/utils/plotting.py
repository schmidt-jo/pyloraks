import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub
import torch
import pathlib as plib
import logging

log_module = logging.getLogger(__name__)


def plot_slice(img_data: torch.tensor, name: str, outpath: plib.Path, suffix: str = ".pdf",
               aspect: bool = True, abs_val: bool = True):
    img_data = img_data.detach().cpu().numpy()
    if img_data.shape.__len__() > 2:
        err = f"img has more than 2 dims: {img_data.shape}"
        log_module.error(err)
        raise AttributeError(err)
    if np.iscomplexobj(img_data):
        phase = np.angle(img_data)
        img_data = np.log10(np.abs(np.clip(img_data, 1e-6, np.max(img_data))))
        data = [img_data, phase]
        names = ["mag", "phase"]
        z_ranges = [(np.min(img_data), np.max(img_data)), (-4, 4)]
    else:
        if abs_val:
            img_data = np.abs(img_data)
        data = [img_data]
        names = ["mag"]
        z_ranges = [(np.min(img_data), np.max(img_data))]
    cols = len(data)
    fig = psub.make_subplots(
        rows=1, cols=cols,
        horizontal_spacing=0.01, vertical_spacing=0.01,
        column_titles=names
    )
    for idx_col in range(cols):
        fig.add_trace(
            go.Heatmap(z=data[idx_col], colorscale="Magma", zmin=z_ranges[idx_col][0], zmax=z_ranges[idx_col][1]),
            row=1, col=idx_col+1
        )
        fig.update_xaxes(visible=False, row=1, col=idx_col+1)
        if aspect:
            if idx_col == 0:
                x = "x"
            else:
                x = f"x{idx_col+1}"
            fig.update_yaxes(scaleanchor=x, visible=False, row=1, col=idx_col+1)
        else:
            fig.update_yaxes(visible=False, row=1, col=idx_col+1)
    fig.update_layout(
        width=800 * cols, height=700
    )
    filename = outpath.joinpath(name).with_suffix(suffix)
    log_module.info(f"Writing file: {filename}")
    if suffix == "html" or suffix == ".html":
        fig.write_html(filename.as_posix())
    else:
        fig.write_image(filename.as_posix())


def plot_sop(img_data: torch.tensor, name: str, outpath: plib.Path, suffix: str = ".pdf"):

    if img_data.shape.__len__() > 2:
        err = f"img has more than 2 dims: {img_data.shape}"
        log_module.error(err)
        raise AttributeError(err)
    shape_ax = torch.sort(torch.tensor(img_data.shape), descending=True)
    img_data = torch.permute(img_data, tuple(shape_ax.indices))
    img_data = img_data.detach().cpu().numpy()
    if np.iscomplexobj(img_data):
        err = "only for real data"
        log_module.error(err)
        raise AttributeError(err)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=img_data, colorscale="Magma")
    )
    fig.update_layout(
        width=1200, height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    filename = outpath.joinpath(name).with_suffix(suffix)
    log_module.info(f"Writing file: {filename}")
    if suffix == "html" or suffix == ".html":
        fig.write_html(filename.as_posix())
    else:
        fig.write_image(filename.as_posix())
