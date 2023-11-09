import typing
import pathlib as plib
from pyloraks import fns_ops, plotting
import time
import logging
import torch
import skimage as ski
log_module = logging.getLogger(__name__)


def test_loops_and_plot(k_space: torch.tensor, idx_nx_ny: torch.tensor, num_samples: int =1000):
    test_loop(list_r=[3, 4, 5], list_rank=[20, 40, 60], k_space=k_space, idx_nx_ny=idx_nx_ny)
    log_module.info(f"check timing: cpu")
    time_start = time.time()

    test_loop(list_r=[3, 4, 5], list_rank=[20, 40, 60], k_space=k_space, plot=False, idx_nx_ny=idx_nx_ny)

    time_total_cpu = time.time() - time_start

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    log_module.info(f"torch device: {device}")
    log_module.info(f"check timing: gpu")
    time_start = time.time()
    k_space = k_space.to(device)
    idx_nx_ny = torch.zeros(size=(num_samples, 2))
    for k in range(2):
        idx_nx_ny[:, k] = torch.randint(low=0, high=k_space.shape[k], size=(1000,))
    idx_nx_ny = idx_nx_ny.to(device)

    test_loop(idx_nx_ny=idx_nx_ny, list_r=[3, 4, 5], list_rank=[20, 40, 60], k_space=k_space, plot=False)

    time_total_gpu = time.time() - time_start
    log_module.info(f"computation speed: cpu: {time_total_cpu:.2f} s, gpu: {time_total_gpu:.2f} s")


def test_loop(idx_nx_ny: torch.tensor, list_r: list, list_rank: list,
              k_space: torch.tensor, plot: bool = True,
              fig_path: typing.Union[str, plib.Path] = ".external_files/figs_wt"):
    svds = []
    for R in list_r:
        # build C matrix, keep only points with valid mappings (can't map points at edges,
        # since they won't have a complete k-space neighborhood within C )
        C_dev, cropped_nx_ny = fns_ops.operator_p_c(idx_nx_ny, radius=R, k_space=k_space)
        # reconstruct k-space
        k_space_reco_dev, xy_reco, s_mat = fns_ops.operator_p_c_adjoint(C_dev, radius=R, k_space_dims=k_space.shape)
        log_module.debug(f"adjoint test: {torch.allclose(xy_reco, cropped_nx_ny)}")

        if plot:
            plotting.plot_k2c(k_space=k_space_reco_dev, c=C_dev, s_mat=s_mat,
                              f_path=fig_path, name=f"R-{R}_lim-pts-num-{idx_nx_ny.shape[0]}")
        # all possible points within radius
        k_pts = torch.tensor([(x, y) for x in torch.arange(R, k_space.shape[0] - R)
                              for y in torch.arange(R, k_space.shape[1] - R)]).to(torch.int)
        # construct C
        C, cropped_nx_ny = fns_ops.operator_p_c(k_pts, radius=R)
        k_space_reco, xy_reco, S_mat = fns_ops.operator_p_c_adjoint(C, radius=R, k_space_dims=k_space.shape)

        if plot:
            plotting.plot_k2c(k_space=k_space_reco, c=C, s_mat=S_mat,
                              f_path=fig_path, name=f"R-{R}_all-pts")

        # f)
        # construct rank deficient C
        for rank in list_rank:
            log_module.info(f"rank-{rank}")

            # reconstruct c with truncated eigenvalues
            rd_c_recon, s, _ = fns_ops.construct_low_rank_matrix_from_svd(matrix_tensor=C, rank=rank)
            svds.append(s)
            rd_k_space_recon, _, _ = fns_ops.operator_p_c_adjoint(
                c_tensor=rd_c_recon, radius=R, k_space_dims=k_space.shape
            )
            if plot:
                # recon image + plot
                plotting.plot_k_img_recon(k_space_recon=k_space_reco, fig_path=fig_path, name=f"R-{R}_rank-{rank}")
    if plot:
        plotting.plot_svd_lines(svds=svds, fig_path=fig_path)


def test_operators(k_space: torch.tensor, radius: int, fig_path: plib.Path):
    k_space_dims = k_space.shape
    k = torch.flatten(k_space)
    c_matrix = fns_ops.operator_p_c(k, radius=radius, k_space_dims=k_space_dims)
    k_recon, _ = fns_ops.operator_p_c_adjoint(c_tensor=c_matrix, radius=radius, k_space_dims=k_space_dims)
    k_recon = torch.reshape(k_recon, k_space_dims)
    l2 = torch.linalg.norm(k_space - k_recon)
    log_module.info(f"rmse - k-recon: {l2:.2f}, can't reconstruct the corners, "
                    f"since they aren't contained in any neighborhood. Hence it won't give 0 but should be low number")
    plotting.plot_img(torch.reshape(k, k_space_dims), log_mag=True, out_path=fig_path, name="test_in_k_space")
    plotting.plot_img(k_recon, log_mag=True, out_path=fig_path, name="test_in_k_recon")
    # generate undersampling
    # dim s
    s = k.shape[0]
    # set sampling mask - dim l - assume 2.5 times acceleration
    l = int(s / 2.5)
    m_i = torch.randint(k.shape[0], size=(l,)).unique(dim=0)  # rows of S x S identity matrix to keep
    # since we keep only unique choices, update l
    l = m_i.shape[0]
    log_module.debug(f"build subsampling operator f")
    # l x s matrix
    f = torch.eye(s)[m_i]
    log_module.debug(f"build adjoint fh")
    # hermitian adjoint (real valued)
    fh = f.T
    # get the multiplicative - in real space might save computation?!
    log_module.debug(f"build fhf")
    fhf = torch.diagonal(torch.matmul(fh, f))
    # get undersampled k-space data
    k_us = fhf * k
    # same cycle
    c_matrix = fns_ops.operator_p_c(k_us, radius=radius, k_space_dims=k_space_dims)
    k_recon, _ = fns_ops.operator_p_c_adjoint(c_tensor=c_matrix, radius=radius, k_space_dims=k_space_dims)
    plotting.plot_img(torch.reshape(k_us, k_space_dims), log_mag=True, out_path=fig_path, name="test_in_us_k_space")
    plotting.plot_img(torch.reshape(k_recon, k_space_dims), log_mag=True, out_path=fig_path, name="test_in_us_k_recon")


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("load phantom")
    fig_path = plib.Path("./external_files/figs_wt").absolute()
    use_x, use_y = 150, 150
    sl_phantom = ski.data.shepp_logan_phantom()
    sl_phantom = ski.transform.resize(sl_phantom, (use_x, use_y))
    sl_phantom = torch.from_numpy(sl_phantom).to(torch.complex128)
    sl_k_space = torch.fft.fftshift(torch.fft.fft2(sl_phantom))
    r = 4

    # test looping through c creation
    # test_loops_and_plot(k_space=sl_k_space)
    test_operators(sl_k_space, radius=r, fig_path=fig_path)
    # test looping through svd performance tests
    # test_performance_svd(k_space=sl_k_space, radius=r)
