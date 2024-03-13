"""
conjugate gradient descent to solve Ax = b, for x
"""
import torch
import tqdm
from pyloraks import fns_ops


def m_1_matrix(f_input_diag: torch.tensor, vvh: torch.tensor, aha: torch.tensor,
               op_x: fns_ops.S | fns_ops.C, dim_s, lam: float = 0.1):
    f_torch = torch.from_numpy(f_input_diag)
    m1_fhf = aha * f_torch
    m1_v = torch.flatten(
        op_x.operator_adjoint(
            torch.matmul(
                op_x.operator(torch.reshape(f_torch, (dim_s, -1))), vvh
            )
        )
    )
    return (m1_fhf - lam * m1_v).numpy()


def cgd(input_a_diagonal: torch.tensor, vector_b: torch.tensor):
    matrix_a_diagonal = m_1_matrix(input_a_diagonal)
    dim = matrix_a_diagonal.shape[0]
    x = torch.rand(dim)
    # get residual, we want to do this vectorized in 1d, the matrix a is assumed to be diagonal
    # and represented as vector
    r = vector_b - matrix_a_diagonal * x
    d = r
    rr = r * r
    xs = [x]

    for i in tqdm.trange(1, dim):
        # matrix multiplication, since a is diagonal matrix we can simply multiply
        Ad = matrix_a_diagonal * d
        alpha = rr / (d * Ad)
        x += alpha * d
        r -= alpha * Ad
        rr_new = r * r
        beta = rr_new / rr
        d = r + beta * d
        rr = rr_new

        xs.append(x)

    return torch.tensor(xs)

