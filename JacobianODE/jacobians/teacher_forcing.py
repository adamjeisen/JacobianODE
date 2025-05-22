import torch
import numpy as np
from scipy.linalg import expm, logm
from .data_utils import compute_lyaps
from tqdm.auto import tqdm
def matrix_product(A, I=None):
    if I is None:
        I = torch.eye(A.shape[-1]).type(A.dtype).to(A.device)
    matrix_prod = torch.zeros(*A.shape[:-3], A.shape[-2], A.shape[-1]).type(A.dtype).to(A.device)
    matrix_prod[:] = I
    for k in range(A.shape[-3] - 1, -1, -1):
        matrix_prod = matrix_prod @ A[..., k, :, :]
    return matrix_prod

def matrix_geometric_mean(A, I=None):
    if I is None:
        I = torch.eye(A.shape[-1]).type(A.dtype).to(A.device)
    matrix_prod = matrix_product(A, I).detach().cpu().numpy()
    # check if any element of matrix_prod is nan or inf
    if np.any(np.isnan(matrix_prod)) or np.any(np.isinf(matrix_prod)):
        return None
    if A.ndim == 3:
        return expm(logm(matrix_prod, disp=False)[0] / (A.shape[-3] - 1))
    elif A.ndim == 4:
        return np.stack([expm(logm(matrix_prod[i], disp=False)[0] / (A.shape[-3] - 1)) for i in range(A.shape[0])])
    elif A.ndim == 5:
        return np.stack([expm(logm(matrix_prod[i][j], disp=False)[0] / (A.shape[-3] - 1)) for i in range(A.shape[0]) for j in range(A.shape[1])])
    else:
        raise ValueError(f"Invalid number of dimensions: {A.ndim}")


def get_alpha_exact(jacs, I=None):
    if I is None:
        I = torch.eye(jacs.shape[-1]).type(jacs.dtype).to(jacs.device)
    geom_mean = matrix_geometric_mean(jacs, I=I)
    if geom_mean is None:
        return 1
    alpha_exact = np.linalg.norm(geom_mean, axis=(-2, -1), ord=2)
    alpha_exact = np.max((np.zeros(alpha_exact.shape), 1 - 1/alpha_exact), axis=0)
    return np.max(alpha_exact)

def matrix_fractional_power(A, p):
    """
    Compute the fractional power (e.g., 1/n) of a matrix using PyTorch.
    
    Args:
        A (torch.Tensor): Input square matrix (n x n).
        p (float): The power to compute (e.g., p = 1/n for the n-th root).
    
    Returns:
        torch.Tensor: The matrix A^(p).
    """
    # Ensure the matrix is on the same device as input
    device = A.device

    # Perform eigen decomposition
    eigvals, eigvecs = torch.linalg.eig(A)  # eigvals are complex
    eigvals = eigvals.to(device)
    eigvecs = eigvecs.to(device)

    # Take the fractional power of the eigenvalues
    # eigvals_root = torch.diag(eigvals ** p)
    # diagonal for batched eigvals
    eigvals_root = torch.diag_embed(eigvals ** p)

    # Reconstruct the matrix
    A_root = eigvecs @ eigvals_root @ torch.linalg.inv(eigvecs)

    # Return real part if the input is real
    if torch.isreal(A).all():
        A_root = A_root.real

    return A_root

def get_alpha_exact_v2(jacs, I=None):
    if I is None:
        I = torch.eye(jacs.shape[-1]).type(jacs.dtype).to(jacs.device)
    matrix_prod = matrix_product(jacs, I=I)
    return torch.max(1 - 1/torch.linalg.norm(matrix_fractional_power(matrix_prod, 1/jacs.shape[-3]), axis=(-2, -1)))

def get_alpha_exact_v3(jacs, I=None):
    if I is None:
        I = torch.eye(jacs.shape[-1]).type(jacs.dtype).to(jacs.device)
    matrix_prod = matrix_product(jacs, I=I)
    return 1 - 1/torch.max(torch.exp((1/jacs.shape[-3])*torch.log(torch.linalg.norm(matrix_prod, axis=(-2, -1)))))

# need to pass in discrete jacobians !!
def get_alpha_explogapprox(jacs):
    angle_mat = torch.zeros_like(jacs)
    angle_mat[jacs < 0] = np.pi
    try:
        alpha_exact = torch.linalg.norm(torch.exp((torch.log(torch.abs(jacs)) + angle_mat*1j).mean(axis=-3)).real, axis=(-2, -1), ord=2)
        alpha_exact = 1 - 1/alpha_exact
    except: # _LinAlgError:
        alpha_exact = torch.ones(1, 1)
    alpha_exact[alpha_exact < 0] = 0
    return torch.max(alpha_exact)

def get_alpha_lyap(jacs):
    alpha = torch.max(1 - 1/torch.exp(compute_lyaps(jacs, dt=1, k=1)))
    if alpha < 0:
        return 0
    return alpha