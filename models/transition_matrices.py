import numpy as np
import scipy.special
import torch


def compute_transition_matrices(
    betas,
    transition_matrix_type,
    K,
    transition_bands=None,
    mask_id=None,
):
    """
    Build the full forward transition tensor for every diffusion timestep.

    Args:
        betas: Tensor or array-like of shape (T,) storing the per-timestep diffusion rates.
        transition_matrix_type: One of {"uniform", "gaussian", "absorbing"}.
        K: Number of discrete states.
        transition_bands: Optional local-band width for structured kernels.
        mask_id: Absorbing state index required when using the absorbing kernel.

    Returns:
        Tensor of shape (T, K, K), where each matrix Q_t defines q(x_t | x_{t-1}).
    """
    if transition_matrix_type == "uniform":
        q_mats = [
            get_uniform_transition_matrix(betas, t, K, transition_bands)
            for t in range(len(betas))
        ]
    elif transition_matrix_type == "gaussian":
        q_mats = [
            get_gaussian_transition_matrix(betas, t, K, transition_bands)
            for t in range(len(betas))
        ]
    elif transition_matrix_type == "absorbing":
        if mask_id is None:
            raise ValueError(
                "mask_id must be provided for absorbing transition matrices."
            )
        q_mats = [
            get_absorbing_transition_matrix(betas, t, K, mask_id)
            for t in range(len(betas))
        ]
    else:
        raise ValueError(
            f"Unknown transition_matrix_type: {transition_matrix_type}"
        )
    return torch.stack(q_mats, dim=0)


def get_uniform_transition_matrix(betas, t, K, transition_bands=None):
    """
    Construct the uniform discrete diffusion matrix at timestep t.

    Args:
        betas: Tensor of shape (T,) containing diffusion coefficients.
        t: Integer timestep index.
        K: Number of discrete states.
        transition_bands: Optional local neighborhood size for the transition band.

    Returns:
        Tensor of shape (K, K) representing the uniform transition kernel.
    """
    beta_t = betas[t].detach().cpu().item()

    if transition_bands is None:
        mat = np.full(
            shape=(K, K), fill_value=beta_t / float(K), dtype=np.float64
        )
        diag_indices = np.diag_indices_from(mat)
        diag_val = 1.0 - beta_t * (K - 1.0) / float(K)
        mat[diag_indices] = diag_val
        return torch.from_numpy(mat)

    assert transition_bands < K and transition_bands > 0, (
        "transition_bands must be less than K and greater than 0."
    )
    mat = np.zeros((K, K), dtype=np.float64)
    off_diag_slice = np.full(
        shape=(K - 1),
        fill_value=beta_t / float(transition_bands + 1),
        dtype=np.float64,
    )
    for k in range(1, transition_bands + 1):
        mat += np.diag(off_diag_slice, k=k)
        mat += np.diag(off_diag_slice, k=-k)
        off_diag_slice = off_diag_slice[:-1]
    diag = 1.0 - mat.sum(axis=1)
    mat += np.diag(diag, k=0)
    return torch.from_numpy(mat)


def get_gaussian_transition_matrix(betas, t, K, transition_bands=None):
    """
    Construct the Gaussian-like discrete diffusion matrix at timestep t.

    Args:
        betas: Tensor of shape (T,) containing diffusion coefficients.
        t: Integer timestep index.
        K: Number of discrete states.
        transition_bands: Optional number of nearby states to include in the kernel.

    Returns:
        Tensor of shape (K, K) representing the Gaussian-style transition kernel.
    """
    beta_t = betas[t].detach().cpu().item()
    transition_bands = (
        transition_bands if transition_bands is not None else K - 1
    )
    mat = np.zeros((K, K), dtype=np.float64)

    values = np.linspace(
        start=0.0, stop=K - 1, num=K, endpoint=True, dtype=np.float64
    )
    values = values * 2.0 / (K - 1.0)
    values = values[: transition_bands + 1]
    values = -values * values / beta_t

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values)
    values = values[transition_bands:]
    for k in range(transition_bands + 1):
        off_diag_slice = np.full(
            shape=(K - k), fill_value=values[k], dtype=np.float64
        )
        mat += np.diag(off_diag_slice, k=k)
        mat += np.diag(off_diag_slice, k=-k)

    diag = 1.0 - mat.sum(axis=1)
    mat += np.diag(diag, k=0)
    return torch.from_numpy(mat)


def get_absorbing_transition_matrix(betas, t, K, mask_id):
    """
    Construct the absorbing-state transition matrix at timestep t.

    Args:
        betas: Tensor of shape (T,) containing diffusion coefficients.
        t: Integer timestep index.
        K: Number of discrete states.
        mask_id: Absorbing state index.

    Returns:
        Tensor of shape (K, K) representing the absorbing-state kernel.
    """
    beta_t = betas[t].detach().cpu().item()
    diag = np.full(shape=(K), fill_value=1.0 - beta_t, dtype=np.float64)
    mat = np.diag(diag, k=0)
    mat[:, mask_id] += beta_t
    return torch.from_numpy(mat)
