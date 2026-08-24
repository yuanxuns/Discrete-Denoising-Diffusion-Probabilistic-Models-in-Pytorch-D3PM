from types import SimpleNamespace

import numpy as np
import torch
import utils
from loguru import logger
from torch import nn
from torch.nn import functional as F

from models.transition_matrices import compute_transition_matrices


class D3PM(nn.Module):
    """
    Noisy data is labeled x_0, ..., x_(T-1), and the original data is labeled as x_start (x_(-1)).
    """

    betas: torch.Tensor
    q_mats: torch.Tensor
    q_bar_mats: torch.Tensor
    q_T_mats: torch.Tensor

    def __init__(
        self,
        *,
        betas,
        model_prediction_type,
        logits_type,
        transition_matrix_type,
        transition_bands,
        loss_type,
        hybrid_coeff,
        K,
        mask_id=None,
        eps=1.0e-5,
    ):
        super().__init__()
        self.betas = betas
        self.num_timesteps = len(betas)
        # "x_start", or "x_prev"
        self.model_prediction_type = model_prediction_type
        # "logits", or "logistic_pars"
        self.logits_type = logits_type
        self.transition_matrix_type = transition_matrix_type
        self.transition_bands = transition_bands
        # "kl", "hybrid", or "cross_entropy_x_start""
        self.loss_type = loss_type
        self.hybrid_coeff = hybrid_coeff
        self.K = K
        self.mask_id = K // 2 if mask_id is None else mask_id
        self.eps = eps

        if not (betas > 0).all() or not (betas < 1).all():
            raise ValueError("All betas must be in the range (0, 1).")

        self.register("betas", betas)

        # Precompute the transition matrices and their products.
        q_mats = compute_transition_matrices(
            betas=betas,
            transition_matrix_type=self.transition_matrix_type,
            K=self.K,
            transition_bands=self.transition_bands,
            mask_id=self.mask_id,
        )
        self.register("q_mats", q_mats)
        assert q_mats.shape == (
            self.num_timesteps,
            self.K,
            self.K,
        )

        q_bar_mat_t = self.q_mats[0]
        q_bar_mats = [q_bar_mat_t]
        for t in range(1, self.num_timesteps):
            q_bar_mat_t = torch.matmul(self.q_mats[t], q_bar_mats[-1])
            q_bar_mats.append(q_bar_mat_t)
        self.register("q_bar_mats", torch.stack(q_bar_mats, dim=0))
        assert self.q_bar_mats.shape == (
            self.num_timesteps,
            self.K,
            self.K,
        )

        self.register("q_T_mats", torch.transpose(self.q_mats, 1, 2))
        assert self.q_T_mats.shape == (
            self.num_timesteps,
            self.K,
            self.K,
        )

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def prior_distribution(self):
        """
        The stationary distribution the sampler starts from. For absorbing transition matrices, this is a one-hot distribution on the absorbing state.
        For uniform and Gaussian transition matrices, this is a uniform distribution over all states.
        """

        if self.transition_matrix_type == "absorbing":
            prior = torch.zeros(
                (self.K,), dtype=torch.float32, device=self.q_mats.device
            )
            prior[self.mask_id] = 1.0
        else:
            prior = torch.full(
                (self.K,),
                fill_value=1.0 / self.K,
                dtype=torch.float32,
                device=self.q_mats.device,
            )
        return prior

    def schedule_diagonostics(
        self, sat_tol=0.01, mix_error_tol=5e-3, frac_tol=0.6, log_warn=False
    ):
        prior = self.prior_distribution()
        mixing_error = float(self.q_bar_mats[-1] - prior[None, :]).abs().max()
        per_row_total_variation = (
            0.5
            * (self.q_bar_mats - prior[None, None, :])
            .abs()
            .sum(-1)
            .max(dim=-1)
            .values
        )
        below = (per_row_total_variation < sat_tol).nonzero()
        t_saturate = int(below[0]) if len(below) > 0 else self.num_timesteps
        frac_useful = t_saturate / self.num_timesteps

        if log_warn:
            text = (
                f"transition mat_type={self.transition_matrix_type}, ",
                f"K={self.K}, transition_bands={self.transition_bands}, ",
                f"num_timesteps={self.num_timesteps}",
            )
            if mixing_error > mix_error_tol:
                logger.warning(
                    f"Mixing error is high: {mixing_error:.4f} ({mixing_error} > {mix_error_tol}). {text}. Consider increasing the number of timesteps/K, or switching to a uniform/absorbing kernel."
                )
            if t_saturate < self.num_timesteps:
                logger.warning(
                    f"Transition matrices saturate at timestep {t_saturate} of {self.num_timesteps} ({frac_useful:.4f}). Timestamps past that carry no information, thus both training and inference become ineffective. {text} Consider decreasing the number of timesteps or a gentler beta schedule."
                )
        return {
            "mixing_error": mixing_error,
            "t_saturate": t_saturate,
            "frac_useful": frac_useful,
        }
