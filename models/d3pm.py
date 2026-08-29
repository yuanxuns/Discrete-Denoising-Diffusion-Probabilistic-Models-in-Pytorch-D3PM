from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from models.d3pm_utils import (
    categorical_kl_logits,
    categorical_kl_probs,
    categorical_log_likelihood,
    gumbel_argmax,
    log_min_exp,
    meanflat,
)
from models.transition_matrices import compute_transition_matrices


class D3PM(nn.Module):
    """
    Noisy data is labeled x_0, ..., x_(T-1), and the original data is labeled as x_start (x_(-1)).
    """

    betas: torch.Tensor
    q_mats: torch.Tensor
    q_bar_mats: torch.Tensor
    q_T_mats: torch.Tensor
    bin_centers: torch.Tensor

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
        eps=1.0e-6,
    ):
        super().__init__()
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

        # Register and dimension update for the forward transition kernels.
        # q_mats: (T, K, K), where T = num_timesteps, K = number of states.
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

        # q_bar_mats: cumulative transition from x_0 to x_t, shape (T, K, K).
        q_bar_mat_t = self.q_mats[0]
        q_bar_mats = [q_bar_mat_t]
        for t in range(1, self.num_timesteps):
            # Row-vector convention: q(x_t | x_start) = Q_0 ... Q_t.
            q_bar_mat_t = torch.matmul(q_bar_mats[-1], self.q_mats[t])
            q_bar_mats.append(q_bar_mat_t)
        self.register("q_bar_mats", torch.stack(q_bar_mats, dim=0))
        assert self.q_bar_mats.shape == (
            self.num_timesteps,
            self.K,
            self.K,
        )

        # q_T_mats: transposed kernels for efficient sampling from q(x_t | x_{t-1}), shape (T, K, K).
        self.register("q_T_mats", torch.transpose(self.q_mats, 1, 2))
        assert self.q_T_mats.shape == (
            self.num_timesteps,
            self.K,
            self.K,
        )

        self.register_buffer(
            "bin_centers", torch.linspace(-1.0, 1.0, self.K), persistent=False
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
        mixing_error = float(
            (self.q_bar_mats[-1] - prior[None, :]).abs().max()
        )
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

    def _at(self, a, t, x):
        """
        Select row-wise transition probabilities for each batch sample at time index t.

        Args:
            a: Tensor of shape (T, K, K), representing the transition matrices.
            t: Integer timestep tensor of shape (batch_size,).
            x: State tensor of shape (batch_size, ...), where each entry is in [0, K-1].

        Returns:
            Tensor of shape (batch_size, ..., K), where the last dimension stores the
            conditional distribution over the next state for each sample.
        """

        batch_size = x.shape[0]
        a_t = torch.index_select(a, dim=0, index=t)
        assert a_t.shape == (batch_size, self.K, self.K)

        # idx: (batch_size, N) -> (batch_size, N, 1) -> (batch_size, N, K)
        idx = (
            x.reshape(batch_size, -1)
            .to(torch.int64)
            .unsqueeze(-1)
            .expand(-1, -1, self.K)
        )
        # out: (batch_size, N, K)
        # out[b, n, k] = a_t[b, x[b, n], k]
        out = torch.gather(a_t, dim=1, index=idx)

        return out.reshape(*x.shape, self.K)

    def _at_onehot(self, a, t, x):
        """
        Apply the selected transition matrices to a one-hot-like batch tensor.

        Args:
            a: Tensor of shape (T, K, K), representing transition matrices.
            t: Integer timestep tensor of shape (batch_size,).
            x: Tensor of shape (batch_size, ..., K), where the last dimension is a
                categorical probability vector or one-hot-like soft assignment.

        Returns:
            Tensor of shape (batch_size, ..., K), obtained by multiplying each distributed
            state representation by the selected transition matrix.
        """

        batch_size = x.shape[0]
        a_t = torch.index_select(a, dim=0, index=t)
        assert a_t.shape == (batch_size, self.K, self.K)

        # out: (batch_size, ..., K)
        out = torch.matmul(x.reshape(batch_size, -1, self.K), a_t)

        return out.reshape(*x.shape)

    def sample_timesteps(self, bs, device):
        """
        Uniformly sample a batch of timesteps.

        Args:
            bs: Integer batch size.
            device: Torch device on which to generate the timesteps.

        Returns:
            Tensor of shape (bs,), containing sampled integer timesteps in [0, T-1].
        """
        return torch.randint(
            low=0, high=self.num_timesteps, size=(bs,), device=device
        )

    def to_internal_t(self, frac, bs, device):
        """
        Convert a scalar fraction in [0, 1] into an internal time index.

        Args:
            frac: Fractional time tensor or float in [0, 1].
            bs: Batch size used to expand the converted index.
            device: Torch device for the generated indices.

        Returns:
            Tensor of shape (bs,), with integer timesteps in [0, T-1].
        """
        idx = torch.floor(frac * self.num_timesteps).to(torch.int64)
        idx = torch.clamp(idx, min=0, max=self.num_timesteps - 1)
        return idx.expand(bs).to(device)

    def q_probs(self, x_start, t):
        """
        Compute the forward noising distribution q(x_t | x_start).

        Args:
            x_start: Initial state tensor of shape (batch_size, ...), each index in [0, K-1].
            t: Integer timestep tensor of shape (batch_size,).

        Returns:
            Tensor of shape (batch_size, ..., K), where the last dimension stores the
            categorical distribution over states at timestep t conditioned on x_start.
        """
        return self._at(self.q_bar_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """
        Sample a noisy state x_t from q(x_t | x_start) using the Gumbel-max trick.

        Args:
            x_start: Clean state tensor of shape (batch_size, ...), values in [0, K-1].
            t: Integer timestep tensor of shape (batch_size,).
            noise: Uniform noise tensor of shape (batch_size, ..., K), in [0, 1].

        Returns:
            Tensor of shape (batch_size, ...), representing the sampled noisy states at timestep t.
        """
        assert noise.shape == (*x_start.shape, self.K)

        logits = torch.log(self.q_probs(x_start, t) + self.eps)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def _get_logits_from_logistic_pars(self, loc, log_scale):
        """
        Convert loc/log_scale parameters of a discretized logistic model into categorical logits.

        Args:
            loc: Tensor of shape (batch_size, ...), representing the logistic centers.
            log_scale: Tensor of shape (batch_size, ...), representing log-standard-deviation.

        Returns:
            Tensor of shape (batch_size, ..., K) of logits for each discrete state.
        """

        # (batch_size, ...) -> (batch_size, ..., 1)
        loc = torch.unsqueeze(loc, dim=-1)
        log_scale = torch.unsqueeze(log_scale, dim=-1)

        inv_scale = torch.exp(-log_scale + 2.0)
        bin_width = 2.0 / (self.K - 1.0)
        # (K) -> (1, ..., K)
        bin_centers = self.bin_centers.view(*([1] * (len(loc.shape) - 1)), -1)

        bin_centers = bin_centers - loc
        log_cdf_minus = F.logsigmoid(
            (bin_centers - bin_width / 2.0) * inv_scale
        )
        log_cdf_plus = F.logsigmoid((bin_centers + bin_width / 2.0) * inv_scale)

        logits = log_min_exp(log_cdf_plus, log_cdf_minus, eps=self.eps)
        return logits

    def q_posterior_logits(self, x_start, x_t, t, is_x_start_logits):
        """
        Compute the posterior logits q(x_{t-1} | x_t, x_start).

        Args:
            x_start: Tensor of shape (batch_size, ...) or (batch_size, ..., K) when
                is_x_start_logits=True.
            x_t: Noisy state tensor of shape (batch_size, ...).
            t: Integer timestep tensor of shape (batch_size,).
            is_x_start_logits: Whether x_start is provided in logits space instead of state-index space.

        Returns:
            Tensor of shape (batch_size, ..., K) containing posterior logits.
        """

        if is_x_start_logits:
            assert x_start.shape == (*x_t.shape, self.K)
        else:
            assert x_start.shape == x_t.shape

        # fact1[..., v] = q(x_t|x_(t-1)=v) = q_mats[t][v, x_t]
        fact1 = self._at(self.q_T_mats, t, x_t)

        if is_x_start_logits:
            t_minus_one = torch.clamp(t - 1, min=0)
            # fact2[..., v] = q(x_(t-1)=v|x_start)
            #               = x_start * q_bar_mats[t-1][x_start, v]
            fact2 = self._at_onehot(
                self.q_bar_mats, t_minus_one, F.softmax(x_start, dim=-1)
            )
            # p(x_start|x_0, x_start)
            tzero_logits = x_start
        else:
            t_minus_one = torch.clamp(t - 1, min=0)
            # fact2[..., v] = q(x_(t-1)=v|x_start)
            #               = x_start * q_bar_mats[t-1][x_start, v]
            fact2 = self._at(self.q_bar_mats, t_minus_one, x_start)
            # p(x_start|x_0, x_start)
            tzero_logits = torch.log(
                F.one_hot(x_start.to(torch.int64), num_classes=self.K)
                + self.eps
            )

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = torch.reshape(
            t, ([out.shape[0]] + [1] * (len(out.shape) - 1))
        )
        return torch.where(t_broadcast == 0, tzero_logits, out)

    def model_x_start_logits(self, model, x, t, model_kwargs=None):
        """
        Predict the clean-state logits from the denoising network.

        Args:
            model: The denoising model, usually taking (x, t) and returning logits or logistic parameters.
            x: Input state tensor of shape (batch_size, ...).
            t: Integer timestep tensor of shape (batch_size,).

        Returns:
            Tensor of shape (batch_size, ..., K) representing the predicted clean-state logits.
        """
        assert t.shape == (x.shape[0],)

        model_output = model(x, t, **(model_kwargs or {}))
        if self.logits_type == "logits":
            return model_output
        elif self.logits_type == "logistic_pars":
            loc, log_scale = model_output
            return self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise ValueError(
                f"Unknown logits_type: {self.logits_type}. Must be 'logits' or 'logistic_pars'."
            )

    def p_logits(self, model, *, x, t, model_kwargs=None):
        """
        Compute the model posterior logits p_theta(x_{t-1} | x_t).

        Args:
            model: Denoising network used to predict the clean state distribution.
            x: Noisy state tensor of shape (batch_size, ...).
            t: Integer timestep tensor of shape (batch_size,).

        Returns:
            A tuple (model_logits, pred_x_start_logits), each shaped (batch_size, ..., K).
            model_logits is the posterior distribution over x_{t-1} after applying the
            one-step posterior correction, while pred_x_start_logits is the direct prediction
            of the clean state logits before the posterior correction.
        """
        # Register and dimension update for the denoising model output.
        # model_logits: (batch_size, ..., K), pred_x_start_logits: (batch_size, ..., K)
        model_logits = self.model_x_start_logits(model, x, t, model_kwargs)

        if self.model_prediction_type == "x_start":
            pred_x_start_logits = model_logits
            t_broadcast = torch.reshape(
                t,
                ([model_logits.shape[0]] + [1] * (len(model_logits.shape) - 1)),
            )
            model_logits = torch.where(
                t_broadcast == 0,
                pred_x_start_logits,
                self.q_posterior_logits(
                    x_start=pred_x_start_logits,
                    x_t=x,
                    t=t,
                    is_x_start_logits=True,
                ),
            )
        elif self.model_prediction_type == "x_prev":
            raise NotImplementedError(
                "model_prediction_type='x_prev' is not implemented yet."
            )
        else:
            raise ValueError(
                f"Unknown model_prediction_type: {self.model_prediction_type}. Must be 'x_start' or 'x_prev'."
            )

        assert (
            model_logits.shape
            == (*x.shape, self.K)
            == pred_x_start_logits.shape
        )

        return model_logits, pred_x_start_logits

    def cum_transition_matrix(self, s, t):
        """
        Compute the cumulative transition matrix from timestep s to timestep t.

        Args:
            s: Start timestep index, where -1 means the initial state distribution.
            t: End timestep index, with s < t < T.

        Returns:
            Tensor of shape (K, K) representing the cumulative matrix q(x_t | x_s).
        """
        assert -1 <= s < t < self.num_timesteps, (
            f"Invalid timesteps: s={s}, t={t}, num_timesteps={self.num_timesteps}"
        )

        if s < 0:
            return self.q_bar_mats[t]

        mat = self.q_mats[s + 1]
        for k in range(s + 2, t + 1):
            mat = mat @ self.q_mats[k]
        return mat

    def q_posterior_logits_strided(self, x_start_logits, x_t, t, s):
        r"""
        Compute the logits of q(x_s | x_t, x_start) for an arbitrary jump s < t.

        The derivation is:
            q(x_s | x_t, x_start) = q(x_s, x_t | x_start) / q(x_t | x_start)
                = q(x_t | x_s, x_start) * q(x_s | x_start) / q(x_t | x_start)
                \propto q(x_t | x_s) * q(x_s | x_start)
                = Q_(s->t)(x_s, x_t) * QBar_s(x_start, x_s)

        Args:
            x_start_logits: Predicted clean-state logits of shape (batch_size, ..., K).
            x_t: Current noisy state tensor of shape (batch_size, ...).
            t: Current timestep index, integer tensor of shape (batch_size,).
            s: Target previous timestep index, with s < t.

        Returns:
            Tensor of shape (batch_size, ..., K) containing the strided posterior logits.
        """
        assert -1 <= s < t < self.num_timesteps, (
            f"Invalid timesteps: s={s}, t={t}, num_timesteps={self.num_timesteps}"
        )

        if s < 0:
            return x_start_logits
        # q(x_t | x_s)
        q_st = self.cum_transition_matrix(s, t)
        # fact1[..., v] = q(x_t | x_s = v), select x_t-th column of q_st
        fact1 = q_st.t().continuous()[x_t.to(torch.int64)]
        # fact2[..., v] = sum_(x_start) p(x_start) * q(x_s = v| x_start)
        fact2 = torch.matmul(
            F.softmax(x_start_logits, dim=-1), self.q_bar_mats[s]
        )
        return torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

    # ====== Sampling ======

    def prior_sample(self, shape, device):
        """
        Draw x_T from the stationary distribution of the forward process.

        Args:
            shape: Tuple describing the sample shape, e.g. (batch_size, ...).
            device: Torch device used to allocate the sampled state tensor.

        Returns:
            Integer tensor of shape `shape` with state values in [0, K-1].
        """

        if self.transition_matrix_type in ["gaussian", "uniform"]:
            return torch.randint(
                low=0,
                high=self.K,
                size=shape,
                dtype=torch.int64,
                device=device,
            )
        elif self.transition_matrix_type == "absorbing":
            return torch.full(
                shape, self.mask_id, dtype=torch.int64, device=device
            )
        else:
            raise ValueError("Undefined transition matrix type")

    def forward_jump(self, x_s, s, t):
        """
        Sample a forward jump from x_s at time s to time t using the cached transition matrix.

        Args:
            x_s: Integer state tensor of shape (batch_size, ...), values in [0, K-1].
            s: Current timestep index.
            t: Target timestep index, where t > s.

        Returns:
            Tensor of shape (batch_size, ...) containing sampled states at time t.
        """
        assert s < t
        mat = self.q_bar_mats[t] if s < 0 else self.cum_transition_matrix(s, t)
        # (batch_size, ..., K)
        probs = mat[x_s.to(torch.int64)]
        # # (batch_size, ...)
        return gumbel_argmax(torch.log(probs + self.eps))

    @torch.no_grad()
    def p_sample(self, model, *, x, t, noise, model_kwargs=None):
        """
        Sample one step from the model p(x_{t-1} | x_t).

        x_t -> model -> p(x_start | x_t) -> p(x_{t-1} | x_t) -> sample x_{t-1}

        Args:
            model: Denoising network used for the reverse transition.
            x: Noisy state tensor of shape (batch_size, ...).
            t: Integer timestep tensor of shape (batch_size,).
            noise: Uniform noise tensor of shape (batch_size, ..., K) in [0, 1].

        Returns:
            A tuple (sample, pred_x_start_probs), where sample has shape (batch_size, ...)
            and pred_x_start_probs has shape (batch_size, ..., K).
        """

        # Register and dimension update for the one-step reverse transition.
        # model_logits: (batch_size, ..., K), pred_x_start_logits: (batch_size, ..., K)
        if model_kwargs is None:
            model_logits, pred_x_start_logits = self.p_logits(
                model=model, x=x, t=t
            )
        else:
            model_logits, pred_x_start_logits = self.p_logits(
                model=model, x=x, t=t, model_kwargs=model_kwargs
            )
        assert noise.shape == model_logits.shape

        # (bs) -> (bs, 1, ..., 1), dim = (1 + len(x.shape))
        nonzero_mask = (
            (t != 0).to(x.dtype).reshape(x.shape[0], *([1] * (len(x.shape))))
        )

        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            model_logits + nonzero_mask * gumbel_noise, dim=-1
        )
        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape

        return sample, F.softmax(pred_x_start_logits, dim=-1)

    @torch.no_grad()
    def p_sample_loop(
        self, model, shape, num_timesteps=None, return_x_T=False,
        model_kwargs=None,
    ):
        device = next(model.parameters()).device
        noise_shape = tuple(shape) + (self.K,)
        x_T = self.prior_sample(tuple(shape), device)
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        x = x_T
        for i in reversed(range(0, num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.int64).to(device)
            x, _ = self.p_sample(
                model=model,
                x=x,
                t=t,
                noise=torch.rand(size=noise_shape).to(x.device),
                model_kwargs=model_kwargs,
            )

        assert x.shape == shape

        if return_x_T:
            return x_T, x
        else:
            return x

    @torch.no_grad()
    def p_sample_loop_strided(
        self,
        model,
        shape,
        num_steps=None,
        device=None,
        greedy_final=True,
        return_intermediates=False,
        resample_r=1,
        resample_jump=1.0,
        model_kwargs=None,
    ):
        if device is None:
            device = next(model.parameters()).device

        if num_steps is None:
            num_steps = self.num_timesteps

        num_steps = max(1, min(int(num_steps), self.num_timesteps))
        repeats = max(1, int(resample_r))

        ts = np.linspace(self.num_timesteps - 1, -1, num_steps + 1)
        ts = np.unique(np.round(ts).astype(int))[::-1]

        x = self.prior_sample(shape, device)
        intermediates = [x]
        nfe = 0
        for i in range(len(ts) - 1):
            t_cur, t_next = int(ts[i]), int(ts[i + 1])
            t_now = t_cur
            for r in range(repeats):
                t_batch = torch.full(
                    (shape[0],), t_now, dtype=torch.int64, device=device
                )
                pred_x_start_logits = self.model_x_start_logits(
                    model, x, t_batch, model_kwargs
                )
                nfe += 1
                # q(x_tnext | x_tnow, x_start)
                logits = self.q_posterior_logits_strided(
                    pred_x_start_logits, x, t_now, t_next
                )
                if t_next < 0 and greedy_final:
                    x_next = torch.argmax(logits, dim=-1)
                else:
                    x_next = gumbel_argmax(logits)

                span = max(1, round(resample_jump * (t_now - t_next)))
                t_back = min(t_cur, t_next + span)

                if r == repeats - 1 or t_next < 0 or t_back <= t_next:
                    x = x_next
                    break

                x = self.forward_jump(x_next, t_next, t_back)
                t_now = t_back

            intermediates.append(x)

        self.last_nfe = nfe
        assert x.shape == shape
        return (x, intermediates) if return_intermediates else x

    # ====== Log Likelihood / Loss Calculation ======
    def vb_terms_bpd(self, model, *, x_start, x_t, t, model_kwargs=None):
        """Calculate specified terms of the variational bound.

        Args:
          model: the denoising network
          x_start: original clean data
          x_t: noisy data
          t: timestep of the noisy data (and the corresponding term of the bound
            to return)

        Returns:
          a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
          (specified by `t`), and `pred_x_start_logits` is logits of
          the denoised image.
        """
        # The logits of q(x_{t-1} | x_t, x_start)
        true_logits = self.q_posterior_logits(
            x_start, x_t, t, is_x_start_logits=False
        )
        # The logits of p_(theta)(x_{t-1} | x_t)
        model_logits, pred_x_start_logits = self.p_logits(
            model, x=x_t, t=t, model_kwargs=model_kwargs
        )

        kl = categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = meanflat(kl) / np.log(2.0)

        decoder_nll = -categorical_log_likelihood(x_start, model_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = meanflat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def prior_bpd(self, x_start):
        """
        Computes KL(q(x_(T-1) | x0) || p(x_(T-1)))
        """

        q_probs = self.q_probs(
            x_start=x_start,
            t=torch.full(
                (x_start.shape[0],),
                self.num_timesteps - 1,
                dtype=torch.int64,
                device=x_start.device,
            ),
        )

        if self.transition_matrix_type in ["gaussian", "uniform"]:
            prior_probs = torch.ones_like(q_probs) / self.K
        elif self.transition_matrix_type == "absorbing":
            absorbing_int = torch.full(
                q_probs.shape[:-1],
                self.mask_id,
                dtype=torch.int64,
                device=q_probs.device,
            )
            prior_probs = F.one_hot(absorbing_int, num_classes=self.K).to(
                q_probs.dtype
            )
        else:
            raise ValueError("Undefined transition matrix type")

        assert prior_probs.shape == q_probs.shape
        kl_prior = categorical_kl_probs(q_probs, prior_probs)
        assert kl_prior.shape == x_start.shape
        return meanflat(kl_prior) / np.log(2.0)

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """
        Negative log-likelihood of the true class == cross entropy.
        Because the target is one hot, - sum_k q_k*log(p_k)
                                    => - log p_theta(x_start)
        """
        ce = -categorical_log_likelihood(x_start, pred_x_start_logits)
        ce = meanflat(ce) / np.log(2.0)
        return ce

    def training_losses(self, model, x_start, t, model_kwargs=None):
        noise = torch.rand(
            size=x_start.shape + (self.K,), device=x_start.device
        )

        # t starts at 0
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.loss_type == "kl":
            losses, _ = self.vb_terms_bpd(
                model=model, x_start=x_start, x_t=x_t, t=t,
                model_kwargs=model_kwargs,
            )

        elif self.loss_type == "cross_entropy_x_start":
            _, pred_x_start_logtits = self.p_logits(
                model=model, x=x_t, t=t, model_kwargs=model_kwargs
            )
            losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logtits
            )
        elif self.loss_type == "hybrid":
            vb_losses, pred_x_start_logtits = self.vb_terms_bpd(
                model=model, x_start=x_start, x_t=x_t, t=t,
                model_kwargs=model_kwargs,
            )
            ce_losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logtits
            )
            losses = vb_losses + self.hybrid_coeff * ce_losses
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == t.shape
        return losses

    @torch.no_grad()
    def calc_bpd_loop(self, model, x_start):
        batch_size = x_start.shape[0]
        noise_shape = x_start.shape + (self.K,)
        vbterms = []

        for t in reversed(range(self.num_timesteps)):
            t_b = torch.full(
                (batch_size,), t, dtype=torch.int64, device=x_start.device
            )
            vb, _ = self.vb_terms_bpd(
                model=model,
                x_start=x_start,
                t=t_b,
                x_t=self.q_sample(
                    x_start=x_start,
                    t=t_b,
                    noise=torch.rand(size=noise_shape, device=x_start.device),
                ),
            )
            vbterms.append(vb)

        vbterms_tb = torch.stack(vbterms, dim=0)
        vbterms_bt = vbterms_tb.T
        assert vbterms_bt.shape == (batch_size, self.num_timesteps)

        prior_b = self.prior_bpd(x_start=x_start)
        total_b = vbterms_tb.sum(dim=0) + prior_b
        return {"total": total_b, "vbterms": vbterms_bt, "prior": prior_b}
