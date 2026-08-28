import torch
from torch.nn import functional as F


def log_min_exp(a, b, eps=1e-6):
    """
    Compute log(exp(a) - exp(b)) (b < a)in a numerically stable way.
    """
    assert torch.all(b < a), "b must be less than a for log_min_exp."
    return a + torch.log1p(-torch.exp(b - a) + eps)


def gumbel_argmax(logits):
    """
    Sample a categorical distribution via the Gumbel-max trick.
    """
    noise = torch.rand_like(logits)
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))
    return torch.argmax(logits + gumbel_noise, dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.0e-6):
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)


def categorical_kl_logits(logits1, logits2, eps=1.0e-6):
    return categorical_kl_probs(
        F.softmax(logits1 + eps, dim=-1), F.softmax(logits2 + eps)
    )


def categorical_log_likelihood(x, logits):
    """
    Inputs:
      x: (bs, ...)
      logits: (bs, ..., K)
    """
    # (bs, ..., K)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, x.to(torch.int64).unsqueeze(-1)).squeeze(-1)


def meanflat(x):
    """
    Take the mean over all dims except the first batch dim.
    """

    return x.mean(dim=tuple(range(1, len(x.shape))))
