import torch

from models.d3pm import D3PM


def make_model():
    betas = torch.linspace(1e-4, 0.02, 5, dtype=torch.float32)
    return D3PM(
        betas=betas,
        model_prediction_type="x_start",
        logits_type="logits",
        transition_matrix_type="uniform",
        transition_bands=2,
        loss_type="kl",
        hybrid_coeff=0.5,
        K=8,
    )


def test_prior_sample_uses_state_space_size():
    d3pm = make_model()
    sample = d3pm.prior_sample((2, 3), torch.device("cpu"))

    assert sample.shape == (2, 3)
    assert torch.all(sample >= 0)
    assert torch.all(sample < d3pm.K)


def test_p_sample_uses_provided_timestep():
    d3pm = make_model()
    x = torch.tensor([0, 1], dtype=torch.int64)
    t = torch.tensor([0, 3], dtype=torch.int64)
    noise = torch.rand((2, d3pm.K), dtype=torch.float32)
    seen = {}

    def fake_p_logits(model, *, x, t):
        seen["t"] = t
        return (
            torch.zeros((x.shape[0], d3pm.K), dtype=torch.float32),
            torch.zeros((x.shape[0], d3pm.K), dtype=torch.float32),
        )

    d3pm.p_logits = fake_p_logits
    d3pm.p_sample(model=None, x=x, t=t, noise=noise)

    assert torch.equal(seen["t"], t)
