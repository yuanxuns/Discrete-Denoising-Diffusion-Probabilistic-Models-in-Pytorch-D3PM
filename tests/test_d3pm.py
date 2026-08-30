import torch

from models.d3pm import D3PM
from models.dit import DiT
from models.mnist_classifier import MNISTClassifier


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


def test_schedule_diagnostics_returns_scalar_metrics():
    diagnostics = make_model().schedule_diagonostics()

    assert isinstance(diagnostics["mixing_error"], float)
    assert 0 <= diagnostics["t_saturate"] <= 5
    assert 0.0 <= diagnostics["frac_useful"] <= 1.0


def test_mnist_classifier_output_shape():
    logits = MNISTClassifier()(torch.rand(2, 1, 28, 28))

    assert logits.shape == (2, 10)


def test_dit_classifier_free_guidance_endpoints():
    model = DiT(
        input_shape=(2,), num_classes=4, num_timesteps=3, hidden_size=8,
        depth=1, num_heads=2, condition_classes=3,
    ).eval()
    x = torch.tensor([[0, 1], [2, 3]])
    t = torch.tensor([0, 1])
    y = torch.tensor([1, 2])

    conditional = model(x, t, y=y)
    unconditional = model(x, t, y=None)

    assert torch.allclose(model(x, t, y=y, cfg_scale=1.0), conditional)
    assert torch.allclose(model(x, t, y=y, cfg_scale=0.0), unconditional)
    assert torch.allclose(
        model(x, t, y=y, cfg_scale=2.0),
        unconditional + 2.0 * (conditional - unconditional),
    )
