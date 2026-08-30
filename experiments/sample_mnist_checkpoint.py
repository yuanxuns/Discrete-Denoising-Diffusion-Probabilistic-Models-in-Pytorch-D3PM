"""Generate a large conditional MNIST batch from a completed D3PM checkpoint.

Example:
    python -m experiments.sample_mnist_checkpoint \
      --checkpoint artifacts/checkpoints/mnist/uniform_seed0.pt \
      --device cuda --samples-per-class 1000 --sample-batch-size 32
"""

import argparse
from pathlib import Path

import torch

from experiments.run_transition_comparison import sample_in_batches, save_samples
from models.d3pm import D3PM
from models.dit import DiT


def load_model_and_d3pm(checkpoint_path, device, hidden_size, depth, num_heads):
    """Restore a DiT and D3PM process from a final training checkpoint.

    Args:
        checkpoint_path: Path to a checkpoint written by the experiment runner.
        device: Target torch device.
        hidden_size: Fallback DiT width for checkpoints created before metadata.
        depth: Fallback DiT depth for old checkpoints.
        num_heads: Fallback number of attention heads for old checkpoints.

    Returns:
        Tuple ``(model, d3pm, checkpoint)`` on ``device``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DiT(
        input_shape=tuple(checkpoint["input_shape"]),
        num_classes=checkpoint["K"],
        num_timesteps=checkpoint["timesteps"],
        hidden_size=checkpoint.get("hidden_size", hidden_size),
        depth=checkpoint.get("depth", depth),
        num_heads=checkpoint.get("num_heads", num_heads),
        condition_classes=checkpoint.get("condition_classes", 10),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    d3pm = D3PM(
        betas=checkpoint["betas"],
        model_prediction_type="x_start",
        logits_type="logits",
        transition_matrix_type=checkpoint["transition"],
        transition_bands=checkpoint.get("transition_bands"),
        loss_type="hybrid",
        hybrid_coeff=checkpoint.get("hybrid_coeff", 0.001),
        K=checkpoint["K"],
    ).to(device)
    return model, d3pm, checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-class", type=int, default=1_000)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="artifacts/mnist_samples")
    # Fallbacks let this utility read checkpoints written before architecture
    # metadata was added to the experiment runner.
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, d3pm, checkpoint = load_model_and_d3pm(
        args.checkpoint, device, args.hidden_size, args.depth, args.heads
    )
    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    samples = sample_in_batches(
        d3pm, model, labels, (28, 28), args.sample_batch_size, args.cfg_scale
    )
    transition, seed = checkpoint["transition"], checkpoint["seed"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{transition}_seed{seed}"
    image_path = output_dir / f"{stem}.png"
    tensor_path = output_dir / f"{stem}.pt"
    save_samples(samples, "mnist", image_path, transition, checkpoint["K"])
    torch.save(
        {
            "samples": samples.cpu(),
            "labels": labels.cpu(),
            "K": checkpoint["K"],
            "transition": transition,
            "seed": seed,
            "checkpoint": str(args.checkpoint),
            "cfg_scale": args.cfg_scale,
        },
        tensor_path,
    )
    print(f"Wrote {image_path} and {tensor_path}")


if __name__ == "__main__":
    main()
