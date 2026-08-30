"""Render generated MNIST samples grouped by their requested condition labels.

Example:
    python -m experiments.plot_mnist_conditional_samples \
      --samples artifacts/mnist_comparison/mnist/uniform_seed0.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_conditional_grid(sample_path, output, samples_per_class=10):
    """Plot one row of generated images for each requested MNIST digit.

    Args:
        sample_path: ``.pt`` file with ``samples: (N, 28, 28)``, ``labels: (N,)``,
            and the quantization level ``K``.
        output: PNG path for the rendered conditional grid.
        samples_per_class: Number of samples to show in each of the ten rows.
    """
    payload = torch.load(sample_path, map_location="cpu", weights_only=True)
    samples, labels, K = payload["samples"], payload["labels"], payload["K"]
    fig, axes = plt.subplots(10, samples_per_class, figsize=(samples_per_class, 10))
    for label in range(10):
        selected = samples[labels == label][:samples_per_class]
        if len(selected) < samples_per_class:
            raise ValueError(f"Condition {label} has only {len(selected)} samples.")
        for column, image in enumerate(selected):
            axis = axes[label, column]
            axis.imshow(image.numpy() / (K - 1), cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
            if column == 0:
                axis.set_ylabel(f"target {label}", rotation=0, labelpad=30, va="center")
    title = (
        f"Conditional MNIST samples — {payload.get('transition', 'unknown')} "
        f"seed {payload.get('seed', 'unknown')}"
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples-per-class", type=int, default=10)
    args = parser.parse_args()
    output = args.output or str(Path(args.samples).with_name(
        f"{Path(args.samples).stem}_conditional_grid.png"
    ))
    plot_conditional_grid(args.samples, output, args.samples_per_class)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
