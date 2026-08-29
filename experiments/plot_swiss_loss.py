"""Render aggregate Swiss-roll training-loss curves from TensorBoard events.

Example:
    python experiments/plot_swiss_loss.py \
      --logdir artifacts/tensorboard/swiss_fair \
      --output docs/images/swiss-roll-training-loss.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TRANSITIONS = ("uniform", "gaussian", "absorbing")


def load_curve(path, tag):
    """Load a scalar TensorBoard curve as ``(steps, values)`` arrays."""
    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    return (
        np.array([event.step for event in events]),
        np.array([event.value for event in events]),
    )


def plot_curves(logdir, output, tag="train/loss_bits_window"):
    """Plot seed mean and standard deviation for every transition.

    Args:
        logdir: Root TensorBoard directory containing ``seed*/swiss`` runs.
        output: PNG destination path.
        tag: Scalar tag to plot; default is the rolling training-loss mean.
    """
    logdir, output = Path(logdir), Path(output)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for transition in TRANSITIONS:
        event_paths = sorted(logdir.glob(
            f"seed*/swiss/{transition}/seed_*/events.out.tfevents.*"
        ))
        curves = [load_curve(path, tag) for path in event_paths]
        if not curves:
            raise FileNotFoundError(f"No {transition} TensorBoard curves found in {logdir}.")
        steps = curves[0][0]
        values = np.stack([curve[1] for curve in curves])
        mean, std = values.mean(axis=0), values.std(axis=0)
        line, = ax.plot(steps, mean, label=transition)
        ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.18)
    ax.set_title("Swiss-roll training convergence")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rolling training loss (bits)")
    ax.legend(title="Transition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="artifacts/tensorboard/swiss_fair")
    parser.add_argument("--output", default="docs/images/swiss-roll-training-loss.png")
    args = parser.parse_args()
    plot_curves(args.logdir, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
