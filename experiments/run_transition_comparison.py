"""Train matched D3PM transition-kernel baselines on MNIST or a 2-D Swiss roll.

Examples:
    python experiments/run_transition_comparison.py --dataset swiss --steps 2000
    python experiments/run_transition_comparison.py --dataset mnist --steps 50000 --device cuda
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models.d3pm import D3PM
from models.dit import DiT


TRANSITIONS = ("uniform", "gaussian", "absorbing")


def make_betas(num_timesteps, transition):
    """Return a matched ``(T,)`` beta schedule for a transition family.

    Gaussian kernels require substantially larger per-step variances than uniform
    kernels to approach their uniform stationary prior over the same horizon.
    """
    end = 0.20 if transition == "gaussian" else 0.02
    return torch.linspace(1e-4, end, num_timesteps)


def make_swiss_dataset(samples, K, seed):
    """Create quantized Swiss-roll data.

    Returns:
        Dataset of ``x: (N, 2)`` integer D3PM states and ``y: (N,)`` section
        labels in ``{0, 1, 2}``.
    """
    points, roll_t = make_swiss_roll(n_samples=samples, noise=0.25, random_state=seed)
    points = points[:, [0, 2]]
    lo, hi = points.min(0), points.max(0)
    states = np.rint((points - lo) / (hi - lo) * (K - 1)).clip(0, K - 1).astype(np.int64)
    labels = np.clip(np.floor((roll_t - roll_t.min()) / (roll_t.max() - roll_t.min()) * 3), 0, 2).astype(np.int64)
    return TensorDataset(torch.from_numpy(states), torch.from_numpy(labels)), (lo, hi)


def make_mnist_dataset(data_dir, K, train):
    """Load MNIST and quantize pixels to an ``(N, 28, 28)`` integer state tensor."""
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transforms.PILToTensor())
    x = torch.stack([image.squeeze(0) for image, _ in dataset])
    x = torch.round(x.float() / 255 * (K - 1)).long()
    y = dataset.targets.to(torch.long)
    return TensorDataset(x, y)


def save_samples(samples, dataset, output, transition, K, swiss_bounds=None):
    """Save samples. ``samples`` is ``(B, *data_shape)`` with values in ``[0, K-1]``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "swiss":
        lo, hi = swiss_bounds
        points = samples.cpu().numpy() / (K - 1) * (hi - lo) + lo
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.7)
        ax.set_title(f"Swiss roll — {transition}")
    else:
        images = samples[:64].cpu().numpy()
        side = 8
        canvas = np.zeros((side * 28, side * 28), dtype=images.dtype)
        for idx, image in enumerate(images):
            row, col = divmod(idx, side)
            canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = image
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(canvas, cmap="gray")
        ax.set_title(f"MNIST — {transition}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_one(args, transition, train_data, swiss_bounds):
    """Train one baseline, logging loss curves, and return its final mean loss.

    TensorBoard events are written to ``<tensorboard_dir>/<dataset>/<transition>``.
    """
    # Reset all stochastic sources so every kernel gets the same initialization,
    # minibatch order, timesteps, and Gumbel draws as far as its computation permits.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data_shape = (2,) if args.dataset == "swiss" else (28, 28)
    labels = 3 if args.dataset == "swiss" else 10
    d3pm = D3PM(
        betas=make_betas(args.timesteps, transition), model_prediction_type="x_start",
        logits_type="logits", transition_matrix_type=transition,
        transition_bands=args.transition_bands, loss_type="hybrid",
        hybrid_coeff=args.hybrid_coeff, K=args.K,
    ).to(device)
    model = DiT(
        input_shape=data_shape, num_classes=args.K, num_timesteps=args.timesteps,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.heads,
        condition_classes=labels,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    iterator = iter(loader)
    losses = []
    log_dir = Path(args.tensorboard_dir) / args.dataset / transition
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config/transition", transition)
    writer.add_text("config/device", str(device))
    writer.add_hparams(
        {
            "K": args.K,
            "timesteps": args.timesteps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        {},
    )
    print(f"TensorBoard logs: {log_dir}", flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        t = d3pm.sample_timesteps(x.shape[0], device)
        loss = d3pm.training_losses(model, x, t, model_kwargs={"y": y}).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        writer.add_scalar("train/loss_bits", loss.item(), step)
        writer.add_scalar("train/loss_bits_window", np.mean(losses[-args.log_every:]), step)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
        if step % args.log_every == 0 or step == args.steps:
            print(f"{transition} step={step:6d} loss={np.mean(losses[-args.log_every:]):.4f}", flush=True)
    model.eval()
    sample_labels = torch.arange(labels, device=device).repeat_interleave(args.samples // labels + 1)[:args.samples]
    samples = d3pm.p_sample_loop(model, (args.samples, *data_shape), model_kwargs={"y": sample_labels})
    output = Path(args.output_dir) / args.dataset / f"{transition}.png"
    save_samples(samples, args.dataset, output, transition, args.K, swiss_bounds)
    final_loss = float(np.mean(losses[-min(len(losses), args.log_every):]))
    writer.add_scalar("train/final_loss_bits", final_loss, args.steps)
    writer.close()
    return final_loss, output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("swiss", "mnist"), required=True)
    parser.add_argument("--output-dir", default="artifacts/transition_comparison")
    parser.add_argument(
        "--tensorboard-dir",
        default="artifacts/tensorboard",
        help="Base directory for TensorBoard event files.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--transition-bands", type=int, default=None)
    parser.add_argument("--hybrid-coeff", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--swiss-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--transitions", nargs="+", choices=TRANSITIONS, default=TRANSITIONS,
        help="Subset of transition kernels to run (default: all three).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == "swiss":
        train_data, swiss_bounds = make_swiss_dataset(args.swiss_samples, args.K, args.seed)
    else:
        train_data, swiss_bounds = make_mnist_dataset(args.data_dir, args.K, train=True), None
    results = []
    for transition in args.transitions:
        loss, output = run_one(args, transition, train_data, swiss_bounds)
        results.append({"transition": transition, "final_train_loss_bits": loss, "samples": str(output)})
    summary = Path(args.output_dir) / args.dataset / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
