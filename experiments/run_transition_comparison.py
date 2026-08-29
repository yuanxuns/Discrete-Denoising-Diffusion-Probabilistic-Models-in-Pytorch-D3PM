"""Train matched D3PM transition-kernel baselines on MNIST or a 2-D Swiss roll.

Examples:
    python experiments/run_transition_comparison.py --dataset swiss --steps 2000
    python experiments/run_transition_comparison.py --dataset mnist --steps 50000 --device cuda
"""

import argparse
import csv
from collections import defaultdict
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
from models.transition_matrices import compute_transition_matrices


TRANSITIONS = ("uniform", "gaussian", "absorbing")


def terminal_total_variation(betas, transition, K, transition_bands):
    """Return the worst-row TV distance between q(x_T | x_0) and its prior."""
    q_mats = compute_transition_matrices(
        betas, transition, K, transition_bands, mask_id=K // 2
    ).float()
    q_bar = q_mats[0]
    for matrix in q_mats[1:]:
        q_bar = q_bar @ matrix
    if transition == "absorbing":
        prior = torch.zeros(K)
        prior[K // 2] = 1.0
    else:
        prior = torch.full((K,), 1.0 / K)
    return float((0.5 * (q_bar - prior).abs().sum(dim=-1)).max())


def make_betas(num_timesteps, transition, K, transition_bands, terminal_tv):
    """Calibrate a linear ``(T,)`` beta schedule to a common terminal TV target.

    The transition families have different mixing rates. A separate binary search
    over the schedule endpoint ensures the sampler's prior is equally accurate for
    every family before comparing learned reverse models.
    """
    low, high = 1e-4, 0.95
    max_betas = torch.linspace(1e-4, high, num_timesteps)
    max_tv = terminal_total_variation(max_betas, transition, K, transition_bands)
    if max_tv > terminal_tv:
        raise ValueError(
            f"terminal_tv={terminal_tv} is unreachable for {transition} with "
            f"T={num_timesteps}; best achievable TV is {max_tv:.6f}."
        )
    for _ in range(32):
        end = (low + high) / 2
        betas = torch.linspace(1e-4, end, num_timesteps)
        if terminal_total_variation(betas, transition, K, transition_bands) > terminal_tv:
            low = end
        else:
            high = end
    betas = torch.linspace(1e-4, high, num_timesteps)
    return betas, terminal_total_variation(betas, transition, K, transition_bands)


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
    dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(labels))
    return dataset, (lo, hi), (points, labels)


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


def save_swiss_reference(reference, output):
    """Save the continuous Swiss-roll target as a reference scatter plot.

    Args:
        reference: Float array of shape ``(N, 2)`` in original data coordinates.
        output: Destination path for the PNG figure.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(reference[:, 0], reference[:, 1], s=3, alpha=0.45)
    ax.set_title("Swiss roll — training distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def sliced_wasserstein_distance(reference, generated, num_projections, seed):
    """Estimate 2-D sliced Wasserstein-1 with equal-sized empirical samples."""
    rng = np.random.default_rng(seed)
    reference = reference[rng.choice(len(reference), len(generated), replace=len(reference) < len(generated))]
    directions = rng.normal(size=(num_projections, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ref_proj = np.sort(reference @ directions.T, axis=0)
    gen_proj = np.sort(generated @ directions.T, axis=0)
    return float(np.abs(ref_proj - gen_proj).mean())


def evaluate_swiss(samples, sample_labels, reference, reference_labels, bounds, K, seed):
    """Compute mean class-conditional SWD in original Swiss-roll coordinates."""
    lo, hi = bounds
    generated = samples.cpu().numpy() / (K - 1) * (hi - lo) + lo
    scores = []
    for label in np.unique(reference_labels):
        scores.append(sliced_wasserstein_distance(
            reference[reference_labels == label], generated[sample_labels == label],
            num_projections=128, seed=seed + int(label),
        ))
    return float(np.mean(scores))


def run_one(args, transition, train_data, swiss_bounds, swiss_reference, seed):
    """Train one baseline, logging loss curves, and return its final mean loss.

    TensorBoard events are written to ``<tensorboard_dir>/<dataset>/<transition>``.
    """
    # Reset all stochastic sources so every kernel gets the same initialization,
    # minibatch order, timesteps, and Gumbel draws as far as its computation permits.
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    data_shape = (2,) if args.dataset == "swiss" else (28, 28)
    labels = 3 if args.dataset == "swiss" else 10
    betas, terminal_tv = make_betas(
        args.timesteps, transition, args.K, args.transition_bands,
        args.terminal_tv,
    )
    d3pm = D3PM(
        betas=betas, model_prediction_type="x_start",
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
    log_dir = Path(args.tensorboard_dir) / args.dataset / transition / f"seed_{seed}"
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
    sample_labels = torch.arange(labels, device=device).repeat_interleave(args.samples_per_class)
    samples = d3pm.p_sample_loop(model, (len(sample_labels), *data_shape), model_kwargs={"y": sample_labels})
    output = Path(args.output_dir) / args.dataset / f"{transition}_seed{seed}.png"
    save_samples(samples, args.dataset, output, transition, args.K, swiss_bounds)
    final_loss = float(np.mean(losses[-min(len(losses), args.log_every):]))
    writer.add_scalar("train/final_loss_bits", final_loss, args.steps)
    writer.add_scalar("schedule/terminal_total_variation", terminal_tv, 0)
    writer.close()
    result = {
        "seed": seed,
        "transition": transition,
        "final_train_loss_bits": final_loss,
        "terminal_tv": terminal_tv,
        "beta_end": float(betas[-1]),
        "samples": str(output),
    }
    if args.dataset == "swiss":
        result["conditional_swd"] = evaluate_swiss(
            samples, sample_labels.cpu().numpy(), *swiss_reference, swiss_bounds,
            args.K, seed,
        )
    return result


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
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--swiss-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--terminal-tv", type=float, default=0.005)
    parser.add_argument(
        "--save-reference-only", action="store_true",
        help="Save the continuous Swiss-roll target figure and exit.",
    )
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
        train_data, swiss_bounds, swiss_reference = make_swiss_dataset(args.swiss_samples, args.K, args.seed)
        if args.save_reference_only:
            output = Path(args.output_dir) / "swiss" / "reference.png"
            save_swiss_reference(swiss_reference[0], output)
            print(f"Wrote {output}")
            return
    else:
        if args.save_reference_only:
            raise ValueError("--save-reference-only is only available for --dataset swiss.")
        train_data, swiss_bounds, swiss_reference = make_mnist_dataset(args.data_dir, args.K, train=True), None, None
    results = []
    for seed in (args.seeds or [args.seed]):
        for transition in args.transitions:
            results.append(run_one(
                args, transition, train_data, swiss_bounds, swiss_reference, seed
            ))
    summary = Path(args.output_dir) / args.dataset / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    grouped = defaultdict(list)
    for result in results:
        grouped[result["transition"]].append(result)
    aggregate = Path(args.output_dir) / args.dataset / "aggregate_summary.csv"
    aggregate_rows = []
    for transition, rows in grouped.items():
        aggregate_rows.append({
            "transition": transition,
            "num_seeds": len(rows),
            "loss_mean": np.mean([row["final_train_loss_bits"] for row in rows]),
            "loss_std": np.std([row["final_train_loss_bits"] for row in rows]),
            "terminal_tv_mean": np.mean([row["terminal_tv"] for row in rows]),
            "conditional_swd_mean": np.mean([row.get("conditional_swd", np.nan) for row in rows]),
            "conditional_swd_std": np.std([row.get("conditional_swd", np.nan) for row in rows]),
        })
    with aggregate.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"Wrote {summary}")
    print(f"Wrote {aggregate}")


if __name__ == "__main__":
    main()
