"""Train a quantized-MNIST classifier and score conditional D3PM samples.

Example:
    python experiments/evaluate_mnist_classifier.py \
      --sample-files artifacts/mnist/uniform_seed0.pt \
                     artifacts/mnist/gaussian_seed0.pt \
                     artifacts/mnist/absorbing_seed0.pt \
      --K 32 --device cuda
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from models.mnist_classifier import MNISTClassifier


class QuantizedMNIST(Dataset):
    """MNIST dataset quantized to the same ``K`` states as the D3PM samples."""

    def __init__(self, root, train, K):
        self.dataset = datasets.MNIST(
            root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.K = K

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = torch.round(image * (self.K - 1)) / (self.K - 1)
        return image, label


def accuracy(model, loader, device):
    """Return classifier accuracy on a loader of ``(B, 1, 28, 28)`` images."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total


def train_classifier(args, device):
    """Load or train a classifier and return it with quantized-test accuracy."""
    checkpoint = Path(args.classifier_checkpoint)
    model = MNISTClassifier().to(device)
    test_loader = DataLoader(
        QuantizedMNIST(args.data_dir, train=False, K=args.K),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    if checkpoint.exists() and not args.retrain:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["K"] != args.K:
            raise ValueError(f"Checkpoint K={state['K']} does not match --K={args.K}.")
        model.load_state_dict(state["model"])
        return model, accuracy(model, test_loader, device)

    train_loader = DataLoader(
        QuantizedMNIST(args.data_dir, train=True, K=args.K),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"classifier epoch={epoch} test_accuracy={accuracy(model, test_loader, device):.4f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "K": args.K}, checkpoint)
    return model, accuracy(model, test_loader, device)


def evaluate_samples(model, path, device, batch_size, expected_K):
    """Score one saved D3PM sample file against its requested conditional labels."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    samples, labels, K = payload["samples"], payload["labels"], payload["K"]
    if K != expected_K:
        raise ValueError(f"Sample file K={K} does not match classifier K={expected_K}.")
    if samples.ndim != 3 or samples.shape[1:] != (28, 28):
        raise ValueError(f"Expected samples with shape (N, 28, 28), got {tuple(samples.shape)}.")
    images = samples.float().unsqueeze(1) / (K - 1)
    predicted, target_probs = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            probs = model(images[start:start + batch_size].to(device)).softmax(dim=1).cpu()
            predicted.append(probs.argmax(dim=1))
            target_probs.append(probs.gather(1, labels[start:start + batch_size, None]).squeeze(1))
    predicted, target_probs = torch.cat(predicted), torch.cat(target_probs)
    rows = []
    for label in range(10):
        mask = labels == label
        if mask.any():
            rows.append({
                "target_class": label,
                "count": int(mask.sum()),
                "target_accuracy": float((predicted[mask] == label).float().mean()),
                "mean_target_probability": float(target_probs[mask].mean()),
            })
    return {
        "sample_file": str(path),
        "transition": payload.get("transition", "unknown"),
        "seed": payload.get("seed", "unknown"),
        "target_accuracy": float((predicted == labels).float().mean()),
        "mean_target_probability": float(target_probs.mean()),
        "per_class": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-files", nargs="+", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classifier-checkpoint", default="artifacts/mnist_classifier_k32.pt")
    parser.add_argument("--output", default="artifacts/mnist_classifier_eval.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, test_accuracy = train_classifier(args, device)
    results = [
        evaluate_samples(model, Path(path), device, args.batch_size, args.K)
        for path in args.sample_files
    ]
    for result in results:
        result["classifier_test_accuracy"] = test_accuracy
        print(
            f"{result['transition']} seed={result['seed']} "
            f"target_accuracy={result['target_accuracy']:.4f} "
            f"target_probability={result['mean_target_probability']:.4f}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="") as f:
        fieldnames = [key for key in results[0] if key != "per_class"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k != "per_class"} for row in results])
    print(f"Wrote {output} and {csv_output}")


if __name__ == "__main__":
    main()
