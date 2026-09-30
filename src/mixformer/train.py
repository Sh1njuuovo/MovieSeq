"""Train the standalone MixFormer model on JSONL implicit-feedback examples."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MixFormer


def load_examples(path, max_history=200):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                history = [int(item) for item in row["history"]][-max_history:]
                candidate = int(row["candidate"])
                label = int(row["label"])
                if not history or min(history) <= 0 or candidate <= 0 or label not in (0, 1):
                    raise ValueError("history/candidate must be positive IDs and label must be 0 or 1")
                rows.append((history, candidate, label))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid example on line {number}: {exc}") from exc
    if not rows:
        raise ValueError("input contains no examples")
    return rows


def train_file(path, epochs=5, dim=64, heads=4, batch_size=128, lr=1e-3, seed=42, output=None):
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    rows = load_examples(path)
    width = max(len(row[0]) for row in rows)
    histories = torch.tensor([[0] * (width - len(h)) + h for h, _, _ in rows], dtype=torch.long)
    candidates = torch.tensor([c for _, c, _ in rows], dtype=torch.long)
    labels = torch.tensor([y for _, _, y in rows], dtype=torch.float32)
    num_items = int(max(histories.max(), candidates.max()))
    model = MixFormer(num_items, dim=dim, heads=heads)
    loader = DataLoader(TensorDataset(histories, candidates, labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    report = {"examples": len(rows), "num_items": num_items, "epochs": []}
    for epoch in range(epochs):
        model.train()
        weighted_loss = 0.0
        for history, candidate, label in loader:
            optimizer.zero_grad()
            loss = criterion(model(history, candidate), label)
            loss.backward()
            optimizer.step()
            weighted_loss += loss.item() * len(label)
        report["epochs"].append({"epoch": epoch + 1, "training_loss": weighted_loss / len(rows)})
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "num_items": num_items, "dim": dim, "heads": heads}, destination / "model.pt")
        (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(train_file(args.examples, args.epochs, args.dim, args.heads, args.batch_size, output=args.output), indent=2))


if __name__ == "__main__":
    main()
