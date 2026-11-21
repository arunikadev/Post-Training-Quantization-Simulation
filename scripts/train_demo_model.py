"""Quick training script to generate realistic demo checkpoints in checkpoints/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ptq_sim.config.schema import ModelConfig
from ptq_sim.data.synthetic import build_synthetic_train_loader
from ptq_sim.models.registry import build_model
from ptq_sim.utils.device import resolve_device
from ptq_sim.utils.seed import set_seed

MODEL_TO_DEFAULT_DATASET = {
    "simple_cnn": "synthetic_image",
    "mini_resnet": "synthetic_image",
    "simple_mlp": "synthetic_tabular",
    "mini_transformer": "synthetic_sequence",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a built-in demo model and save its state_dict.")
    parser.add_argument(
        "--model",
        type=str,
        default="mini_resnet",
        choices=sorted(MODEL_TO_DEFAULT_DATASET.keys()),
        help="Demo model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--num-samples", type=int, default=1536, help="Number of synthetic training samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu').")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Destination path for saved state_dict (.pth). Defaults to checkpoints/demo_<model>.pth",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_name = MODEL_TO_DEFAULT_DATASET[args.model]

    model, _ = build_model(ModelConfig(name=args.model))
    model.to(device)
    train_loader = build_synthetic_train_loader(
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += int(x.shape[0])
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | loss={total_loss / total:.4f} | "
            f"train_acc={correct / total:.4f}"
        )

    save_path = Path(args.save_path or f"checkpoints/demo_{args.model}.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state_dict, save_path)
    print(f"Saved state_dict to '{save_path}'.")


if __name__ == "__main__":
    main()
