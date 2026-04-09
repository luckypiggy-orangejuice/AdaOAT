"""Train the adaptive OAT prefix-quality scorer from precomputed offline labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adaptive_oat_vla.common import ensure_parent
from adaptive_oat_vla.models import ActionQualityScorer


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI args for scorer training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Disable causal attention. Default keeps prefix-causal scoring.",
    )
    return parser


def main() -> None:
    """Fit the scorer and save weights plus training metadata."""
    args = build_argparser().parse_args()
    payload = torch.load(args.labels, map_location="cpu", weights_only=False)
    latents = payload["latents"].float()
    scores = payload["scores"].float()
    candidate_ks = [int(k) for k in payload["candidate_ks"]]

    dataset = TensorDataset(latents, scores)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ActionQualityScorer(
        latent_dim=latents.shape[-1],
        latent_horizon=latents.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        candidate_ks=candidate_ks,
        causal=not args.no_causal,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for z_q, score in loader:
            z_q = z_q.to(args.device, non_blocking=True)
            score = score.to(args.device, non_blocking=True)
            pred = model(z_q)
            loss = F.mse_loss(pred, score)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch} loss={total / max(count, 1):.6f}")

    save_payload = {
        "state_dict": model.state_dict(),
        "candidate_ks": candidate_ks,
        "latent_dim": latents.shape[-1],
        "latent_horizon": latents.shape[1],
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "causal": not args.no_causal,
        "candidate_indices": model.candidate_indices,
        "source_labels": str(args.labels),
    }
    output_path = ensure_parent(args.output)
    torch.save(save_payload, output_path)
    print(f"saved_scorer={output_path}")


if __name__ == "__main__":
    main()
