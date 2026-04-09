"""Label the minimum acceptable OAT prefix length K* from scorer outputs or offline scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from adaptive_oat_vla.common import ensure_parent
from adaptive_oat_vla.models import ActionQualityScorer


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI args for K* / EOS-label generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to scorer label .pt file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.2,
        help="Used when --threshold is omitted: threshold = mean(full_score) * multiplier",
    )
    parser.add_argument("--scorer-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    return parser


def load_scores(args) -> tuple[torch.Tensor, List[int], float]:
    """Load either ground-truth scores or scorer-predicted scores and derive the threshold."""
    label_payload = torch.load(args.labels, map_location="cpu", weights_only=False)
    candidate_ks = [int(k) for k in label_payload["candidate_ks"]]
    if args.scorer_checkpoint:
        scorer_payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
        scorer = ActionQualityScorer(
            latent_dim=int(scorer_payload["latent_dim"]),
            latent_horizon=int(scorer_payload["latent_horizon"]),
            hidden_dim=int(scorer_payload["hidden_dim"]),
            num_layers=int(scorer_payload["num_layers"]),
            num_heads=int(scorer_payload["num_heads"]),
            dropout=float(scorer_payload.get("dropout", 0.1)),
            candidate_ks=candidate_ks,
            causal=bool(scorer_payload.get("causal", True)),
        ).to(args.device)
        scorer.load_state_dict(scorer_payload["state_dict"])
        scorer.eval()
        with torch.no_grad():
            scores = scorer(label_payload["latents"].to(args.device)).cpu()
    else:
        scores = label_payload["scores"].float()

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(scores[:, -1].mean().item() * args.threshold_multiplier)
    return scores, candidate_ks, threshold


def main() -> None:
    """Assign one minimum acceptable prefix length K* per sample and save JSON/pt output."""
    args = build_argparser().parse_args()
    scores, candidate_ks, threshold = load_scores(args)

    optimal_ks: List[int] = []
    eos_positions: List[int] = []
    for row in scores:
        k_star = candidate_ks[-1]
        for idx, keep_k in enumerate(candidate_ks):
            if row[idx].item() < threshold:
                k_star = keep_k
                break
        optimal_ks.append(k_star)
        eos_positions.append(k_star)

    payload = {
        "optimal_ks": optimal_ks,
        "eos_positions": eos_positions,
        "candidate_ks": candidate_ks,
        "candidate_indices": [k - 1 for k in candidate_ks],
        "threshold": threshold,
        "num_samples": len(optimal_ks),
    }

    output_path = ensure_parent(args.output)
    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(payload, indent=2))
    else:
        torch.save(payload, output_path)
    print(f"saved_eos_labels={output_path}")
    print(f"threshold={threshold:.6f}")
    print(f"avg_k={sum(optimal_ks) / max(len(optimal_ks), 1):.4f}")


if __name__ == "__main__":
    main()
