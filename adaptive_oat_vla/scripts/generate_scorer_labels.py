"""Generate offline scorer targets from a frozen OAT tokenizer on LIBERO zarr data."""

from __future__ import annotations

import argparse
import dill
from pathlib import Path
from typing import Dict, List

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from adaptive_oat_vla.common import ensure_parent, load_checkpoint_cfg, parse_candidate_ks
from oat.common.hydra_util import register_new_resolvers
from oat.tokenizer.oat.tokenizer import OATTok


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI args for scorer-label generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--oat-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--candidate-ks", default="1,2,4,8")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def resolve_dataset(cfg: OmegaConf, split: str, repo_root: Path):
    """Instantiate the tokenizer-training dataset used by the OAT checkpoint."""
    register_new_resolvers()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    zarr_path = Path(cfg.task.tokenizer.dataset.zarr_path)
    if not zarr_path.is_absolute():
        cfg.task.tokenizer.dataset.zarr_path = str((repo_root / zarr_path).resolve())
    dataset = hydra.utils.instantiate(cfg.task.tokenizer.dataset)
    if split == "val":
        dataset = dataset.get_validation_dataset()
    return dataset


def load_oat_tokenizer_from_payload(checkpoint: str | Path, device: str):
    with open(checkpoint, "rb") as handle:
        payload = torch.load(handle, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    state_dicts = payload["state_dicts"]
    tokenizer.load_state_dict(state_dicts["model"])
    if getattr(cfg.training, "use_ema", False) and "ema_model" in state_dicts:
        tokenizer.load_state_dict(state_dicts["ema_model"])
    tokenizer.to(device)
    tokenizer.eval()
    return tokenizer, cfg


def main() -> None:
    """Run offline OAT prefix reconstruction sweeps and save scorer labels."""
    args = build_argparser().parse_args()
    candidate_ks = parse_candidate_ks(args.candidate_ks)
    tokenizer, cfg = load_oat_tokenizer_from_payload(args.oat_checkpoint, device=args.device)
    repo_root = Path(args.oat_checkpoint).resolve().parents[3]
    dataset = resolve_dataset(cfg, args.split, repo_root=repo_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    latents_out: List[torch.Tensor] = []
    token_ids_out: List[torch.Tensor] = []
    scores_out: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            actions = batch["action"].to(args.device, non_blocking=True)
            latents, token_ids = tokenizer.encode(actions)
            batch_scores = []
            for keep_k in candidate_ks:
                recons = tokenizer.decode(latents, eval_keep_k=[keep_k] * actions.shape[0])
                mse = F.mse_loss(recons, actions, reduction="none").mean(dim=(1, 2))
                batch_scores.append(mse)
            scores = torch.stack(batch_scores, dim=1)
            latents_out.append(latents.cpu())
            token_ids_out.append(token_ids.cpu())
            scores_out.append(scores.cpu())

    payload: Dict[str, object] = {
        "latents": torch.cat(latents_out, dim=0),
        "token_ids": torch.cat(token_ids_out, dim=0),
        "scores": torch.cat(scores_out, dim=0),
        "candidate_ks": candidate_ks,
        "candidate_indices": [k - 1 for k in candidate_ks],
        "latent_horizon": int(tokenizer.latent_horizon),
        "split": args.split,
        "source_checkpoint": str(args.oat_checkpoint),
    }
    output_path = ensure_parent(args.output)
    torch.save(payload, output_path)
    print(f"saved_labels={output_path}")
    print(f"num_samples={payload['latents'].shape[0]}")
    print(f"score_shape={tuple(payload['scores'].shape)}")


if __name__ == "__main__":
    main()
