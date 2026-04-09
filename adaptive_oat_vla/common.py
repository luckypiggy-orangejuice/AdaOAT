"""Shared helpers for loading OAT artifacts and parsing adaptive-token configs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import dill
import hydra
import torch
from omegaconf import OmegaConf

from oat.common.hydra_util import register_new_resolvers


def parse_candidate_ks(values: Sequence[int] | str) -> List[int]:
    """Normalize candidate K values from CLI/config input."""
    if isinstance(values, str):
        items = [v.strip() for v in values.split(",") if v.strip()]
        ks = [int(v) for v in items]
    else:
        ks = [int(v) for v in values]
    if not ks:
        raise ValueError("candidate_ks must be non-empty")
    if sorted(set(ks)) != ks:
        raise ValueError(f"candidate_ks must be unique and sorted, got {ks}")
    return ks


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a target file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_cfg(checkpoint: str | Path) -> OmegaConf:
    """Read only the serialized Hydra config from an OAT checkpoint."""
    register_new_resolvers()
    with open(checkpoint, "rb") as handle:
        payload = torch.load(handle, pickle_module=dill, map_location="cpu")
    return payload["cfg"]


def load_oat_tokenizer(checkpoint: str | Path, device: str = "cuda"):
    """Restore a frozen OAT tokenizer from checkpoint payload without workspace wrappers."""
    register_new_resolvers()
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
