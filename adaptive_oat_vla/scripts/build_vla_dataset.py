"""Build lightweight VLA training annotations from OAT token labels and K*/EOS labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import dill
import numpy as np
import torch
import zarr

from adaptive_oat_vla.common import ensure_parent
from adaptive_oat_vla.scripts.generate_scorer_labels import resolve_dataset
from oat.common.seq_sampler import downsample_mask, get_val_mask
from oat.common.hydra_util import register_new_resolvers


DEFAULT_OBS_KEYS = [
    "agentview_rgb",
    "robot0_eye_in_hand_rgb",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "task_uid",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oat-checkpoint", required=True)
    parser.add_argument("--labels", required=True, help="Output of generate_scorer_labels.py")
    parser.add_argument("--eos-labels", required=True, help="Output of label_eos_positions.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--obs-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--random-larger-k-prob",
        type=float,
        default=0.0,
        help="Optional offline augmentation: sample a larger K from {K*, ..., Kmax}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def padded_prefix(full_tokens: torch.Tensor, target_k: int, max_k: int) -> torch.Tensor:
    prefix = torch.full((max_k,), fill_value=-1, dtype=torch.long)
    prefix[:target_k] = full_tokens[:target_k]
    return prefix


def build_obs_indices(
    current_idx: int,
    obs_steps: int,
    episode_start: int,
) -> list[int]:
    start = current_idx - obs_steps + 1
    return [max(episode_start, start + offset) for offset in range(obs_steps)]


def load_eos_labels(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint_cfg(path: str | Path):
    with open(path, "rb") as handle:
        payload = torch.load(handle, pickle_module=dill, map_location="cpu")
    return payload["cfg"]


def resolve_aligned_indices(cfg, split: str, repo_root: Path, episode_ends: np.ndarray) -> np.ndarray:
    """Compute tokenizer dataset sequence indices without materializing ReplayBuffer."""
    dataset_cfg = cfg.task.tokenizer.dataset
    try:
        n_obs_steps = int(dataset_cfg.n_obs_steps)
        n_action_steps = int(dataset_cfg.n_action_steps)
        seed = int(dataset_cfg.seed)
        val_ratio = float(dataset_cfg.val_ratio)
        max_train_episodes = dataset_cfg.get("max_train_episodes", None)
        if max_train_episodes is not None:
            max_train_episodes = int(max_train_episodes)
    except Exception:
        action_dataset = resolve_dataset(cfg, split, repo_root=repo_root)
        return action_dataset.seq_sampler.indices

    val_mask = get_val_mask(
        n_episodes=len(episode_ends),
        val_ratio=val_ratio,
        seed=seed,
    )
    train_mask = downsample_mask(
        mask=~val_mask,
        max_n=max_train_episodes,
        seed=seed,
    )
    episode_mask = train_mask if split == "train" else ~train_mask

    pad_before = max(n_obs_steps - 1, 0)
    pad_after = max(n_action_steps - 1, 0)
    seq_len = pad_before + 1 + pad_after
    indices = []
    for episode_idx, use_episode in enumerate(episode_mask):
        if not use_episode:
            continue
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        episode_end = int(episode_ends[episode_idx])
        episode_length = episode_end - episode_start
        min_start = -pad_before
        max_start = episode_length - seq_len + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + episode_start
            buffer_end_idx = min(idx + seq_len, episode_length) + episode_start
            start_offset = buffer_start_idx - (idx + episode_start)
            end_offset = (idx + seq_len + episode_start) - buffer_end_idx
            sample_start_idx = start_offset
            sample_end_idx = seq_len - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    return np.asarray(indices, dtype=np.int64)


def main() -> None:
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)

    print(f"loading_labels={args.labels}", flush=True)
    labels = torch.load(args.labels, map_location="cpu", weights_only=False)
    print(f"loading_eos_labels={args.eos_labels}", flush=True)
    eos_labels = load_eos_labels(args.eos_labels)
    candidate_ks = [int(k) for k in labels["candidate_ks"]]
    optimal_ks = [int(k) for k in eos_labels["optimal_ks"]]
    token_ids = labels["token_ids"].long()
    max_k = int(token_ids.shape[1])

    if token_ids.shape[0] != len(optimal_ks):
        raise ValueError(
            f"labels/eos count mismatch: {token_ids.shape[0]} vs {len(optimal_ks)}"
        )

    register_new_resolvers()
    print(f"loading_checkpoint_cfg={args.oat_checkpoint}", flush=True)
    cfg = load_checkpoint_cfg(args.oat_checkpoint)
    repo_root = Path(args.oat_checkpoint).resolve().parents[3]
    zarr_path = Path(cfg.task.tokenizer.dataset.zarr_path)
    if not zarr_path.is_absolute():
        zarr_path = (repo_root / zarr_path).resolve()
    root = zarr.open(str(zarr_path), mode="r")
    data_group = root["data"]
    episode_ends = root["meta"]["episode_ends"][:]
    print(f"loading_prompts={zarr_path}", flush=True)
    prompts_array = data_group["prompt"][:]
    print("computing_aligned_indices", flush=True)
    aligned_indices = resolve_aligned_indices(
        cfg=cfg,
        split=args.split,
        repo_root=repo_root,
        episode_ends=episode_ends,
    )
    if token_ids.shape[0] > len(aligned_indices):
        raise ValueError(
            f"dataset/label count mismatch: dataset={len(aligned_indices)} labels={token_ids.shape[0]}"
        )
    print(
        f"building_annotations split={args.split} labels={token_ids.shape[0]} indices={len(aligned_indices)}",
        flush=True,
    )

    num_samples = token_ids.shape[0] if args.max_samples is None else min(
        token_ids.shape[0], args.max_samples
    )

    buffer_start_indices = []
    obs_indices = []
    prompts = []
    target_ks = []
    optimal_ks_out = []
    full_token_ids = []
    target_token_ids = []
    eos_positions = []

    for sample_idx in range(num_samples):
        if sample_idx % 50000 == 0 and sample_idx > 0:
            print(f"processed={sample_idx}", flush=True)
        buffer_start_idx = int(aligned_indices[sample_idx][0])
        episode_idx = int(np.searchsorted(episode_ends, buffer_start_idx, side="right"))
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])

        optimal_k = optimal_ks[sample_idx]
        if args.random_larger_k_prob > 0.0 and rng.random() < args.random_larger_k_prob:
            valid_ks = [k for k in candidate_ks if k >= optimal_k]
            target_k = rng.choice(valid_ks)
        else:
            target_k = optimal_k

        full_ids = token_ids[sample_idx]
        buffer_start_indices.append(buffer_start_idx)
        obs_indices.append(build_obs_indices(buffer_start_idx, args.obs_steps, episode_start))
        prompt = prompts_array[buffer_start_idx]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        prompts.append(str(prompt))
        target_ks.append(target_k)
        optimal_ks_out.append(optimal_k)
        full_token_ids.append(full_ids)
        target_token_ids.append(padded_prefix(full_ids, target_k=target_k, max_k=max_k))
        eos_positions.append(target_k)

    payload = {
        "split": args.split,
        "zarr_path": str(zarr_path),
        "obs_keys": DEFAULT_OBS_KEYS,
        "obs_steps": args.obs_steps,
        "candidate_ks": candidate_ks,
        "source_checkpoint": str(args.oat_checkpoint),
        "source_labels": str(args.labels),
        "source_eos_labels": str(args.eos_labels),
        "buffer_start_indices": torch.tensor(buffer_start_indices, dtype=torch.int64),
        "obs_indices": torch.tensor(obs_indices, dtype=torch.int64),
        "prompts": prompts,
        "optimal_ks": torch.tensor(optimal_ks_out, dtype=torch.int64),
        "target_ks": torch.tensor(target_ks, dtype=torch.int64),
        "eos_positions": torch.tensor(eos_positions, dtype=torch.int64),
        "full_oat_token_ids": torch.stack(full_token_ids, dim=0),
        "target_oat_token_ids": torch.stack(target_token_ids, dim=0),
    }

    output_path = ensure_parent(args.output)
    torch.save(payload, output_path)
    print(f"saved_vla_annotations={output_path}")
    print(f"num_samples={num_samples}")
    print(f"avg_target_k={payload['target_ks'].float().mean().item():.4f}")


if __name__ == "__main__":
    main()
