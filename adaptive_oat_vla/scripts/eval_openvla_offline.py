"""Offline evaluation for adaptive OpenVLA checkpoints on val/train annotations."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn.functional as F
import zarr
from PIL import Image
from tqdm import tqdm

from adaptive_oat_vla.common import ensure_parent, load_oat_tokenizer
from adaptive_oat_vla.common_vla import (
    load_annotation_payload,
    load_openvla_model,
    load_processor_and_tokenizer,
)
from adaptive_oat_vla.vla_dataset import AdaptiveVLACollator, AdaptiveVLADataset


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Adapter dir or full model dir.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--action-vocab-map", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--image-key", default="agentview_rgb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=9)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oat-checkpoint", default=None)
    parser.add_argument("--force-k", type=int, default=None, help="Ignore EOS and use the first K valid ACT tokens.")
    parser.add_argument("--output", required=True)
    return parser


def _prepare_generation_inputs(
    processor,
    prompt: str,
    primary_image: torch.Tensor,
    device: str,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    image = Image.fromarray(primary_image.cpu().numpy()).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    out = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value) and dtype is not None:
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device)
    return out


def _split_generation(generated: torch.Tensor, input_len: int, eos_token_id: int, max_action_tokens: int) -> tuple[list[int], bool]:
    new_tokens = generated[input_len:].tolist()
    if eos_token_id in new_tokens:
        eos_pos = new_tokens.index(eos_token_id)
        return [int(t) for t in new_tokens[:eos_pos]], True
    return [int(t) for t in new_tokens[:max_action_tokens]], False


def _collect_action_ids(
    token_ids: list[int],
    *,
    eos_token_id: int,
    act_id_to_oat: dict[int, int],
    max_action_tokens: int,
    force_k: int | None,
) -> tuple[list[int], bool]:
    oat_indices: list[int] = []
    hit_eos = False
    for tok in token_ids:
        tok = int(tok)
        if force_k is None and tok == eos_token_id:
            hit_eos = True
            break
        if force_k is not None and tok == eos_token_id:
            continue
        if tok in act_id_to_oat:
            oat_indices.append(act_id_to_oat[tok])
            if force_k is not None and len(oat_indices) >= force_k:
                break
            if force_k is None and len(oat_indices) >= max_action_tokens:
                break
    if force_k is not None and len(oat_indices) < force_k:
        oat_indices.extend([0] * (force_k - len(oat_indices)))
    return oat_indices, hit_eos


def main() -> None:
    args = build_argparser().parse_args()
    model_path = Path(args.model)
    if (model_path / "adapter_config.json").exists() and not (model_path / "action_token_rows.pt").exists():
        raise FileNotFoundError(
            f"{model_path} is a PEFT adapter dir but action_token_rows.pt is missing. "
            "This adapter was saved before action-token embedding export was implemented."
        )

    processor, tokenizer, _ = load_processor_and_tokenizer(
        model_or_adapter=args.model,
        processor_name=args.processor,
        trust_remote_code=args.trust_remote_code,
    )
    torch_dtype = torch.bfloat16 if args.bf16 else None
    model = load_openvla_model(
        model_or_adapter=args.model,
        tokenizer=tokenizer,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device=args.device,
    )
    model.eval()

    dataset = AdaptiveVLADataset(
        annotation_path=args.annotations,
        action_vocab_map_path=args.action_vocab_map,
        random_larger_k_prob=0.0,
        seed=args.seed,
        image_key=args.image_key,
    )
    collator = AdaptiveVLACollator(processor=processor, tokenizer=tokenizer)
    vocab_map = load_annotation_payload(args.action_vocab_map)
    eos_token_id = int(vocab_map.get("action_eos_token_id", vocab_map["eos_token_id"]))
    max_action_tokens = int(len(dataset.candidate_ks) and max(dataset.candidate_ks))
    act_id_to_oat = {int(v): idx for idx, v in enumerate(vocab_map["action_token_ids"])}
    oat_tok = None
    action_root = None
    action_horizon = None
    if args.oat_checkpoint:
        oat_tok, oat_cfg = load_oat_tokenizer(args.oat_checkpoint, device=args.device)
        action_root = zarr.open(dataset.annotations["zarr_path"], mode="r")["data"]["action"]
        action_horizon = int(oat_cfg.horizon)

    num_samples = min(len(dataset), args.max_samples)
    generated_hist = Counter()
    target_hist = Counter()
    eos_hits = 0
    exact_matches = 0
    token_accuracy_sum = 0.0
    decoded_action_mse_sum = 0.0
    decoded_action_mse_count = 0
    records = []

    for idx in tqdm(range(num_samples), desc="offline-eval", mininterval=1.0):
        sample = dataset[idx]
        prompt = collator._prepare_text(sample)
        inputs = _prepare_generation_inputs(
            processor=processor,
            prompt=prompt,
            primary_image=sample["primary_image"],
            device=args.device,
            dtype=torch.bfloat16 if args.bf16 and args.device.startswith("cuda") else None,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max(args.max_new_tokens, (args.force_k or 0) + 4),
                eos_token_id=None if args.force_k is not None else eos_token_id,
                do_sample=False,
            )[0]

        generated_ids, hit_eos = _split_generation(
            generated=generated,
            input_len=inputs["input_ids"].shape[1],
            eos_token_id=eos_token_id,
            max_action_tokens=max(max_action_tokens, (args.force_k or 0) + 4),
        )
        pred_oat_ids, hit_eos = _collect_action_ids(
            generated_ids,
            eos_token_id=eos_token_id,
            act_id_to_oat=act_id_to_oat,
            max_action_tokens=max_action_tokens,
            force_k=args.force_k,
        )
        generated_action_token_ids = [vocab_map["action_token_ids"][idx] for idx in pred_oat_ids]
        target_ids = sample["action_token_ids"][:-1].tolist()
        eos_hits += int(hit_eos)
        generated_hist[len(pred_oat_ids)] += 1
        target_hist[len(target_ids)] += 1
        exact_matches += int(generated_action_token_ids == target_ids)
        aligned = max(len(target_ids), len(generated_action_token_ids), 1)
        matches = sum(int(a == b) for a, b in zip(generated_action_token_ids, target_ids))
        token_accuracy_sum += matches / aligned
        if oat_tok is not None and action_root is not None and action_horizon is not None:
            gt_start = int(dataset.annotations["buffer_start_indices"][idx])
            gt_action = torch.as_tensor(
                action_root[gt_start : gt_start + action_horizon],
                dtype=torch.float32,
                device=args.device,
            ).unsqueeze(0)
            pred_tokens = torch.tensor([pred_oat_ids], dtype=torch.long, device=args.device)
            with torch.inference_mode():
                pred_action = oat_tok.detokenize(pred_tokens).float()
            decoded_action_mse_sum += float(F.mse_loss(pred_action, gt_action).item())
            decoded_action_mse_count += 1

        if idx < 32:
            records.append(
                {
                    "index": idx,
                    "prompt": sample["prompt"],
                    "optimal_k": int(sample["optimal_k"]),
                    "target_ids": target_ids,
                    "generated_ids": generated_action_token_ids,
                    "generated_oat_ids": pred_oat_ids,
                    "hit_eos": hit_eos,
                }
            )

    metrics = {
        "model": args.model,
        "annotations": args.annotations,
        "num_samples": num_samples,
        "eos_hit_rate": eos_hits / max(num_samples, 1),
        "avg_generated_k": sum(k * c for k, c in generated_hist.items()) / max(num_samples, 1),
        "avg_target_k": sum(k * c for k, c in target_hist.items()) / max(num_samples, 1),
        "exact_match_rate": exact_matches / max(num_samples, 1),
        "mean_token_accuracy": token_accuracy_sum / max(num_samples, 1),
        "mean_decoded_action_mse": (
            decoded_action_mse_sum / max(decoded_action_mse_count, 1)
            if decoded_action_mse_count > 0 else None
        ),
        "force_k": args.force_k,
        "generated_k_hist": dict(sorted(generated_hist.items())),
        "target_k_hist": dict(sorted(target_hist.items())),
        "sample_predictions": records,
    }

    out_path = ensure_parent(args.output)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
