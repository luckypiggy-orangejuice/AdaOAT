"""Stage-3 SFT skeleton for adaptive OAT-VLA using HF causal LMs/processors."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sys
import traceback
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, AutoTokenizer

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )

    HAS_FSDP = True
except Exception:
    HAS_FSDP = False

try:
    from transformers import AutoModelForVision2Seq as OpenVLAAutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as OpenVLAAutoModel

from adaptive_oat_vla.common import ensure_parent
from adaptive_oat_vla.common_vla import (
    load_annotation_payload,
    load_openvla_model,
    load_processor_and_tokenizer,
    make_action_eos_token,
    make_action_tokens,
    save_action_token_rows,
)
from adaptive_oat_vla.vla_dataset import AdaptiveVLACollator, AdaptiveVLADataset


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Expanded model dir or base HF repo")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--action-vocab-map", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processor", default=None, help="Optional processor repo/path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--codebook-size", type=int, default=1000)
    parser.add_argument("--image-key", default="agentview_rgb")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eos-loss-weight", type=float, default=1.0)
    parser.add_argument("--random-larger-k-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--val-annotations", default=None)
    parser.add_argument("--val-random-larger-k-prob", type=float, default=0.0)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--generation-eval-samples", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--keep-top-k-ckpts", type=int, default=3)
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--step-metrics-file", default=None)
    parser.add_argument("--use-data-parallel", action="store_true")
    parser.add_argument("--distributed-backend", default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-trainer-state", default=None)
    return parser


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def is_fsdp_model(model) -> bool:
    return HAS_FSDP and isinstance(model, FSDP)


def require_collective_checkpoint(model) -> bool:
    return is_distributed() and is_fsdp_model(model)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def maybe_init_distributed() -> tuple[torch.device, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank
    return torch.device("cpu"), local_rank


def maybe_apply_lora(model, args):
    if not args.use_lora:
        return model, False
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "peft is not installed, but --use-lora was requested."
        ) from exc

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # New <ACT_i> tokens are introduced via embedding resize, so keep token embeddings and lm_head trainable.
    input_emb = model.get_input_embeddings()
    if input_emb is not None:
        input_emb.weight.requires_grad_(True)
    output_emb = model.get_output_embeddings()
    if output_emb is not None and hasattr(output_emb, "weight"):
        output_emb.weight.requires_grad_(True)
    return model, True


def ensure_action_vocab(processor, model, codebook_size: int, output_dir: Path) -> Path:
    action_tokens = make_action_tokens(codebook_size)
    action_eos_token = make_action_eos_token()
    tokenizer = processor.tokenizer
    existing_ids = []
    missing = []
    unk_id = tokenizer.unk_token_id
    for tok in action_tokens + [action_eos_token]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id == unk_id:
            missing.append(tok)
        else:
            existing_ids.append(tok_id)

    added = 0
    if missing:
        added = tokenizer.add_tokens(missing, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    act_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in action_tokens]
    action_eos_token_id = tokenizer.convert_tokens_to_ids(action_eos_token)
    payload = {
        "codebook_size": codebook_size,
        "num_added_tokens": added,
        "action_tokens": action_tokens,
        "action_token_ids": act_token_ids,
        "action_eos_token": action_eos_token,
        "action_eos_token_id": action_eos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
    }
    vocab_map_path = output_dir / "action_vocab_map.json"
    vocab_map_path.write_text(json.dumps(payload, indent=2))
    return vocab_map_path


def compute_loss_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_mask: torch.Tensor,
    eos_token_id: int,
    eos_loss_weight: float,
) -> dict[str, torch.Tensor]:
    labels, action_mask = align_labels_and_masks(logits, labels, action_mask)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = action_mask[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    act_mask = (shift_mask & (shift_labels != eos_token_id)).reshape(-1)
    eos_mask = (shift_mask & (shift_labels == eos_token_id)).reshape(-1)
    weighted_per_token = per_token
    if eos_loss_weight != 1.0:
        weighted_per_token = per_token * torch.where(
            eos_mask,
            per_token.new_full((per_token.numel(),), float(eos_loss_weight)),
            per_token.new_ones((per_token.numel(),)),
        )
    mask_flat = shift_mask.reshape(-1).float()
    def masked_mean_tensor(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    return {
        "loss_weighted": masked_mean_tensor(weighted_per_token, mask_flat),
        "loss_unweighted": masked_mean_tensor(per_token, mask_flat),
        "loss_action_tokens": masked_mean_tensor(per_token, act_mask),
        "loss_eos": masked_mean_tensor(per_token, eos_mask),
        "eos_fraction": eos_mask.float().sum() / mask_flat.sum().clamp_min(1.0),
    }


def align_labels_and_masks(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.size(1) != labels.size(1):
        delta = logits.size(1) - labels.size(1)
        if delta > 0:
            # Vision-language models may prepend image tokens internally, so pad labels/masks
            # on the left to keep action supervision aligned to the generated text positions.
            pad_labels = torch.full(
                (labels.size(0), delta),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            pad_mask = torch.zeros(
                (action_mask.size(0), delta),
                dtype=action_mask.dtype,
                device=action_mask.device,
            )
            labels = torch.cat([pad_labels, labels], dim=1)
            action_mask = torch.cat([pad_mask, action_mask], dim=1)
        else:
            labels = labels[:, -logits.size(1) :]
            action_mask = action_mask[:, -logits.size(1) :]
    return labels, action_mask


def compute_batch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_mask: torch.Tensor,
    eos_token_id: int,
    eos_loss_weight: float = 1.0,
) -> dict[str, float]:
    labels, action_mask = align_labels_and_masks(logits, labels, action_mask)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = action_mask[:, 1:].contiguous()

    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    preds = shift_logits.argmax(dim=-1)

    act_mask = shift_mask & (shift_labels != eos_token_id)
    eos_mask = shift_mask & (shift_labels == eos_token_id)
    pred_eos_mask = shift_mask & (preds == eos_token_id)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if mask.sum().item() == 0:
            return float("nan")
        return float(values[mask].float().mean().item())

    def masked_acc(mask: torch.Tensor) -> float:
        if mask.sum().item() == 0:
            return float("nan")
        return float((preds[mask] == shift_labels[mask]).float().mean().item())

    components = compute_loss_components(
        logits,
        labels,
        action_mask,
        eos_token_id=eos_token_id,
        eos_loss_weight=eos_loss_weight,
    )
    eos_tp = (pred_eos_mask & eos_mask).sum().item()
    eos_precision = float(eos_tp / max(pred_eos_mask.sum().item(), 1))
    eos_recall = float(eos_tp / max(eos_mask.sum().item(), 1))
    return {
        "loss": float(components["loss_unweighted"].item()),
        "loss_weighted": float(components["loss_weighted"].item()),
        "loss_action_tokens": float(components["loss_action_tokens"].item()),
        "loss_eos": float(components["loss_eos"].item()),
        "action_token_accuracy": masked_acc(act_mask),
        "eos_recall": eos_recall,
        "eos_precision": eos_precision,
        "eos_fraction": float(components["eos_fraction"].item()),
    }


def batch_to_device(batch: dict, device: str) -> tuple[dict, torch.Tensor, torch.Tensor]:
    labels = batch["labels"].to(device)
    action_mask = batch["action_mask"].to(device)
    model_inputs = {}
    for key, value in batch.items():
        if key in {"labels", "action_mask", "prompts", "chosen_k", "optimal_k"}:
            continue
        if torch.is_tensor(value):
            model_inputs[key] = value.to(device)
    return model_inputs, labels, action_mask


def evaluate(
    model,
    loader: DataLoader,
    device: str,
    bf16: bool,
    eos_token_id: int,
    eos_loss_weight: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "loss_weighted": 0.0,
        "loss_action_tokens": 0.0,
        "loss_eos": 0.0,
        "action_token_accuracy": 0.0,
        "eos_precision": 0.0,
        "eos_recall": 0.0,
        "eos_fraction": 0.0,
    }
    count = 0
    autocast_enabled = bf16 and device.startswith("cuda")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            model_inputs, labels, action_mask = batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(**model_inputs)
                metrics = compute_batch_metrics(
                    outputs.logits.float(),
                    labels,
                    action_mask,
                    eos_token_id=eos_token_id,
                    eos_loss_weight=eos_loss_weight,
                )
            for key in totals:
                value = metrics[key]
                if not math.isnan(value):
                    totals[key] += float(value)
            count += 1
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    model.train()
    if count == 0:
        return {f"val_{k}": float("nan") for k in totals}
    return {f"val_{k}": v / count for k, v in totals.items()}


def evaluate_generation_metrics(
    model,
    dataset: AdaptiveVLADataset,
    collator: AdaptiveVLACollator,
    processor,
    eos_token_id: int,
    device: str,
    bf16: bool,
    max_samples: int,
) -> dict[str, float | dict[str, int]]:
    model.eval()
    eval_model = unwrap_model(model)
    k_hist: dict[int, int] = {}
    total_k = 0
    max_action_tokens = dataset.max_action_tokens
    num_samples = min(len(dataset), max_samples)
    for idx in range(num_samples):
        sample = dataset[idx]
        prompt = collator._prepare_text(sample)
        image = Image.fromarray(sample["primary_image"].detach().cpu().numpy()).convert("RGB")
        proc_inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        proc_inputs = {
            key: (
                value.to(device=device, dtype=torch.bfloat16)
                if torch.is_floating_point(value) and bf16 and device.startswith("cuda")
                else value.to(device)
            )
            for key, value in proc_inputs.items()
        }
        with torch.inference_mode():
            generated = eval_model.generate(
                **proc_inputs,
                max_new_tokens=max_action_tokens + 1,
                eos_token_id=eos_token_id,
                do_sample=False,
            )[0]
        input_len = proc_inputs["input_ids"].shape[1]
        gen_tokens = generated[input_len:].tolist()
        if eos_token_id in gen_tokens:
            gen_tokens = gen_tokens[: gen_tokens.index(eos_token_id)]
        else:
            gen_tokens = gen_tokens[:max_action_tokens]
        k = len(gen_tokens)
        k_hist[k] = k_hist.get(k, 0) + 1
        total_k += k
    model.train()
    return {
        "val_avg_predicted_k": total_k / max(num_samples, 1),
        "val_k_distribution": {str(k): v for k, v in sorted(k_hist.items())},
    }


def create_scheduler(optimizer, total_optimizer_steps: int, warmup_steps: int, peak_lr: float, min_lr: float):
    if total_optimizer_steps <= 0:
        return None

    min_ratio = min_lr / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(total_optimizer_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(
    model,
    tokenizer,
    action_vocab_map: dict,
    checkpoint_dir: Path,
    *,
    epoch: int,
    global_step: int,
    batch_idx: int | None,
    optimizer=None,
    scheduler=None,
    extra_metadata: dict | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_model = unwrap_model(model)
    fsdp_collective = require_collective_checkpoint(model)

    def _safe_save_action_rows() -> None:
        try:
            save_action_token_rows(base_model, action_vocab_map, checkpoint_dir)
        except Exception as exc:
            if is_main_process():
                print(f"warning=save_action_token_rows_failed checkpoint={checkpoint_dir} error={exc}", flush=True)

    if fsdp_collective:
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            state_dict = model.state_dict()
        save_exc = None
        if is_main_process():
            try:
                base_model.save_pretrained(checkpoint_dir, state_dict=state_dict)
                tokenizer.save_pretrained(checkpoint_dir)
                _safe_save_action_rows()
            except Exception as exc:
                save_exc = exc
        dist.barrier()
        if save_exc is not None:
            raise save_exc
    else:
        base_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        _safe_save_action_rows()

    if not is_main_process():
        return

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "batch_idx": batch_idx,
    }
    # FSDP optimizer state serialization can fail depending on torch version
    # and optimizer backend; keep checkpoints robust by skipping it there.
    if optimizer is not None and not fsdp_collective:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None and not fsdp_collective:
        state["scheduler"] = scheduler.state_dict()
    if extra_metadata:
        state["metadata"] = extra_metadata
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    progress = {
        "epoch": epoch,
        "global_step": global_step,
        "batch_idx": batch_idx,
        "has_optimizer_state": (optimizer is not None and not fsdp_collective),
        "has_scheduler_state": (scheduler is not None and not fsdp_collective),
    }
    if extra_metadata:
        progress.update(extra_metadata)
    (checkpoint_dir / "progress.json").write_text(json.dumps(progress, indent=2))


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = ensure_parent(Path(args.output_dir) / "model.sentinel").parent
    peak_lr = args.peak_lr or args.lr
    device, local_rank = maybe_init_distributed()
    world_size = get_world_size()
    main_process = is_main_process()
    if args.device.startswith("cuda") and device.type == "cuda":
        args.device = str(device)

    processor_name = args.processor
    adapter_config_path = Path(args.model) / "adapter_config.json"
    is_adapter_init = adapter_config_path.exists()
    try:
        processor, tokenizer, _ = load_processor_and_tokenizer(
            model_or_adapter=args.model,
            processor_name=processor_name,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else None
    if is_adapter_init:
        model = load_openvla_model(
            model_or_adapter=args.model,
            tokenizer=tokenizer,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            trainable=True,
        )
    else:
        model = OpenVLAAutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            # Non-reentrant checkpointing is more stable with DDP full-finetune.
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    generated_vocab_map = ensure_action_vocab(
        processor or type("P", (), {"tokenizer": tokenizer})(),
        model,
        args.codebook_size,
        output_dir,
    )
    if processor is not None and hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    action_vocab_map_path = args.action_vocab_map or str(generated_vocab_map)
    action_vocab_map = load_annotation_payload(action_vocab_map_path)
    lora_enabled = hasattr(model, "peft_config")
    if not lora_enabled:
        model, lora_enabled = maybe_apply_lora(model, args)
    else:
        input_emb = model.get_input_embeddings()
        if input_emb is not None:
            input_emb.weight.requires_grad_(True)
        output_emb = model.get_output_embeddings()
        if output_emb is not None and hasattr(output_emb, "weight"):
            output_emb.weight.requires_grad_(True)
    model.to(device)
    if args.use_data_parallel and main_process:
        print("warning=use_data_parallel_ignored distributed_training_uses_ddp")
    if world_size > 1:
        if args.distributed_backend == "fsdp":
            if not HAS_FSDP:
                raise RuntimeError("FSDP requested, but torch.distributed.fsdp is unavailable.")
            mixed_precision = None
            if args.bf16:
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                device_id=local_rank if device.type == "cuda" else None,
                use_orig_params=True,
                forward_prefetch=False,
                limit_all_gathers=True,
                sync_module_states=False,
            )
        else:
            # Full fine-tuning + gradient checkpointing is more stable when DDP
            # treats the graph as static and does not search for unused params.
            find_unused_parameters = False
            model = DDP(
                model,
                device_ids=[local_rank],
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=False,
            )

    dataset = AdaptiveVLADataset(
        annotation_path=args.annotations,
        action_vocab_map_path=action_vocab_map_path,
        random_larger_k_prob=args.random_larger_k_prob,
        image_key=args.image_key,
    )
    collator = AdaptiveVLACollator(processor=processor, tokenizer=tokenizer)
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=get_rank(), shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = None
    val_dataset = None
    if args.val_annotations:
        val_dataset = AdaptiveVLADataset(
            annotation_path=args.val_annotations,
            action_vocab_map_path=action_vocab_map_path,
            random_larger_k_prob=args.val_random_larger_k_prob,
            image_key=args.image_key,
        )
        if main_process:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collator,
            )

    # Disable foreach kernels to reduce optimizer-step peak memory in full FT.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    steps_per_epoch = math.ceil(len(loader) / max(args.grad_accumulation_steps, 1))
    total_optimizer_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.max_steps)
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps=args.warmup_steps,
        peak_lr=peak_lr,
        min_lr=args.min_lr,
    )
    resume_state = None
    start_epoch = 0
    initial_global_step = 0
    if args.resume_trainer_state:
        resume_state = torch.load(args.resume_trainer_state, map_location="cpu", weights_only=False)
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        if scheduler is not None and "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        initial_global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        if main_process:
            print(
                f"resumed_trainer_state={args.resume_trainer_state} "
                f"start_epoch={start_epoch} initial_global_step={initial_global_step}",
                flush=True,
            )
    metrics_path = None
    if args.metrics_file:
        metrics_path = ensure_parent(args.metrics_file)
    step_metrics_path = None
    if args.step_metrics_file:
        step_metrics_path = ensure_parent(args.step_metrics_file)
    wandb_run = None
    if args.wandb_mode != "disabled" and main_process:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project or "adaptive-oat-vla",
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    global_step = initial_global_step
    stop_training = False
    best_val_loss = math.inf
    best_dir = output_dir / "best"
    latest_dir = output_dir / "latest"
    autocast_enabled = args.bf16 and args.device.startswith("cuda")
    no_improve_count = 0
    saved_ckpts: list[tuple[float, Path]] = []
    eos_token_id = int(action_vocab_map.get("action_eos_token_id", action_vocab_map["eos_token_id"]))
    last_batch_idx = None
    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_totals = {
                "loss_weighted": 0.0,
                "loss_unweighted": 0.0,
                "loss_action_tokens": 0.0,
                "loss_eos": 0.0,
                "action_token_accuracy": 0.0,
                "eos_precision": 0.0,
                "eos_recall": 0.0,
                "eos_fraction": 0.0,
            }
            count = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(loader):
                last_batch_idx = batch_idx
                model_inputs, labels, action_mask = batch_to_device(batch, args.device)
                sync_context = (
                    model.no_sync()
                    if world_size > 1
                    and ((batch_idx + 1) % max(args.grad_accumulation_steps, 1) != 0)
                    and ((batch_idx + 1) != len(loader))
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                        outputs = model(**model_inputs)
                        loss_components = compute_loss_components(
                            outputs.logits.float(),
                            labels,
                            action_mask,
                            eos_token_id=eos_token_id,
                            eos_loss_weight=args.eos_loss_weight,
                        )
                        loss = loss_components["loss_weighted"]
                        scaled_loss = loss / max(args.grad_accumulation_steps, 1)

                    scaled_loss.backward()
                train_metrics = compute_batch_metrics(
                    outputs.logits.float().detach(),
                    labels,
                    action_mask,
                    eos_token_id=eos_token_id,
                    eos_loss_weight=args.eos_loss_weight,
                )
                should_step = ((batch_idx + 1) % max(args.grad_accumulation_steps, 1) == 0) or ((batch_idx + 1) == len(loader))
                if should_step:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                train_totals["loss_weighted"] += float(loss_components["loss_weighted"].item())
                train_totals["loss_unweighted"] += float(loss_components["loss_unweighted"].item())
                train_totals["loss_action_tokens"] += float(loss_components["loss_action_tokens"].item())
                train_totals["loss_eos"] += float(loss_components["loss_eos"].item())
                for key in ("action_token_accuracy", "eos_precision", "eos_recall", "eos_fraction"):
                    value = train_metrics[key]
                    if not math.isnan(value):
                        train_totals[key] += float(value)
                count += 1
                if main_process and should_step and args.log_every > 0 and (global_step % args.log_every == 0):
                    train_log = {
                        "step": global_step,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "train_loss_step": float(loss_components["loss_weighted"].item()),
                        "train_loss_unweighted_step": float(loss_components["loss_unweighted"].item()),
                        "train_loss_action_tokens_step": float(loss_components["loss_action_tokens"].item()),
                        "train_loss_eos_step": float(loss_components["loss_eos"].item()),
                        "train_action_token_accuracy_step": float(train_metrics["action_token_accuracy"]),
                        "train_eos_precision_step": float(train_metrics["eos_precision"]),
                        "train_eos_recall_step": float(train_metrics["eos_recall"]),
                        "train_eos_fraction_step": float(train_metrics["eos_fraction"]),
                        "chosen_k_mean": float(batch["chosen_k"].float().mean().item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    print(
                        f"step={global_step} epoch={epoch} batch={batch_idx} "
                        f"train_loss={loss_components['loss_weighted'].item():.6f} "
                        f"action_loss={loss_components['loss_action_tokens'].item():.6f} "
                        f"eos_loss={loss_components['loss_eos'].item():.6f} "
                        f"chosen_k_mean={batch['chosen_k'].float().mean().item():.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.7f}"
                    )
                    if step_metrics_path is not None:
                        with step_metrics_path.open("a") as handle:
                            handle.write(json.dumps(train_log) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(train_log, step=global_step)
                should_save_step = (
                    (main_process or require_collective_checkpoint(model))
                    and should_step
                    and args.save_every > 0
                    and (global_step % args.save_every == 0)
                )
                if should_save_step:
                    ckpt_dir = output_dir / f"step-{global_step}"
                    save_training_checkpoint(
                        model,
                        tokenizer,
                        action_vocab_map,
                        ckpt_dir,
                        epoch=epoch,
                        global_step=global_step,
                        batch_idx=batch_idx,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    save_training_checkpoint(
                        model,
                        tokenizer,
                        action_vocab_map,
                        latest_dir,
                        epoch=epoch,
                        global_step=global_step,
                        batch_idx=batch_idx,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra_metadata={"source": f"step-{global_step}"},
                    )
                if should_step and args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    break

            train_loss = train_totals["loss_weighted"] / max(count, 1)
            log_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_loss_unweighted": train_totals["loss_unweighted"] / max(count, 1),
                "train_loss_action_tokens": train_totals["loss_action_tokens"] / max(count, 1),
                "train_loss_eos": train_totals["loss_eos"] / max(count, 1),
                "train_action_token_accuracy": train_totals["action_token_accuracy"] / max(count, 1),
                "train_eos_precision": train_totals["eos_precision"] / max(count, 1),
                "train_eos_recall": train_totals["eos_recall"] / max(count, 1),
                "train_eos_fraction": train_totals["eos_fraction"] / max(count, 1),
            }
            should_run_val = main_process and val_loader is not None and ((epoch + 1) % max(args.val_every_epochs, 1) == 0 or stop_training)
            if should_run_val:
                val_metrics = evaluate(
                    model=unwrap_model(model),
                    loader=val_loader,
                    device=args.device,
                    bf16=args.bf16,
                    eos_token_id=eos_token_id,
                    eos_loss_weight=args.eos_loss_weight,
                    max_batches=args.max_val_batches,
                )
                log_payload.update(val_metrics)
                gen_metrics = evaluate_generation_metrics(
                    model=unwrap_model(model),
                    dataset=val_dataset,
                    collator=collator,
                    processor=processor,
                    eos_token_id=eos_token_id,
                    device=args.device,
                    bf16=args.bf16,
                    max_samples=args.generation_eval_samples,
                )
                log_payload.update(gen_metrics)
                val_loss = float(log_payload["val_loss"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    save_training_checkpoint(
                        model,
                        tokenizer,
                        action_vocab_map,
                        best_dir,
                        epoch=epoch,
                        global_step=global_step,
                        batch_idx=last_batch_idx,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra_metadata={"metrics": log_payload},
                    )
                    (best_dir / "best_metrics.json").write_text(json.dumps(log_payload, indent=2))
                else:
                    no_improve_count += 1
                if args.save_every_epochs > 0 and ((epoch + 1) % args.save_every_epochs == 0 or stop_training):
                    epoch_dir = output_dir / f"epoch-{epoch+1:03d}"
                    save_training_checkpoint(
                        model,
                        tokenizer,
                        action_vocab_map,
                        epoch_dir,
                        epoch=epoch,
                        global_step=global_step,
                        batch_idx=last_batch_idx,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra_metadata={"metrics": log_payload},
                    )
                    (epoch_dir / "metrics.json").write_text(json.dumps(log_payload, indent=2))
                    saved_ckpts.append((val_loss, epoch_dir))
                    saved_ckpts = sorted(saved_ckpts, key=lambda x: x[0])
                    while len(saved_ckpts) > max(args.keep_top_k_ckpts, 0):
                        _, stale_path = saved_ckpts.pop(-1)
                        if stale_path.exists():
                            shutil.rmtree(stale_path)
            if is_distributed():
                dist.barrier()
            if main_process and metrics_path is not None:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(log_payload) + "\n")
            if main_process:
                print(" ".join(
                    f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={json.dumps(v)}" if isinstance(v, dict) else f"{k}={v}"
                    for k, v in log_payload.items()
                ))
            if main_process and wandb_run is not None:
                wandb_run.log(log_payload, step=global_step)
            should_save_latest_epoch = args.save_every_epochs > 0
            if should_save_latest_epoch and (main_process or require_collective_checkpoint(model)):
                save_training_checkpoint(
                    model,
                    tokenizer,
                    action_vocab_map,
                    latest_dir,
                    epoch=epoch,
                    global_step=global_step,
                    batch_idx=last_batch_idx,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_metadata={"metrics": log_payload, "source": "epoch-end"},
                )
            if main_process and args.early_stopping_patience > 0 and should_run_val and no_improve_count >= args.early_stopping_patience:
                print(f"early_stop_triggered epoch={epoch} no_improve_count={no_improve_count}")
                break
            if stop_training:
                break
    except Exception as exc:
        should_save_crash = main_process or require_collective_checkpoint(model)
        if should_save_crash:
            crash_dir = output_dir / f"crash-step-{global_step}"
            save_training_checkpoint(
                model,
                tokenizer,
                action_vocab_map,
                crash_dir,
                epoch=epoch if 'epoch' in locals() else -1,
                global_step=global_step,
                batch_idx=last_batch_idx,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_metadata={
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc() if main_process else "",
                },
            )
            save_training_checkpoint(
                model,
                tokenizer,
                action_vocab_map,
                latest_dir,
                epoch=epoch if 'epoch' in locals() else -1,
                global_step=global_step,
                batch_idx=last_batch_idx,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_metadata={
                    "source": crash_dir.name,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                },
            )
        raise

    if main_process or require_collective_checkpoint(model):
        save_training_checkpoint(
            model,
            tokenizer,
            action_vocab_map,
            output_dir,
            epoch=epoch if 'epoch' in locals() else -1,
            global_step=global_step,
            batch_idx=last_batch_idx,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_metadata={"source": "final"},
        )

    if main_process:
        metadata = {
        "annotations": args.annotations,
        "action_vocab_map": action_vocab_map_path,
        "generated_action_vocab_map": str(generated_vocab_map),
        "processor": processor_name if processor is not None else None,
        "trust_remote_code": args.trust_remote_code,
        "lora_enabled": lora_enabled,
        "resumed_from_adapter": is_adapter_init,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "lr": peak_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "eos_loss_weight": args.eos_loss_weight,
        "val_every_epochs": args.val_every_epochs,
        "save_every_epochs": args.save_every_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "keep_top_k_ckpts": args.keep_top_k_ckpts,
        "val_annotations": args.val_annotations,
        "best_val_loss": None if best_val_loss == math.inf else best_val_loss,
        "world_size": world_size,
        "resume_trainer_state": args.resume_trainer_state,
        "start_epoch": start_epoch,
        }
        (output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))
        if wandb_run is not None:
            wandb_run.finish()
        print(f"saved_adaptive_vla={output_dir}")
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
