"""Shared helpers for adaptive VLA stage: action-token vocab and annotation loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForVision2Seq as OpenVLAAutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as OpenVLAAutoModel


def make_action_tokens(codebook_size: int) -> list[str]:
    return [f"<ACT_{idx}>" for idx in range(int(codebook_size))]


def make_action_eos_token() -> str:
    return "<ACT_EOS>"


def load_annotation_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return torch.load(path, map_location="cpu", weights_only=False)


def maybe_sample_larger_k(
    optimal_k: int,
    candidate_ks: Sequence[int],
    random_larger_k_prob: float,
    rng,
) -> int:
    if random_larger_k_prob <= 0.0:
        return int(optimal_k)
    if rng.random() >= random_larger_k_prob:
        return int(optimal_k)
    valid = [int(k) for k in candidate_ks if int(k) >= int(optimal_k)]
    return int(rng.choice(valid))


def load_processor_and_tokenizer(
    model_or_adapter: str | Path,
    processor_name: str | None = None,
    trust_remote_code: bool = False,
):
    model_or_adapter_str = str(model_or_adapter)
    model_or_adapter_path = Path(model_or_adapter_str)
    adapter_config_path = model_or_adapter_path / "adapter_config.json"
    base_name = None
    if adapter_config_path.exists():
        base_name = json.loads(adapter_config_path.read_text())["base_model_name_or_path"]
    processor_source = processor_name or base_name or model_or_adapter_str

    try:
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=trust_remote_code,
        )
    except OSError:
        if base_name is None or processor_source == base_name:
            raise
        processor = AutoProcessor.from_pretrained(
            base_name,
            trust_remote_code=trust_remote_code,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_or_adapter_str,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    return processor, tokenizer, base_name


def save_action_token_rows(
    model,
    action_vocab_map: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    all_token_ids = list(action_vocab_map["action_token_ids"])
    action_eos_token_id = action_vocab_map.get("action_eos_token_id")
    if action_eos_token_id is not None:
        all_token_ids.append(int(action_eos_token_id))
    token_ids = torch.as_tensor(all_token_ids, dtype=torch.long)
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    payload: dict[str, Any] = {"token_ids": token_ids.cpu()}
    payload["input_rows"] = input_emb.weight.detach().index_select(0, token_ids.to(input_emb.weight.device)).cpu()
    if output_emb is not None and hasattr(output_emb, "weight"):
        payload["output_rows"] = output_emb.weight.detach().index_select(0, token_ids.to(output_emb.weight.device)).cpu()
    out_path = output_dir / "action_token_rows.pt"
    torch.save(payload, out_path)
    return out_path


def maybe_restore_action_token_rows(model, model_or_adapter: str | Path) -> bool:
    model_or_adapter = Path(model_or_adapter)
    rows_path = model_or_adapter / "action_token_rows.pt"
    if not rows_path.exists():
        return False
    payload = torch.load(rows_path, map_location="cpu", weights_only=False)
    token_ids = payload["token_ids"].to(dtype=torch.long)
    input_emb = model.get_input_embeddings()
    input_emb.weight.data.index_copy_(
        0,
        token_ids.to(input_emb.weight.device),
        payload["input_rows"].to(device=input_emb.weight.device, dtype=input_emb.weight.dtype),
    )
    output_emb = model.get_output_embeddings()
    if output_emb is not None and hasattr(output_emb, "weight") and "output_rows" in payload:
        output_emb.weight.data.index_copy_(
            0,
            token_ids.to(output_emb.weight.device),
            payload["output_rows"].to(device=output_emb.weight.device, dtype=output_emb.weight.dtype),
        )
    return True


def load_openvla_model(
    model_or_adapter: str | Path,
    tokenizer,
    base_model_name: str | Path | None = None,
    trust_remote_code: bool = False,
    torch_dtype=None,
    device: str | None = None,
    trainable: bool = False,
):
    model_or_adapter = Path(model_or_adapter)
    adapter_config_path = model_or_adapter / "adapter_config.json"
    base_name = str(model_or_adapter)
    is_adapter = adapter_config_path.exists()
    if is_adapter:
        base_name = json.loads(adapter_config_path.read_text())["base_model_name_or_path"]
    if base_model_name is not None:
        base_name = str(base_model_name)

    model = OpenVLAAutoModel.from_pretrained(
        base_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    maybe_restore_action_token_rows(model, model_or_adapter)
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(model_or_adapter), is_trainable=trainable)
    if device is not None:
        model = model.to(device)
    return model
