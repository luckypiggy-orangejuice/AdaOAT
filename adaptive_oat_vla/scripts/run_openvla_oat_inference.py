"""Run OpenVLA-style generation and map generated action tokens back through OAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adaptive_oat_vla.common import load_oat_tokenizer
from adaptive_oat_vla.common_vla import (
    load_annotation_payload,
    load_openvla_model,
    load_processor_and_tokenizer,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--action-vocab-map", required=True)
    parser.add_argument("--oat-checkpoint", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--agentview-image", required=True)
    parser.add_argument("--wrist-image", default=None)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    args = build_argparser().parse_args()
    processor, tokenizer, _ = load_processor_and_tokenizer(
        model_or_adapter=args.model,
        processor_name=args.processor,
        trust_remote_code=args.trust_remote_code,
    )
    model = load_openvla_model(
        model_or_adapter=args.model,
        tokenizer=tokenizer,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    oat_tok, _ = load_oat_tokenizer(args.oat_checkpoint, device=args.device)
    vocab_map = load_annotation_payload(args.action_vocab_map)
    act_id_to_oat = {
        int(v): idx for idx, v in enumerate(vocab_map["action_token_ids"])
    }
    eos_token_id = int(vocab_map.get("action_eos_token_id", vocab_map["eos_token_id"]))

    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"
    agentview = load_rgb(args.agentview_image)
    if args.wrist_image:
        images = [agentview, load_rgb(args.wrist_image)]
    else:
        images = agentview

    inputs = processor(prompt, images, return_tensors="pt")
    inputs = {
        k: (v.to(device=args.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(args.device))
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=False,
        )[0]

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[input_len:]
    if (new_tokens == eos_token_id).any():
        eos_pos = int((new_tokens == eos_token_id).nonzero()[0].item())
        new_tokens = new_tokens[:eos_pos]
    else:
        new_tokens = new_tokens[:8]

    oat_indices = [act_id_to_oat[int(tok.item())] for tok in new_tokens if int(tok.item()) in act_id_to_oat]
    if not oat_indices:
        raise RuntimeError("No generated action tokens mapped back to OAT codebook ids.")

    oat_tensor = torch.tensor([oat_indices], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        action = oat_tok.detokenize(oat_tensor)[:, :16, :].detach().cpu().numpy()

    payload = {
        "instruction": args.instruction,
        "generated_token_ids": [int(x) for x in new_tokens.tolist()],
        "oat_indices": oat_indices,
        "action": np.asarray(action).tolist(),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"saved_inference={args.output}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
