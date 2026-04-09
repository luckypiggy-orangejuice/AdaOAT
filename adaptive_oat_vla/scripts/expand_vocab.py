"""Expand an HF tokenizer/model with OAT action tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_oat_vla.common import ensure_parent
from adaptive_oat_vla.common_vla import make_action_eos_token, make_action_tokens


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base HF model path or repo id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codebook-size", type=int, default=1000)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = ensure_parent(Path(args.output_dir) / "model.sentinel").parent
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    action_tokens = make_action_tokens(args.codebook_size)
    action_eos_token = make_action_eos_token()
    added = tokenizer.add_tokens(action_tokens + [action_eos_token], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    act_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in action_tokens]
    action_eos_token_id = tokenizer.convert_tokens_to_ids(action_eos_token)
    payload = {
        "base_model": args.model,
        "codebook_size": args.codebook_size,
        "num_added_tokens": added,
        "action_tokens": action_tokens,
        "action_token_ids": act_token_ids,
        "action_eos_token": action_eos_token,
        "action_eos_token_id": action_eos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
    }
    metadata_path = output_dir / "action_vocab_map.json"
    metadata_path.write_text(json.dumps(payload, indent=2))
    print(f"saved_model={output_dir}")
    print(f"saved_vocab_map={metadata_path}")
    print(f"num_added_tokens={added}")


if __name__ == "__main__":
    main()
