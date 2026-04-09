"""Evaluate a Stage-3 OpenVLA checkpoint on LIBERO by detokenizing into OAT actions."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import hydra
import numpy as np
import torch
from PIL import Image

from adaptive_oat_vla.common import ensure_parent, load_oat_tokenizer
from adaptive_oat_vla.common_vla import (
    load_annotation_payload,
    load_openvla_model,
    load_processor_and_tokenizer,
)
from oat.policy.base_policy import BasePolicy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Stage-3 adapter/checkpoint directory.")
    parser.add_argument("--action-vocab-map", required=True)
    parser.add_argument("--oat-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--base-model", default=None, help="Optional local/base model path for adapter loading.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--task-config", default="/workspace/wcz/oat/oat/config/task/policy/libero/libero10.yaml")
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--n-test-vis", type=int, default=None)
    parser.add_argument("--n-parallel-envs", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--image-key", default="agentview_rgb", choices=["agentview_rgb", "robot0_eye_in_hand_rgb"])
    parser.add_argument("--max-new-tokens", type=int, default=9)
    parser.add_argument("--force-k", type=int, default=None, help="Ignore EOS and use the first K valid ACT tokens.")
    return parser


def _format_state(obs_dict: dict[str, Any], batch_idx: int) -> str:
    parts = []
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "task_uid"):
        value = obs_dict[key]
        if isinstance(value, torch.Tensor):
            flat = value[batch_idx, -1].flatten().tolist()
        else:
            flat = torch.as_tensor(value[batch_idx, -1]).flatten().tolist()
        parts.append(f"{key}=" + ",".join(f"{float(v):.4f}" for v in flat))
    return " | ".join(parts)


def _extract_prompt(prompt_batch: Any, batch_idx: int) -> str:
    if isinstance(prompt_batch, (list, tuple)):
        return str(prompt_batch[batch_idx])
    if isinstance(prompt_batch, np.ndarray):
        return str(prompt_batch[batch_idx])
    return str(prompt_batch[batch_idx])


def _to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().to(device="cpu")
    if torch.is_floating_point(image):
        # LiberoRunner casts observations to policy.dtype. Recover display-space RGB here.
        if float(image.max().item()) <= 1.5:
            image = image * 255.0
        image = image.clamp(0, 255).to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    return image.numpy()


@dataclass
class GenerationStats:
    calls: int = 0
    sequences: int = 0
    total_generated_k: int = 0
    invalid_sequences: int = 0
    empty_sequences: int = 0

    def to_dict(self) -> dict[str, float | int]:
        avg_k = self.total_generated_k / max(self.sequences, 1)
        invalid_rate = self.invalid_sequences / max(self.sequences, 1)
        empty_rate = self.empty_sequences / max(self.sequences, 1)
        return {
            "policy_calls": self.calls,
            "num_sequences": self.sequences,
            "avg_generated_k": avg_k,
            "invalid_sequence_rate": invalid_rate,
            "empty_sequence_rate": empty_rate,
        }


class OpenVLAOATPolicy(BasePolicy):
    def __init__(
        self,
        model_dir: str | Path,
        action_vocab_map_path: str | Path,
        oat_checkpoint: str | Path,
        device: str = "cuda:0",
        processor_name: str | None = None,
        base_model_name: str | None = None,
        trust_remote_code: bool = False,
        image_key: str = "agentview_rgb",
        max_new_tokens: int = 9,
        force_k: int | None = None,
    ):
        super().__init__()
        self.image_key = image_key
        self.max_new_tokens = int(max_new_tokens)
        self.force_k = None if force_k is None else int(force_k)
        self._device = torch.device(device)
        self._dtype = torch.bfloat16 if "cuda" in device else torch.float32
        self.processor, self.tokenizer, _ = load_processor_and_tokenizer(
            model_or_adapter=model_dir,
            processor_name=processor_name,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model = load_openvla_model(
            model_or_adapter=model_dir,
            tokenizer=self.tokenizer,
            base_model_name=base_model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self._dtype,
            device=device,
        )
        self.model.eval()
        self.oat_tok, _ = load_oat_tokenizer(oat_checkpoint, device=device)
        vocab_map = load_annotation_payload(action_vocab_map_path)
        self.act_token_id_to_oat = {
            int(v): idx for idx, v in enumerate(vocab_map["action_token_ids"])
        }
        self.eos_token_id = int(vocab_map.get("action_eos_token_id", vocab_map["eos_token_id"]))
        self.stats = GenerationStats()

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def reset(self):
        return None

    def get_observation_encoder(self):
        return None

    def get_observation_modalities(self):
        return ["rgb", "state", "language"]

    def get_observation_ports(self):
        return [
            self.image_key,
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "task_uid",
            "prompt",
        ]

    def get_policy_name(self) -> str:
        return "openvla_oat_stage3"

    def _build_batch_inputs(self, obs_dict: dict[str, Any]) -> tuple[dict[str, torch.Tensor], list[str]]:
        rgb_tensor = obs_dict[self.image_key]
        batch_size = int(rgb_tensor.shape[0])
        images = []
        texts = []
        for batch_idx in range(batch_size):
            rgb = _to_rgb_uint8(rgb_tensor[batch_idx, -1])
            images.append(Image.fromarray(rgb).convert("RGB"))
            prompt = _extract_prompt(obs_dict["prompt"], batch_idx)
            state_text = _format_state(obs_dict, batch_idx)
            texts.append(
                f"In: What action should the robot take to {prompt}?\n"
                f"State: {state_text}\n"
                "Out:"
            )
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return inputs, texts

    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.stats.calls += 1
        inputs, _ = self._build_batch_inputs(obs_dict)
        batch_token_ids: list[list[int]] = []
        batch_size = int(inputs["input_ids"].shape[0])
        for row in range(batch_size):
            row_inputs = {
                k: (
                    v[row : row + 1].to(device=self.device, dtype=torch.bfloat16)
                    if torch.is_floating_point(v)
                    else v[row : row + 1].to(self.device)
                )
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                generated = self.model.generate(
                    **row_inputs,
                    max_new_tokens=max(self.max_new_tokens, (self.force_k or 0) + 4),
                    eos_token_id=None if self.force_k is not None else self.eos_token_id,
                    do_sample=False,
                )

            input_len = int(row_inputs["input_ids"].shape[1])
            new_tokens = generated[0, input_len:]
            oat_indices = []
            invalid = False
            if self.force_k is None:
                if (new_tokens == self.eos_token_id).any():
                    eos_pos = int((new_tokens == self.eos_token_id).nonzero()[0].item())
                    new_tokens = new_tokens[:eos_pos]
                else:
                    new_tokens = new_tokens[:8]
            for tok in new_tokens.tolist():
                tok = int(tok)
                if tok == self.eos_token_id and self.force_k is None:
                    break
                if tok == self.eos_token_id and self.force_k is not None:
                    continue
                if tok in self.act_token_id_to_oat:
                    oat_indices.append(self.act_token_id_to_oat[tok])
                    if self.force_k is not None and len(oat_indices) >= self.force_k:
                        break
                else:
                    invalid = True
            if invalid:
                self.stats.invalid_sequences += 1
            if not oat_indices:
                self.stats.empty_sequences += 1
                # Fall back to a single valid token so rollout can continue.
                oat_indices = [0]
            if self.force_k is not None and len(oat_indices) < self.force_k:
                oat_indices.extend([0] * (self.force_k - len(oat_indices)))
            self.stats.total_generated_k += len(oat_indices)
            self.stats.sequences += 1
            batch_token_ids.append(oat_indices)

        token_tensors = [
            torch.tensor([seq], dtype=torch.long, device=self.device)
            for seq in batch_token_ids
        ]
        with torch.inference_mode():
            action = self.oat_tok.detokenize(token_tensors).detach()
        return {"action": action}


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"eval_output_dir={output_dir}", flush=True)

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.task_config)
    if "n_obs_steps" not in cfg:
        cfg.n_obs_steps = 2
    if "n_action_steps" not in cfg:
        cfg.n_action_steps = 8
    env_kwargs = OmegaConf.to_container(cfg.env_runner, resolve=True)
    env_kwargs.pop("_target_", None)
    if args.n_test is not None:
        env_kwargs["n_test"] = args.n_test
    if args.n_test_vis is not None:
        env_kwargs["n_test_vis"] = args.n_test_vis
    elif env_kwargs.get("n_test_vis") is not None and env_kwargs.get("n_test") is not None:
        env_kwargs["n_test_vis"] = min(int(env_kwargs["n_test_vis"]), int(env_kwargs["n_test"]))
    if args.n_parallel_envs is not None:
        env_kwargs["n_parallel_envs"] = args.n_parallel_envs
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    env_kwargs["output_dir"] = str(output_dir)
    print("loading_policy", flush=True)

    policy = OpenVLAOATPolicy(
        model_dir=args.model,
        action_vocab_map_path=args.action_vocab_map,
        oat_checkpoint=args.oat_checkpoint,
        device=args.device,
        processor_name=args.processor,
        base_model_name=args.base_model,
        trust_remote_code=args.trust_remote_code,
        image_key=args.image_key,
        max_new_tokens=args.max_new_tokens,
        force_k=args.force_k,
    )
    print("policy_ready", flush=True)

    print("initializing_env_runner", flush=True)
    env_runner = hydra.utils.instantiate({"_target_": cfg.env_runner._target_, **env_kwargs})
    print("env_runner_ready", flush=True)
    print("starting_rollout", flush=True)
    runner_log = env_runner.run(policy)
    env_runner.close()
    print("rollout_done", flush=True)

    numeric_log = {}
    for key, value in runner_log.items():
        if isinstance(value, (int, float, bool, np.floating, np.integer)):
            numeric_log[key] = float(value)

    result = {
        "model": str(args.model),
        "oat_checkpoint": str(args.oat_checkpoint),
        "image_key": args.image_key,
        "force_k": args.force_k,
        "n_test": int(env_kwargs["n_test"]),
        "n_parallel_envs": int(env_kwargs["n_parallel_envs"]),
        "max_episode_steps": int(env_kwargs["max_episode_steps"]),
        **numeric_log,
        **policy.stats.to_dict(),
    }
    out_path = ensure_parent(output_dir / "eval_log.json")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
