"""Dataset and collator utilities for adaptive OAT-VLA stage-3 SFT."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset

from adaptive_oat_vla.common_vla import load_annotation_payload, maybe_sample_larger_k


DEFAULT_IMAGE_KEYS = ["agentview_rgb", "robot0_eye_in_hand_rgb"]
DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "task_uid"]


def _format_state(state_dict: dict[str, Any]) -> str:
    parts = []
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            flat = value.flatten().tolist()
        else:
            flat = torch.as_tensor(value).flatten().tolist()
        parts.append(f"{key}=" + ",".join(f"{float(v):.4f}" for v in flat))
    return " | ".join(parts)


class AdaptiveVLADataset(Dataset):
    """Read lightweight annotations and fetch observations lazily from zarr."""

    def __init__(
        self,
        annotation_path: str | Path,
        action_vocab_map_path: str | Path,
        random_larger_k_prob: float = 0.0,
        seed: int = 42,
        image_key: str = "agentview_rgb",
    ):
        super().__init__()
        self.annotations = load_annotation_payload(annotation_path)
        self.action_vocab_map = load_annotation_payload(action_vocab_map_path)
        self.root = zarr.open(self.annotations["zarr_path"], mode="r")
        self.data = self.root["data"]
        self.candidate_ks = [int(k) for k in self.annotations["candidate_ks"]]
        self.random_larger_k_prob = float(random_larger_k_prob)
        self.rng = random.Random(seed)
        self.max_action_tokens = int(self.annotations["full_oat_token_ids"].shape[1])
        self.image_key = str(image_key)
        if self.image_key not in DEFAULT_IMAGE_KEYS:
            raise ValueError(f"Unsupported image_key {self.image_key}")

        self.action_token_ids = self.action_vocab_map["action_token_ids"]
        self.eos_token_id = int(
            self.action_vocab_map.get("action_eos_token_id", self.action_vocab_map["eos_token_id"])
        )

    def __len__(self) -> int:
        return int(len(self.annotations["prompts"]))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        obs_indices = self.annotations["obs_indices"][idx].tolist()
        prompt = self.annotations["prompts"][idx]
        optimal_k = int(self.annotations["optimal_ks"][idx])
        chosen_k = maybe_sample_larger_k(
            optimal_k=optimal_k,
            candidate_ks=self.candidate_ks,
            random_larger_k_prob=self.random_larger_k_prob,
            rng=self.rng,
        )

        full_oat_ids = self.annotations["full_oat_token_ids"][idx].tolist()
        chosen_oat_ids = full_oat_ids[:chosen_k]
        action_ids = [self.action_token_ids[oat_id] for oat_id in chosen_oat_ids]
        action_ids.append(self.eos_token_id)

        images = {key: torch.as_tensor(self.data[key][obs_indices]) for key in DEFAULT_IMAGE_KEYS}
        states = {
            key: torch.as_tensor(self.data[key][obs_indices[-1]])
            for key in DEFAULT_STATE_KEYS
        }

        return {
            "prompt": prompt,
            "images": images,
            "primary_image": images[self.image_key][-1],
            "states": states,
            "state_text": _format_state(states),
            "action_token_ids": torch.tensor(action_ids, dtype=torch.long),
            "optimal_k": optimal_k,
            "chosen_k": chosen_k,
            "obs_indices": torch.tensor(obs_indices, dtype=torch.long),
        }


@dataclass
class AdaptiveVLACollator:
    """Build supervised LM targets for action-token generation.

    This collator is processor-agnostic: if the provided processor accepts images, it
    gets both images and text; otherwise it falls back to text-only tokenization.
    """

    processor: Any
    tokenizer: Any
    image_prompt_template: str = (
        "In: What action should the robot take to {prompt}?\n"
        "State: {state_text}\n"
        "Out:"
    )

    def _prepare_text(self, sample: dict[str, Any]) -> str:
        return self.image_prompt_template.format(
            prompt=sample["prompt"],
            state_text=sample["state_text"],
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [self._prepare_text(sample) for sample in batch]
        rgb_images = [Image.fromarray(sample["primary_image"].numpy()).convert("RGB") for sample in batch]

        processor_inputs = None
        if self.processor is not None:
            processor_inputs = self.processor(
                text=texts,
                images=rgb_images,
                return_tensors="pt",
                padding=True,
            )
        else:
            processor_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
            )

        input_ids = processor_inputs["input_ids"]
        attention_mask = processor_inputs["attention_mask"]

        action_lengths = [len(sample["action_token_ids"]) for sample in batch]
        max_action_len = max(action_lengths)
        labels = torch.full(
            (len(batch), input_ids.shape[1] + max_action_len),
            fill_value=-100,
            dtype=torch.long,
        )
        final_input_ids = torch.full_like(labels, fill_value=self.tokenizer.pad_token_id)
        final_attention_mask = torch.zeros_like(labels, dtype=torch.long)
        action_mask = torch.zeros_like(labels, dtype=torch.bool)

        final_input_ids[:, : input_ids.shape[1]] = input_ids
        final_attention_mask[:, : input_ids.shape[1]] = attention_mask

        for row, sample in enumerate(batch):
            prefix_len = int(attention_mask[row].sum().item())
            action_ids = sample["action_token_ids"]
            final_input_ids[row, :prefix_len] = input_ids[row, :prefix_len]
            final_input_ids[row, prefix_len : prefix_len + len(action_ids)] = action_ids
            final_attention_mask[row, prefix_len : prefix_len + len(action_ids)] = 1
            labels[row, prefix_len : prefix_len + len(action_ids)] = action_ids
            action_mask[row, prefix_len : prefix_len + len(action_ids)] = True

        result = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "labels": labels,
            "action_mask": action_mask,
            "prompts": [sample["prompt"] for sample in batch],
            "chosen_k": torch.tensor([sample["chosen_k"] for sample in batch], dtype=torch.long),
            "optimal_k": torch.tensor([sample["optimal_k"] for sample in batch], dtype=torch.long),
        }
        # Keep vision inputs from the processor so OpenVLA sees real image features during SFT.
        for key, value in processor_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            result[key] = value
        return result
