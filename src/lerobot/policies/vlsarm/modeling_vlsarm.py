#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VL-SARM reward model backed by a frame-wise VLM encoder."""

import logging
import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sarm.sarm_utils import normalize_stage_tau, pad_state_to_max_dim
from lerobot.policies.vlsarm.configuration_vlsarm import VLSARMConfig
from lerobot.utils.constants import OBS_STR


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / denom


class TemporalStageProgressHead(nn.Module):
    """Shared temporal trunk with separate stage/progress heads."""

    def __init__(self, config: VLSARMConfig, vl_hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_frames = config.num_frames

        self.state_proj = nn.Sequential(
            nn.LayerNorm(config.max_state_dim),
            nn.Linear(config.max_state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(1, config.state_hidden_dim),
            nn.GELU(),
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
        )
        self.role_embedding = nn.Embedding(2, config.state_hidden_dim)

        fused_dim = vl_hidden_size + 3 * config.state_hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, config.hidden_dim),
            nn.GELU(),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_frames, config.hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        self.trunk_norm = nn.LayerNorm(config.hidden_dim)

        dense_classes = config.num_dense_stages or config.num_sparse_stages
        self.stage_heads = nn.ModuleDict(
            {
                "sparse": nn.Linear(config.hidden_dim, config.num_sparse_stages),
                "dense": nn.Linear(config.hidden_dim, dense_classes),
            }
        )
        self.stage_embeddings = nn.ModuleDict(
            {
                "sparse": nn.Embedding(config.num_sparse_stages, config.stage_condition_dim),
                "dense": nn.Embedding(dense_classes, config.stage_condition_dim),
            }
        )
        self.progress_heads = nn.ModuleDict(
            {
                "sparse": nn.Sequential(
                    nn.LayerNorm(config.hidden_dim + config.stage_condition_dim),
                    nn.Linear(config.hidden_dim + config.stage_condition_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                ),
                "dense": nn.Sequential(
                    nn.LayerNorm(config.hidden_dim + config.stage_condition_dim),
                    nn.Linear(config.hidden_dim + config.stage_condition_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                ),
            }
        )

    def forward_trunk(
        self,
        frame_features: torch.Tensor,
        state_features: torch.Tensor,
        frame_delta_indices: torch.Tensor,
        frame_is_rewind: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        delta = frame_delta_indices.to(frame_features.dtype).unsqueeze(-1)
        delta = delta / max(float(self.config.frame_gap), 1.0)
        delta_emb = self.delta_proj(delta)

        role_ids = frame_is_rewind.long()
        role_emb = self.role_embedding(role_ids)

        state_emb = self.state_proj(state_features)
        x = torch.cat([frame_features, state_emb, delta_emb, role_emb], dim=-1)
        x = self.input_proj(x) + self.pos_embedding[:, : x.shape[1]]

        padding_mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.temporal(x, src_key_padding_mask=padding_mask)
        return self.trunk_norm(h)

    def forward_scheme(
        self,
        hidden_states: torch.Tensor,
        scheme: str,
        stage_distribution: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_logits = self.stage_heads[scheme](hidden_states)
        if stage_distribution is None:
            stage_distribution = torch.softmax(stage_logits, dim=-1)

        stage_emb = stage_distribution @ self.stage_embeddings[scheme].weight
        tau_in = torch.cat([hidden_states, stage_emb], dim=-1)
        tau = torch.sigmoid(self.progress_heads[scheme](tau_in)).squeeze(-1)
        return stage_logits, tau


class VLSARMRewardModel(PreTrainedPolicy):
    """Parallel, pluggable SARM variant using a Qwen-style frame encoder."""

    name = "vlsarm"
    config_class = VLSARMConfig

    def __init__(self, config: VLSARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.dataset_meta = dataset_meta
        self.runtime_device = torch.device(config.device)

        if config.annotation_mode == "single_stage":
            logging.info("VL-SARM initialized in single_stage mode")
        elif dataset_meta is not None:
            self._load_temporal_proportions(dataset_meta)

        vl_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
        }
        torch_dtype = self._get_vl_torch_dtype()
        if torch_dtype is not None:
            vl_kwargs["torch_dtype"] = torch_dtype

        self.vl_model = AutoModelForImageTextToText.from_pretrained(config.vl_model_name, **vl_kwargs)
        self.vl_processor = AutoProcessor.from_pretrained(config.vl_model_name, trust_remote_code=config.trust_remote_code)
        self.vl_image_token_ids = self._infer_vl_image_token_ids()

        self.vl_hidden_size = self._infer_vl_hidden_size()
        self.head = TemporalStageProgressHead(config=config, vl_hidden_size=self.vl_hidden_size)

        if config.freeze_vl_backbone:
            self._freeze_vl_backbone()

        self.to(self.runtime_device)

    def _get_vl_torch_dtype(self) -> torch.dtype | None:
        if self.config.vl_torch_dtype == "auto":
            return None
        if self.config.vl_torch_dtype == "float16":
            return torch.float16
        if self.config.vl_torch_dtype == "float32":
            return torch.float32
        if self.config.vl_torch_dtype == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"Unsupported vl_torch_dtype: {self.config.vl_torch_dtype}")

    def _infer_vl_hidden_size(self) -> int:
        cfg = self.vl_model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return int(cfg.text_config.hidden_size)
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        if hasattr(self.vl_model, "language_model") and hasattr(self.vl_model.language_model.config, "hidden_size"):
            return int(self.vl_model.language_model.config.hidden_size)
        raise ValueError("Unable to infer hidden size for VL backbone")

    def _freeze_vl_backbone(self) -> None:
        self.vl_model.eval()
        for param in self.vl_model.parameters():
            param.requires_grad = False

    def _infer_vl_image_token_ids(self) -> tuple[int, ...]:
        token_ids: list[int] = []
        for obj in (
            getattr(self, "vl_model", None),
            getattr(getattr(self, "vl_model", None), "config", None),
            getattr(self, "vl_processor", None),
            getattr(getattr(self, "vl_processor", None), "tokenizer", None),
        ):
            token_id = getattr(obj, "image_token_id", None)
            if isinstance(token_id, int):
                token_ids.append(token_id)

        tokenizer = getattr(self.vl_processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
            image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if isinstance(image_token_id, int) and image_token_id >= 0:
                token_ids.append(image_token_id)

        unique_ids = tuple(sorted({token_id for token_id in token_ids if token_id >= 0}))
        if not unique_ids:
            logging.warning(
                "Unable to infer VL image token ids for %s; falling back to last-token pooling.",
                self.config.vl_model_name,
            )
        return unique_ids

    def _load_temporal_proportions(self, dataset_meta) -> None:
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        # Reuse the proven annotation loading logic from SARM without changing its code path.
        loader = object.__new__(SARMRewardModel)
        loader.config = self.config
        loader._load_temporal_proportions(dataset_meta)

    def to(self, device):
        super().to(device)
        self.runtime_device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vl_backbone:
            self.vl_model.eval()
        return self

    def _frame_to_pil(self, frame: torch.Tensor | np.ndarray) -> Image.Image:
        if isinstance(frame, torch.Tensor):
            array = frame.detach().cpu()
            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = array.permute(1, 2, 0)
            array = array.numpy()
        else:
            array = np.asarray(frame)
            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))

        if array.ndim != 3:
            raise ValueError(f"Expected frame with 3 dims, got shape {array.shape}")

        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 255.0)
            if array.max() <= 1.0:
                array = array * 255.0
            array = array.astype(np.uint8)

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        return Image.fromarray(array)

    def _build_prompt(self, task_text: str) -> str:
        task_text = task_text.strip() if task_text else "unspecified task"
        return self.config.frame_prompt_template.format(task=task_text)

    def _manual_multimodal_prompt(self, prompt: str) -> str:
        # Qwen3.5 expects one image placeholder token sequence per image.
        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        return f"<|im_start|>user\n{image_placeholder}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _apply_chat_template_if_available(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(self.vl_processor, "apply_chat_template"):
            try:
                return self.vl_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                try:
                    return self.vl_processor.apply_chat_template(messages, tokenize=False)
                except ValueError as exc:
                    if "chat template" not in str(exc).lower():
                        raise
            except ValueError as exc:
                if "chat template" not in str(exc).lower():
                    raise

        tokenizer = getattr(self.vl_processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                try:
                    return tokenizer.apply_chat_template(messages, tokenize=False)
                except ValueError as exc:
                    if "chat template" not in str(exc).lower():
                        raise
            except ValueError as exc:
                if "chat template" not in str(exc).lower():
                    raise

        return self._manual_multimodal_prompt(prompt)

    def _build_text_and_images(
        self, frame_images: torch.Tensor | np.ndarray, task_texts: list[str]
    ) -> tuple[list[str], list[Image.Image]]:
        if isinstance(frame_images, np.ndarray):
            frame_images = torch.from_numpy(frame_images)

        batch_size, seq_len = frame_images.shape[:2]
        texts: list[str] = []
        images: list[Image.Image] = []

        for b_idx in range(batch_size):
            prompt = self._build_prompt(task_texts[b_idx])
            for t_idx in range(seq_len):
                image = self._frame_to_pil(frame_images[b_idx, t_idx])
                text = self._apply_chat_template_if_available(image=image, prompt=prompt)
                texts.append(text)
                images.append(image)

        return texts, images

    def _move_vl_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.runtime_device)
            else:
                moved[key] = value
        return moved

    def _extract_sequence_features(self, outputs) -> torch.Tensor:
        if getattr(outputs, "hidden_states", None) is not None:
            return outputs.hidden_states[-1]
        if getattr(outputs, "last_hidden_state", None) is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, tuple) and outputs:
            return outputs[0]
        raise ValueError("VL model output does not expose hidden states")

    def _pool_last_valid_token(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_indices]

    def _build_image_token_mask(self, model_inputs: dict[str, Any]) -> torch.Tensor | None:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None or not self.vl_image_token_ids:
            return None

        image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.vl_image_token_ids:
            image_mask |= input_ids == token_id

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            image_mask &= attention_mask.to(torch.bool)

        return image_mask

    def _pool_visual_tokens(self, hidden_states: torch.Tensor, model_inputs: dict[str, Any]) -> torch.Tensor:
        image_mask = self._build_image_token_mask(model_inputs)
        attention_mask = model_inputs.get("attention_mask")
        fallback = self._pool_last_valid_token(hidden_states, attention_mask=attention_mask)
        if image_mask is None:
            return fallback

        token_counts = image_mask.sum(dim=1)
        if torch.all(token_counts == 0):
            return fallback

        pooled = fallback.clone()
        has_image_tokens = token_counts > 0
        masked_hidden = hidden_states[has_image_tokens] * image_mask[has_image_tokens].unsqueeze(-1).to(hidden_states.dtype)
        pooled[has_image_tokens] = masked_hidden.sum(dim=1) / token_counts[has_image_tokens].unsqueeze(-1).to(
            hidden_states.dtype
        )
        return pooled

    def _encode_frames(self, frame_images: torch.Tensor | np.ndarray, task_texts: list[str]) -> torch.Tensor:
        if isinstance(frame_images, np.ndarray):
            frame_images = torch.from_numpy(frame_images)

        batch_size, seq_len = frame_images.shape[:2]
        texts, images = self._build_text_and_images(frame_images, task_texts)
        total_frames = len(images)
        encoded_chunks: list[torch.Tensor] = []

        for start in range(0, total_frames, self.config.vl_batch_size):
            stop = min(start + self.config.vl_batch_size, total_frames)
            processor_kwargs = {
                "text": texts[start:stop],
                "images": images[start:stop],
                "padding": True,
                "return_tensors": "pt",
            }
            if self.config.max_prompt_length is not None and self.config.max_prompt_length > 0:
                processor_kwargs["max_length"] = self.config.max_prompt_length
            model_inputs = self.vl_processor(**processor_kwargs)
            model_inputs = self._move_vl_inputs_to_device(dict(model_inputs))

            context_manager = torch.no_grad if self.config.freeze_vl_backbone else nullcontext
            with context_manager():
                outputs = self.vl_model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

            hidden_states = self._extract_sequence_features(outputs)
            pooled = self._pool_visual_tokens(hidden_states, model_inputs)
            encoded_chunks.append(pooled)

        frame_features = torch.cat(encoded_chunks, dim=0)
        frame_features = frame_features.view(batch_size, seq_len, -1)
        return frame_features.to(torch.float32)

    def _prepare_inputs(
        self,
        frame_images: torch.Tensor | np.ndarray,
        task_texts: list[str] | str,
        state_features: torch.Tensor | np.ndarray | None = None,
        lengths: torch.Tensor | np.ndarray | None = None,
        frame_delta_indices: torch.Tensor | np.ndarray | None = None,
        frame_is_rewind: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(frame_images, np.ndarray):
            frame_images = torch.from_numpy(frame_images)
        if frame_images.ndim == 4:
            frame_images = frame_images.unsqueeze(0)

        if isinstance(task_texts, str):
            task_texts = [task_texts]
        task_texts = list(task_texts)

        batch_size, seq_len = frame_images.shape[:2]
        if len(task_texts) == 1 and batch_size > 1:
            task_texts = task_texts * batch_size

        if state_features is None:
            state = torch.zeros(batch_size, seq_len, self.config.max_state_dim, dtype=torch.float32)
        else:
            if isinstance(state_features, np.ndarray):
                state = torch.from_numpy(state_features).float()
            else:
                state = state_features.float()
            if state.ndim == 2:
                state = state.unsqueeze(0)
            state = pad_state_to_max_dim(state, self.config.max_state_dim)

        if lengths is None:
            lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
        else:
            lens = torch.from_numpy(lengths) if isinstance(lengths, np.ndarray) else lengths
            lens = lens.to(torch.int32)

        if frame_delta_indices is None:
            deltas = torch.tensor(self.config.observation_delta_indices, dtype=torch.int32).unsqueeze(0).expand(
                batch_size, -1
            )
        else:
            deltas = torch.from_numpy(frame_delta_indices) if isinstance(frame_delta_indices, np.ndarray) else frame_delta_indices
            if deltas.ndim == 1:
                deltas = deltas.unsqueeze(0).expand(batch_size, -1)
            deltas = deltas.to(torch.int32)

        if frame_is_rewind is None:
            time_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            rewind = time_idx >= (self.config.n_obs_steps + 1)
        else:
            rewind = torch.from_numpy(frame_is_rewind) if isinstance(frame_is_rewind, np.ndarray) else frame_is_rewind
            if rewind.ndim == 1:
                rewind = rewind.unsqueeze(0).expand(batch_size, -1)
            rewind = rewind.to(torch.bool)

        return frame_images, task_texts, state, lens, deltas, rewind

    def _predict_scheme(
        self,
        frame_images: torch.Tensor | np.ndarray,
        task_texts: list[str] | str,
        state_features: torch.Tensor | np.ndarray | None,
        lengths: torch.Tensor | np.ndarray | None,
        frame_delta_indices: torch.Tensor | np.ndarray | None,
        frame_is_rewind: torch.Tensor | np.ndarray | None,
        scheme: str,
        stage_distribution: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            frame_images,
            task_texts,
            state_features,
            lengths,
            frame_delta_indices,
            frame_is_rewind,
        ) = self._prepare_inputs(
            frame_images=frame_images,
            task_texts=task_texts,
            state_features=state_features,
            lengths=lengths,
            frame_delta_indices=frame_delta_indices,
            frame_is_rewind=frame_is_rewind,
        )

        frame_features = self._encode_frames(frame_images, task_texts)
        hidden_states = self.head.forward_trunk(
            frame_features=frame_features.to(self.runtime_device),
            state_features=state_features.to(self.runtime_device),
            frame_delta_indices=frame_delta_indices.to(self.runtime_device),
            frame_is_rewind=frame_is_rewind.to(self.runtime_device),
            lengths=lengths.to(self.runtime_device),
        )
        return self.head.forward_scheme(hidden_states, scheme=scheme, stage_distribution=stage_distribution)

    def _encode_temporal_context(
        self,
        frame_images: torch.Tensor,
        task_texts: list[str],
        state_features: torch.Tensor,
        lengths: torch.Tensor,
        frame_delta_indices: torch.Tensor,
        frame_is_rewind: torch.Tensor,
    ) -> torch.Tensor:
        frame_features = self._encode_frames(frame_images, task_texts)
        state_features = pad_state_to_max_dim(state_features.float(), self.config.max_state_dim)
        return self.head.forward_trunk(
            frame_features=frame_features.to(self.runtime_device),
            state_features=state_features.to(self.runtime_device),
            frame_delta_indices=frame_delta_indices.to(self.runtime_device),
            frame_is_rewind=frame_is_rewind.to(self.runtime_device),
            lengths=lengths.to(self.runtime_device),
        )

    def _train_scheme(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        scheme: str,
        reduction: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages
        if num_classes is None:
            raise ValueError(f"num_classes for scheme '{scheme}' is None")

        gt_stage = torch.floor(targets).long().clamp(0, num_classes - 1)
        gt_tau = torch.remainder(targets, 1.0)
        valid_mask = (torch.arange(targets.shape[1], device=targets.device).unsqueeze(0) < lengths.unsqueeze(1)).to(
            targets.dtype
        )

        stage_distribution = None
        if random.random() < self.config.gt_stage_ratio:
            stage_distribution = F.one_hot(gt_stage, num_classes=num_classes).float().to(self.runtime_device)

        stage_logits, tau_pred = self.head.forward_scheme(
            hidden_states,
            scheme=scheme,
            stage_distribution=stage_distribution,
        )

        ce = F.cross_entropy(
            stage_logits.transpose(1, 2),
            gt_stage.to(self.runtime_device),
            reduction="none",
        )
        mse = (tau_pred - gt_tau.to(self.runtime_device)) ** 2

        sample_stage_loss = _masked_mean(ce, valid_mask.to(self.runtime_device))
        sample_progress_loss = _masked_mean(mse, valid_mask.to(self.runtime_device))
        sample_total_loss = (
            self.config.stage_loss_weight * sample_stage_loss
            + self.config.progress_loss_weight * sample_progress_loss
        )

        if reduction == "none":
            total_loss = sample_total_loss
        else:
            total_loss = sample_total_loss.mean()

        metrics = {
            "stage_loss": sample_stage_loss.mean().item(),
            "progress_loss": sample_progress_loss.mean().item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def calculate_rewards(
        self,
        task_texts: list[str] | str,
        frame_images: np.ndarray | torch.Tensor,
        state_features: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
        frame_delta_indices: np.ndarray | torch.Tensor | None = None,
        frame_is_rewind: np.ndarray | torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_stages: bool = False,
        return_confidence: bool = False,
        head_mode: str = "sparse",
        frame_index: int | None = None,
    ) -> np.ndarray | tuple:
        stage_logits, tau_pred = self._predict_scheme(
            frame_images=frame_images,
            task_texts=task_texts,
            state_features=state_features,
            lengths=lengths,
            frame_delta_indices=frame_delta_indices,
            frame_is_rewind=frame_is_rewind,
            scheme=head_mode,
        )

        stage_probs = F.softmax(stage_logits, dim=-1)
        stage_idx = stage_probs.argmax(dim=-1)
        stage_conf = stage_probs.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)
        raw_reward = stage_idx.float() + tau_pred

        if head_mode == "sparse":
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=self.config.num_sparse_stages,
                temporal_proportions=self.config.sparse_temporal_proportions,
                subtask_names=self.config.sparse_subtask_names,
            )
        else:
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=self.config.num_dense_stages,
                temporal_proportions=self.config.dense_temporal_proportions,
                subtask_names=self.config.dense_subtask_names,
            )

        if frame_index is None:
            frame_index = self.config.current_frame_index

        rewards = normalized_reward.cpu().numpy() if return_all_frames else normalized_reward[:, frame_index].cpu().numpy()
        outputs: list[Any] = [rewards]

        if return_stages:
            outputs.append(stage_probs.cpu().numpy())
        if return_confidence:
            outputs.append(stage_conf.cpu().numpy())

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("VL-SARM is a reward model and does not predict actions")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("VL-SARM is a reward model and does not select actions")

    def forward(self, batch, reduction: str = "mean"):
        observation = batch.get(OBS_STR, batch)

        frame_images = observation["frame_images"]
        task_texts = observation["task_texts"]
        state_features = observation.get("state_features")
        lengths = observation.get("lengths")
        frame_delta_indices = observation.get("frame_delta_indices")
        frame_is_rewind = observation.get("frame_is_rewind")

        if state_features is None:
            raise ValueError("state_features is required for VL-SARM training")
        if lengths is None:
            raise ValueError("lengths is required for VL-SARM training")
        if frame_delta_indices is None:
            raise ValueError("frame_delta_indices is required for VL-SARM training")
        if frame_is_rewind is None:
            raise ValueError("frame_is_rewind is required for VL-SARM training")

        total_loss: torch.Tensor | None = None
        output_dict: dict[str, float] = {}

        sparse_targets = observation.get("sparse_targets")
        if sparse_targets is None:
            sparse_targets = observation.get("targets")
        if sparse_targets is None:
            raise ValueError("sparse_targets (or targets) is required for VL-SARM training")

        hidden_states = self._encode_temporal_context(
            frame_images=frame_images,
            task_texts=task_texts,
            state_features=state_features,
            lengths=lengths,
            frame_delta_indices=frame_delta_indices,
            frame_is_rewind=frame_is_rewind,
        )

        sparse_loss, sparse_metrics = self._train_scheme(
            hidden_states=hidden_states,
            lengths=lengths,
            targets=sparse_targets.to(torch.float32),
            scheme="sparse",
            reduction=reduction,
        )
        total_loss = sparse_loss
        output_dict["sparse_stage_loss"] = sparse_metrics["stage_loss"]
        output_dict["sparse_progress_loss"] = sparse_metrics["progress_loss"]

        if self.config.uses_dual_heads:
            dense_targets = observation.get("dense_targets")
            if dense_targets is not None:
                dense_loss, dense_metrics = self._train_scheme(
                    hidden_states=hidden_states,
                    lengths=lengths,
                    targets=dense_targets.to(torch.float32),
                    scheme="dense",
                    reduction=reduction,
                )
                total_loss = total_loss + dense_loss if total_loss is not None else dense_loss
                output_dict["dense_stage_loss"] = dense_metrics["stage_loss"]
                output_dict["dense_progress_loss"] = dense_metrics["progress_loss"]

        if total_loss is None:
            raise ValueError("VL-SARM total_loss should never be None")

        if reduction == "none":
            output_dict["total_loss"] = total_loss.mean().item()
        else:
            output_dict["total_loss"] = total_loss.item()
        return total_loss, output_dict
