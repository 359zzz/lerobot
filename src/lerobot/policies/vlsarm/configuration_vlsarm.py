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

"""
VL-SARM: a parallel, pluggable SARM variant backed by a vision-language model.

This keeps the SARM supervision contract (`stage` + `within-stage progress`)
while replacing the legacy CLIP + dual-transformer stack with:
1. a frame-wise VL encoder
2. a shared temporal trunk
3. lightweight stage/progress heads
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("vlsarm")
@dataclass
class VLSARMConfig(PreTrainedConfig):
    annotation_mode: str = "single_stage"  # "single_stage", "dense_only", or "dual"
    n_obs_steps: int = 8
    frame_gap: int = 30
    max_rewind_steps: int = 4

    # VL backbone
    vl_model_name: str = "Qwen/Qwen3.5-4B-Base"
    trust_remote_code: bool = True
    freeze_vl_backbone: bool = True
    vl_torch_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32", or "auto"
    vl_batch_size: int = 8
    max_prompt_length: int = 256
    frame_prompt_template: str = (
        "Task: {task}\n"
        "Encode this robot observation for stage and progress estimation."
    )

    # Temporal head
    hidden_dim: int = 768
    state_hidden_dim: int = 256
    stage_condition_dim: int = 128
    num_heads: int = 12
    num_layers: int = 4
    dropout: float = 0.1
    max_state_dim: int = 32
    batch_size: int = 32
    stage_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    gt_stage_ratio: float = 0.75

    rewind_probability: float = 0.8
    language_perturbation_probability: float = 0.2

    # Sparse annotations
    num_sparse_stages: int = 1
    sparse_subtask_names: list[str] | None = None
    sparse_temporal_proportions: list[float] | None = None

    # Dense annotations
    num_dense_stages: int | None = None
    dense_subtask_names: list[str] | None = None
    dense_temporal_proportions: list[float] | None = None

    pretrained_model_path: str | None = None
    image_key: str = OBS_IMAGES + ".top"
    state_key: str = OBS_STATE

    input_features: dict = field(default_factory=lambda: {})
    output_features: dict = field(default_factory=lambda: {})

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.annotation_mode not in ["single_stage", "dense_only", "dual"]:
            raise ValueError(
                f"annotation_mode must be 'single_stage', 'dense_only', or 'dual', got {self.annotation_mode}"
            )

        if self.annotation_mode == "single_stage":
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]
            self.num_dense_stages = None
            self.dense_subtask_names = None
            self.dense_temporal_proportions = None
        elif self.annotation_mode == "dense_only":
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]

        self.input_features = {}
        self.output_features = {}

        if self.image_key:
            self.input_features[self.image_key] = PolicyFeature(shape=(480, 640, 3), type=FeatureType.VISUAL)

        self.input_features[self.state_key] = PolicyFeature(
            shape=(self.max_state_dim,),
            type=FeatureType.STATE,
        )

        if self.annotation_mode in ["dense_only", "dual"]:
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
            dense_stages = self.num_dense_stages or self.num_sparse_stages
            self.output_features["dense_stage"] = PolicyFeature(
                shape=(self.num_frames, dense_stages), type=FeatureType.REWARD
            )
            self.output_features["dense_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
        else:
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )

        if self.max_rewind_steps >= self.n_obs_steps:
            raise ValueError(
                f"max_rewind_steps ({self.max_rewind_steps}) must be less than n_obs_steps ({self.n_obs_steps})"
            )
        if self.num_sparse_stages < 1:
            raise ValueError(f"num_sparse_stages must be at least 1, got {self.num_sparse_stages}")
        if (
            self.annotation_mode in ["dense_only", "dual"]
            and self.num_dense_stages is not None
            and self.num_dense_stages < 2
        ):
            raise ValueError(f"num_dense_stages must be at least 2, got {self.num_dense_stages}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=2e-5 if self.freeze_vl_backbone else 1e-5,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=2e-5 if self.freeze_vl_backbone else 1e-5,
            decay_lr=2e-6 if self.freeze_vl_backbone else 1e-6,
            num_warmup_steps=500,
            num_decay_steps=50000,
        )

    def validate_features(self) -> None:
        pass

    @property
    def uses_dual_heads(self) -> bool:
        return self.annotation_mode in ["dense_only", "dual"]

    @property
    def num_frames(self) -> int:
        return 1 + self.n_obs_steps + self.max_rewind_steps

    @property
    def max_length(self) -> int:
        return self.num_frames

    @property
    def current_frame_index(self) -> int:
        return self.n_obs_steps // 2

    @property
    def observation_delta_indices(self) -> list[int]:
        half_steps = self.n_obs_steps // 2
        past_deltas = [-self.frame_gap * i for i in range(half_steps, 0, -1)]
        future_deltas = [self.frame_gap * i for i in range(1, half_steps + 1)]
        obs_deltas = past_deltas + [0] + future_deltas
        rewind_deltas = [-self.frame_gap * (i + 1) for i in range(self.max_rewind_steps)]
        return obs_deltas + rewind_deltas

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
