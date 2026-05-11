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

"""VL-SARM processor for sequence assembly and stage/progress target generation."""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from faker import Faker
except ImportError:
    Faker = None

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.sarm.sarm_utils import (
    apply_rewind_augmentation,
    compute_absolute_indices,
    find_stage_and_tau,
    pad_state_to_max_dim,
)
from lerobot.policies.vlsarm.configuration_vlsarm import VLSARMConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    from_tensor_to_numpy,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _make_props_dict(names: list[str] | None, props: list[float] | None) -> dict[str, float] | None:
    return dict(zip(names, props, strict=True)) if names and props else None


@ProcessorStepRegistry.register(name="vlsarm_sequence_processor")
class VLSARMSequenceProcessorStep(ProcessorStep):
    """Prepare raw frame sequences and SARM-style targets for VL-SARM."""

    def __init__(
        self,
        config: VLSARMConfig | dict[str, Any],
        image_key: str | None = None,
        dataset_meta=None,
        dataset_stats: dict | None = None,
    ):
        super().__init__()
        if isinstance(config, dict):
            config = VLSARMConfig(**config)

        self.config = config
        self.image_key = image_key or config.image_key
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats
        self.annotation_mode = config.annotation_mode

        self.sparse_temporal_proportions = _make_props_dict(
            config.sparse_subtask_names, config.sparse_temporal_proportions
        )
        self.sparse_subtask_names = config.sparse_subtask_names
        self.dense_subtask_names = config.dense_subtask_names if config.uses_dual_heads else None
        self.dense_temporal_proportions = (
            _make_props_dict(config.dense_subtask_names, config.dense_temporal_proportions)
            if config.uses_dual_heads
            else None
        )

        self.verbs = ["move", "grasp", "rotate", "push", "pull", "slide", "lift", "place"]
        self.fake = Faker() if Faker is not None else None

    def get_config(self) -> dict[str, Any]:
        return {
            "config": {
                "annotation_mode": self.config.annotation_mode,
                "n_obs_steps": self.config.n_obs_steps,
                "frame_gap": self.config.frame_gap,
                "max_rewind_steps": self.config.max_rewind_steps,
                "max_state_dim": self.config.max_state_dim,
                "rewind_probability": self.config.rewind_probability,
                "language_perturbation_probability": self.config.language_perturbation_probability,
                "num_sparse_stages": self.config.num_sparse_stages,
                "sparse_subtask_names": self.config.sparse_subtask_names,
                "sparse_temporal_proportions": self.config.sparse_temporal_proportions,
                "num_dense_stages": self.config.num_dense_stages,
                "dense_subtask_names": self.config.dense_subtask_names,
                "dense_temporal_proportions": self.config.dense_temporal_proportions,
                "image_key": self.config.image_key,
                "state_key": self.config.state_key,
            },
            "image_key": self.image_key,
        }

    def _find_episode_for_frame(self, frame_idx: int) -> int:
        for ep_idx in range(len(self.dataset_meta.episodes)):
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            if ep_start <= frame_idx < ep_end:
                return ep_idx
        return 0

    def _get_episode_indices(self, frame_indices: np.ndarray, episode_index) -> np.ndarray:
        if episode_index is None:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        episode_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(episode_index)))
        if len(episode_indices) == 1 and len(frame_indices) > 1:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])
        return episode_indices

    def _generate_perturbed_task(self) -> str:
        num_words = random.randint(1, 5)
        verb = random.choice(self.verbs)
        if self.fake is not None:
            words = self.fake.words(nb=num_words)
        else:
            words = [f"token{random.randint(0, 999)}" for _ in range(num_words)]
        phrase = " ".join([verb] + words)
        return phrase

    def _get_annotation_config(self, annotation_type: str) -> tuple[list[str], dict[str, float] | None]:
        if annotation_type == "dense":
            return self.dense_subtask_names, self.dense_temporal_proportions
        return self.sparse_subtask_names, self.sparse_temporal_proportions

    def _load_episode_annotations(
        self,
        ep_idx: int,
        episodes_df: pd.DataFrame | None,
        annotation_type: str,
        global_names: list[str],
    ) -> tuple[list | None, list | None, list | None]:
        if episodes_df is None or len(global_names) == 1:
            return None, None, None

        def col(suffix: str) -> str:
            prefixed = f"{annotation_type}_{suffix}"
            return prefixed if prefixed in episodes_df.columns else suffix

        col_names = col("subtask_names")
        if col_names not in episodes_df.columns or ep_idx >= len(episodes_df):
            return None, None, None

        subtask_names = episodes_df.loc[ep_idx, col_names]
        if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
            return None, None, None

        return (
            subtask_names,
            episodes_df.loc[ep_idx, col("subtask_start_frames")],
            episodes_df.loc[ep_idx, col("subtask_end_frames")],
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        frame_index = comp_data.get("index")
        episode_index = comp_data.get("episode_index")

        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None and self.dataset_meta is not None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")

        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = (
            self._get_episode_indices(frame_indices, episode_index) if self.dataset_meta is not None else None
        )

        image = observation.get(self.image_key)
        if isinstance(image, torch.Tensor):
            image_tensor = image.detach().clone()
        else:
            image_tensor = torch.as_tensor(image)

        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        batch_size = image_tensor.shape[0]
        total_frames = image_tensor.shape[1]
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        n_obs_frames = 1 + n_obs_steps

        rewind_steps = torch.zeros(batch_size, dtype=torch.int32)
        apply_rewind = self.training and random.random() < self.config.rewind_probability

        if apply_rewind and self.dataset_meta is not None and episode_indices is not None:
            for b_idx, (ep_idx, sample_frame_idx) in enumerate(
                zip(episode_indices.tolist(), frame_indices.tolist(), strict=True)
            ):
                ep_idx = int(ep_idx)
                sample_frame_idx = int(sample_frame_idx)
                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                rewind_step, _ = apply_rewind_augmentation(
                    sample_frame_idx,
                    ep_start,
                    n_obs_steps,
                    max_rewind_steps,
                    frame_gap=self.config.frame_gap,
                )
                rewind_steps[b_idx] = rewind_step

        lengths = n_obs_frames + rewind_steps

        for b_idx in range(batch_size):
            valid_len = int(lengths[b_idx].item())
            if valid_len < total_frames:
                image_tensor[b_idx, valid_len:] = 0

        state_key = self.config.state_key
        state_data = observation.get(state_key)
        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float().detach().clone()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        elif state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

        for b_idx in range(batch_size):
            valid_len = int(lengths[b_idx].item())
            if valid_len < state_tensor.shape[1]:
                state_tensor[b_idx, valid_len:] = 0

        state_tensor = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

        task = comp_data.get("task")
        if isinstance(task, list):
            tasks = [str(t) for t in task]
        elif task is None:
            tasks = [""] * batch_size
        else:
            tasks = [str(task)] * batch_size

        apply_perturbation = self.training and random.random() < self.config.language_perturbation_probability
        if apply_perturbation:
            tasks = [self._generate_perturbed_task() for _ in range(batch_size)]

        observation["frame_images"] = image_tensor
        observation["state_features"] = state_tensor
        observation["task_texts"] = tasks
        observation["lengths"] = lengths
        observation["frame_delta_indices"] = torch.tensor(
            self.config.observation_delta_indices, dtype=torch.int32
        ).unsqueeze(0).expand(batch_size, -1)
        observation["frame_is_rewind"] = (
            observation["frame_delta_indices"][:, : total_frames]
            != 0
        ) & (
            torch.arange(total_frames).unsqueeze(0) >= (n_obs_steps + 1)
        )

        if self.dataset_meta is not None and episode_indices is not None:
            episodes_df = self.dataset_meta.episodes.to_pandas()
            if self.sparse_temporal_proportions is not None:
                if apply_perturbation:
                    sparse_targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)
                else:
                    sparse_targets = self._compute_batch_targets(
                        frame_indices, episode_indices, rewind_steps, episodes_df, "sparse"
                    )
                observation["sparse_targets"] = sparse_targets

            if self.config.uses_dual_heads and self.dense_temporal_proportions is not None:
                if apply_perturbation:
                    dense_targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)
                else:
                    dense_targets = self._compute_batch_targets(
                        frame_indices, episode_indices, rewind_steps, episodes_df, "dense"
                    )
                observation["dense_targets"] = dense_targets

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def _compute_batch_targets(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
        rewind_steps: torch.Tensor,
        episodes_df: pd.DataFrame | None,
        annotation_type: str,
    ) -> torch.Tensor:
        batch_size = len(frame_indices)
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        total_frames = 1 + n_obs_steps + max_rewind_steps
        frame_gap = self.config.frame_gap

        global_names, temporal_props = self._get_annotation_config(annotation_type)
        targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)

        for b_idx in range(batch_size):
            ep_idx = int(episode_indices[b_idx])
            frame_idx = int(frame_indices[b_idx])

            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            ep_length = ep_end - ep_start

            subtask_names, subtask_start_frames, subtask_end_frames = self._load_episode_annotations(
                ep_idx, episodes_df, annotation_type, global_names
            )

            obs_indices, _ = compute_absolute_indices(frame_idx, ep_start, ep_end, n_obs_steps, frame_gap=frame_gap)
            obs_indices = obs_indices.tolist()

            for t_idx, abs_idx in enumerate(obs_indices):
                rel_frame = abs_idx - ep_start
                targets[b_idx, t_idx] = find_stage_and_tau(
                    rel_frame,
                    ep_length,
                    subtask_names,
                    subtask_start_frames,
                    subtask_end_frames,
                    global_names,
                    temporal_props,
                    return_combined=True,
                )

            rewind_step = int(rewind_steps[b_idx].item())
            if rewind_step > 0:
                _, rewind_indices = apply_rewind_augmentation(
                    frame_idx,
                    ep_start,
                    n_obs_steps,
                    max_rewind_steps,
                    frame_gap=frame_gap,
                    rewind_step=rewind_step,
                )

                for r_idx, abs_idx in enumerate(rewind_indices[:rewind_step]):
                    rel_frame = max(0, abs_idx - ep_start)
                    targets[b_idx, n_obs_steps + 1 + r_idx] = find_stage_and_tau(
                        rel_frame,
                        ep_length,
                        subtask_names,
                        subtask_start_frames,
                        subtask_end_frames,
                        global_names,
                        temporal_props,
                        return_combined=True,
                    )

        return targets

    @property
    def training(self) -> bool:
        return getattr(self, "_training_mode", True)

    def train(self, mode: bool = True):
        self._training_mode = mode
        return self

    def eval(self):
        return self.train(False)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION]["frame_images"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(self.config.num_frames, 3, 480, 640),
        )
        features[PipelineFeatureType.OBSERVATION]["state_features"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.config.num_frames, self.config.max_state_dim),
        )
        features[PipelineFeatureType.OBSERVATION]["frame_delta_indices"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.config.num_frames,),
        )
        features[PipelineFeatureType.OBSERVATION]["frame_is_rewind"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.config.num_frames,),
        )
        return features


def make_vlsarm_pre_post_processors(
    config: VLSARMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create pre-processor and post-processor pipelines for VL-SARM."""
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[
                AddBatchDimensionProcessorStep(),
                RenameObservationsProcessorStep(rename_map={}),
                NormalizerProcessorStep(
                    features={**config.input_features, **config.output_features},
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                VLSARMSequenceProcessorStep(
                    config=config,
                    dataset_meta=dataset_meta,
                    dataset_stats=dataset_stats,
                ),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
