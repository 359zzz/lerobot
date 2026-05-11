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

import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from lerobot.policies.factory import get_policy_class
from lerobot.policies.vlsarm.configuration_vlsarm import VLSARMConfig
from lerobot.policies.vlsarm.processor_vlsarm import (
    VLSARMSequenceProcessorStep,
    make_vlsarm_pre_post_processors,
)
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.types import TransitionKey


class MockDatasetMeta:
    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    @property
    def episodes(self):
        mock = MagicMock()
        mock.__len__ = lambda s: len(self._episodes)
        mock.__getitem__ = lambda s, idx: self._episodes[idx]
        mock.to_pandas = lambda: pd.DataFrame(self._episodes)
        return mock


@pytest.fixture
def dual_vlsarm_config():
    return VLSARMConfig(
        annotation_mode="dual",
        n_obs_steps=8,
        max_rewind_steps=4,
        frame_gap=30,
        rewind_probability=0.0,
        language_perturbation_probability=0.0,
        sparse_subtask_names=["reach", "grasp", "lift"],
        sparse_temporal_proportions=[0.3, 0.4, 0.3],
        num_sparse_stages=3,
        dense_subtask_names=["approach", "contact", "close_gripper", "lift_up"],
        dense_temporal_proportions=[0.25, 0.25, 0.25, 0.25],
        num_dense_stages=4,
    )


@pytest.fixture
def dataset_meta():
    episodes = [
        {
            "dataset_from_index": 0,
            "dataset_to_index": 300,
            "task": "pick up the cube",
            "sparse_subtask_names": ["reach", "grasp", "lift"],
            "sparse_subtask_start_frames": [0, 90, 210],
            "sparse_subtask_end_frames": [90, 210, 300],
            "dense_subtask_names": ["approach", "contact", "close_gripper", "lift_up"],
            "dense_subtask_start_frames": [0, 75, 150, 225],
            "dense_subtask_end_frames": [75, 150, 225, 300],
        }
    ]
    return MockDatasetMeta(episodes)


def test_vlsarm_processor_outputs_sequence_and_targets(dual_vlsarm_config, dataset_meta):
    processor = VLSARMSequenceProcessorStep(config=dual_vlsarm_config, dataset_meta=dataset_meta)
    processor.train(True)

    num_frames = dual_vlsarm_config.num_frames
    dummy_image = torch.randint(0, 255, (num_frames, 3, 224, 224), dtype=torch.uint8)
    dummy_state = torch.randn(num_frames, 6)

    transition = {
        TransitionKey.OBSERVATION: {
            dual_vlsarm_config.image_key: dummy_image,
            dual_vlsarm_config.state_key: dummy_state,
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "index": 150,
            "episode_index": 0,
            "task": "pick up the cube",
        },
    }

    result = processor(transition)
    obs = result[TransitionKey.OBSERVATION]

    assert obs["frame_images"].shape == (1, num_frames, 3, 224, 224)
    assert obs["state_features"].shape == (1, num_frames, dual_vlsarm_config.max_state_dim)
    assert obs["lengths"].shape == (1,)
    assert obs["frame_delta_indices"].shape == (1, num_frames)
    assert obs["frame_is_rewind"].shape == (1, num_frames)
    assert obs["task_texts"] == ["pick up the cube"]
    assert obs["sparse_targets"].shape == (1, num_frames)
    assert obs["dense_targets"].shape == (1, num_frames)


def test_vlsarm_preprocessor_round_trip(dual_vlsarm_config):
    preprocessor, _ = make_vlsarm_pre_post_processors(config=dual_vlsarm_config, dataset_meta=None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessor.save_pretrained(tmp_dir, config_filename="policy_preprocessor.json")
        loaded = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="policy_preprocessor.json")

    step_names = [step.__class__.__name__ for step in loaded.steps]
    assert "VLSARMSequenceProcessorStep" in step_names


def test_vlsarm_policy_factory_registration():
    policy_cls = get_policy_class("vlsarm")
    assert policy_cls.name == "vlsarm"
