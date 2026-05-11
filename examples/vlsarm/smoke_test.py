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

"""Minimal smoke test for the Qwen-backed VL-SARM scaffold."""

import argparse
import time

import torch

from lerobot.policies.vlsarm.configuration_vlsarm import VLSARMConfig
from lerobot.policies.vlsarm.modeling_vlsarm import VLSARMRewardModel


def build_dummy_batch(config: VLSARMConfig, height: int, width: int) -> dict:
    batch_size = 1
    seq_len = config.num_frames

    frame_images = torch.randint(
        low=0,
        high=256,
        size=(batch_size, seq_len, 3, height, width),
        dtype=torch.uint8,
    )
    state_features = torch.randn(batch_size, seq_len, min(8, config.max_state_dim), dtype=torch.float32)
    lengths = torch.tensor([seq_len], dtype=torch.int32)
    frame_delta_indices = torch.tensor(config.observation_delta_indices, dtype=torch.int32).unsqueeze(0)
    frame_is_rewind = (
        torch.arange(seq_len, dtype=torch.int32).unsqueeze(0) >= (config.n_obs_steps + 1)
    )
    sparse_targets = torch.linspace(0.0, 0.999, steps=seq_len, dtype=torch.float32).unsqueeze(0)

    return {
        "observation": {
            "frame_images": frame_images,
            "task_texts": ["fold the cloth neatly"],
            "state_features": state_features,
            "lengths": lengths,
            "frame_delta_indices": frame_delta_indices,
            "frame_is_rewind": frame_is_rewind,
            "sparse_targets": sparse_targets,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run a minimal VL-SARM smoke test.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--trainable-vl", action="store_true")
    args = parser.parse_args()

    config = VLSARMConfig(
        annotation_mode="single_stage",
        vl_model_name=args.model_name,
        device=args.device,
        vl_torch_dtype=args.dtype,
        freeze_vl_backbone=not args.trainable_vl,
    )

    print("Building VL-SARM model...")
    t0 = time.time()
    model = VLSARMRewardModel(config)
    t1 = time.time()
    print(f"Model ready in {t1 - t0:.2f}s on {config.device}")

    batch = build_dummy_batch(config=config, height=args.height, width=args.width)

    print("Running forward()...")
    t2 = time.time()
    loss, metrics = model.forward(batch)
    t3 = time.time()
    print(f"loss={float(loss):.6f}")
    print(f"metrics={metrics}")
    print(f"forward_time={t3 - t2:.2f}s")

    print("Running calculate_rewards()...")
    t4 = time.time()
    rewards = model.calculate_rewards(
        task_texts=batch["observation"]["task_texts"],
        frame_images=batch["observation"]["frame_images"],
        state_features=batch["observation"]["state_features"],
        lengths=batch["observation"]["lengths"],
        frame_delta_indices=batch["observation"]["frame_delta_indices"],
        frame_is_rewind=batch["observation"]["frame_is_rewind"],
        return_all_frames=True,
    )
    t5 = time.time()
    print(f"reward_shape={tuple(rewards.shape)}")
    print(f"reward_time={t5 - t4:.2f}s")

    if torch.cuda.is_available() and "cuda" in str(config.device):
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"cuda_peak_mem_gb={peak_mem_gb:.2f}")


if __name__ == "__main__":
    main()
