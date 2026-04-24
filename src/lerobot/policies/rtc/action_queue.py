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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.generation = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def snapshot(self) -> dict[str, int | None]:
        """Return a thread-safe snapshot of queue state for diagnostics."""
        with self.lock:
            queue_len = None if self.queue is None else len(self.queue)
            original_queue_len = None if self.original_queue is None else len(self.original_queue)
            qsize = 0 if queue_len is None else queue_len - self.last_index
            return {
                "qsize": qsize,
                "last_index": self.last_index,
                "queue_len": queue_len,
                "original_queue_len": original_queue_len,
                "generation": self.generation,
            }

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """Get leftover processed actions (the actions currently executed by the robot).

        Returns:
            Tensor | None: Remaining processed actions (remaining_steps, action_dim),
                or None if no processed queue exists.
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> dict[str, int | float | None]:
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            action_index_after_inference = self.last_index
            indexes_diff = (
                max(0, action_index_after_inference - action_index_before_inference)
                if action_index_before_inference is not None
                else 0
            )
            old_head_action = None
            old_qsize = 0
            if self.queue is not None and self.last_index < len(self.queue):
                old_head_action = self.queue[self.last_index].clone()
                old_qsize = len(self.queue) - self.last_index
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
            raw_new_head_action = None
            boundary_blend_steps = 0

            if self.cfg.enabled:
                (
                    clamped_delay,
                    raw_new_head_action,
                    boundary_blend_steps,
                ) = self._replace_actions_queue(original_actions, processed_actions, delay)
            else:
                clamped_delay = 0
                self._append_actions_queue(original_actions, processed_actions)

            queue_len = 0 if self.queue is None else len(self.queue)
            original_queue_len = 0 if self.original_queue is None else len(self.original_queue)
            new_head_action = None if self.queue is None or len(self.queue) == 0 else self.queue[0].clone()
            replacement_jump_raw_l2 = None
            replacement_jump_raw_max_abs = None
            replacement_jump_l2 = None
            replacement_jump_max_abs = None
            if old_head_action is not None and raw_new_head_action is not None:
                raw_jump = (raw_new_head_action - old_head_action).float()
                replacement_jump_raw_l2 = float(raw_jump.norm().item())
                replacement_jump_raw_max_abs = float(raw_jump.abs().max().item())
            if old_head_action is not None and new_head_action is not None:
                jump = (new_head_action - old_head_action).float()
                replacement_jump_l2 = float(jump.norm().item())
                replacement_jump_max_abs = float(jump.abs().max().item())
            return {
                "action_index_after_inference": action_index_after_inference,
                "indexes_diff": indexes_diff,
                "real_delay": real_delay,
                "resolved_delay": delay,
                "clamped_delay": clamped_delay,
                "queue_generation": self.generation,
                "qsize_after_merge": queue_len - self.last_index,
                "queue_len_after_merge": queue_len,
                "original_queue_len_after_merge": original_queue_len,
                "old_qsize_before_merge": old_qsize,
                "boundary_blend_steps": boundary_blend_steps,
                "replacement_had_old_head": int(old_head_action is not None),
                "replacement_had_new_head": int(new_head_action is not None),
                "replacement_jump_raw_l2": replacement_jump_raw_l2,
                "replacement_jump_raw_max_abs": replacement_jump_raw_max_abs,
                "replacement_jump_l2": replacement_jump_l2,
                "replacement_jump_max_abs": replacement_jump_max_abs,
            }

    def _replace_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
    ) -> tuple[int, Tensor | None, int]:
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        raw_new_head_action = None
        if clamped_delay < len(processed_actions):
            raw_new_head_action = processed_actions[clamped_delay].clone()

        new_original_queue = original_actions[clamped_delay:].clone()
        new_processed_queue = processed_actions[clamped_delay:].clone()
        boundary_blend_steps = self._compute_boundary_blend_steps(len(new_processed_queue))
        if boundary_blend_steps > 0:
            new_original_queue[:boundary_blend_steps] = self._blend_queue_prefix(
                old_prefix=self.original_queue[self.last_index : self.last_index + boundary_blend_steps],
                new_prefix=new_original_queue[:boundary_blend_steps],
            )
            new_processed_queue[:boundary_blend_steps] = self._blend_queue_prefix(
                old_prefix=self.queue[self.last_index : self.last_index + boundary_blend_steps],
                new_prefix=new_processed_queue[:boundary_blend_steps],
            )

        self.original_queue = new_original_queue
        self.queue = new_processed_queue

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")
        logger.debug(f"boundary_blend_steps: {boundary_blend_steps}")

        self.generation += 1
        self.last_index = 0
        return clamped_delay, raw_new_head_action, boundary_blend_steps

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.generation += 1
        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.

        Returns:
            int: Delay to use.
        """
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay

    def _compute_boundary_blend_steps(self, new_queue_len: int) -> int:
        """Return how many initial RTC steps should be blended for continuity."""
        if self.queue is None or self.last_index >= len(self.queue) or new_queue_len <= 0:
            return 0
        horizon = max(0, int(self.cfg.execution_horizon))
        if horizon <= 0:
            return 0
        old_remaining = len(self.queue) - self.last_index
        return min(old_remaining, new_queue_len, horizon)

    @staticmethod
    def _blend_queue_prefix(old_prefix: Tensor, new_prefix: Tensor) -> Tensor:
        """Blend old and new chunk prefixes so the boundary changes gradually."""
        if len(old_prefix) == 0 or len(new_prefix) == 0:
            return new_prefix
        if len(old_prefix) != len(new_prefix):
            raise ValueError("old_prefix and new_prefix must have matching lengths")

        if len(new_prefix) == 1:
            alpha = torch.zeros(1, device=new_prefix.device, dtype=new_prefix.dtype)
        else:
            alpha = torch.linspace(0, 1, steps=len(new_prefix), device=new_prefix.device, dtype=new_prefix.dtype)
        alpha = alpha.view(-1, *([1] * (new_prefix.ndim - 1)))
        return torch.lerp(
            old_prefix.to(device=new_prefix.device, dtype=new_prefix.dtype), new_prefix, alpha
        )
