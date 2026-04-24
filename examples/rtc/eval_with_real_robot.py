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
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies on real robots.

This script demonstrates:
1. Creating a robot and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

For simulation environments, see eval_with_simulation.py

Usage:
    # Run RTC with Real robot with RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot without RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=false \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot with pi0.5 policy
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/pi05_check_rtc \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with bi_openarm_follower (dual-arm OpenArms) and pi0.5 policy
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=lerobot-data-collection/folding_final \
        --robot.type=bi_openarm_follower \
        --robot.cameras='{left_wrist: {type: opencv, index_or_path: "/dev/video4", width: 1280, height: 720, fps: 30}, base: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30}}' \
        --robot.left_arm_config.port=can0 \
        --robot.left_arm_config.side=left \
        --robot.left_arm_config.can_interface=socketcan \
        --robot.left_arm_config.disable_torque_on_disconnect=true \
        --robot.left_arm_config.max_relative_target=8.0 \
        --robot.right_arm_config.port=can1 \
        --robot.right_arm_config.side=right \
        --robot.right_arm_config.can_interface=socketcan \
        --robot.right_arm_config.disable_torque_on_disconnect=true \
        --robot.right_arm_config.max_relative_target=8.0 \
        --task="Fold the T-shirt properly" \
        --fps=30 \
        --duration=2000 \
        --interpolation_multiplier=3 \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --rtc.max_guidance_weight=5.0 \
        --rtc.prefix_attention_schedule=LINEAR \
        --device=cuda
"""

import json
import logging
import math
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
)
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.processor.relative_action_processor import to_relative_actions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Convert diagnostic values to JSON in the background writer thread."""
    if isinstance(value, Tensor):
        tensor = value.detach()
        if tensor.numel() == 1:
            return float(tensor.cpu().item())
        if tensor.numel() <= 16:
            return tensor.cpu().flatten().tolist()
        return {"shape": list(tensor.shape)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class AsyncJsonlLogger:
    """Single background writer so diagnostics do not block control/inference threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.SimpleQueue[dict[str, Any] | None] = queue.SimpleQueue()
        self._closed = False
        self._thread = Thread(target=self._run, daemon=True, name="RTCPerfLogger")
        self._thread.start()

    def log(self, record: dict[str, Any]) -> None:
        if not self._closed:
            self._queue.put(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1) as f:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                payload = {"wall_time_s": time.time(), **record}
                f.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _configure_rtc_metrics(
    policy,
    metrics_logger: AsyncJsonlLogger | None,
    metrics_every_n_steps: int,
    metrics_context: dict[str, Any],
) -> int:
    """Attach RTC metric callbacks to every reachable RTCProcessor instance."""
    if metrics_logger is None:
        return 0

    configured = 0
    seen_objects: set[int] = set()
    seen_processors: set[int] = set()
    stack = [policy]

    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen_objects:
            continue
        seen_objects.add(id(obj))

        rtc_processor = getattr(obj, "rtc_processor", None)
        if rtc_processor is not None and id(rtc_processor) not in seen_processors:
            rtc_processor.metrics_callback = metrics_logger.log
            rtc_processor.metrics_every_n_steps = max(1, metrics_every_n_steps)
            rtc_processor.metrics_context = metrics_context
            seen_processors.add(id(rtc_processor))
            configured += 1

        for attr in ("module", "base_model", "model"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)

    return configured


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)
    interpolation_multiplier: int = 1  # Control rate multiplier (1=off, 2=2x, 3=3x)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 30

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    # Optional diagnostics. When enabled, records are written as JSON lines to this file by a background
    # thread so disk IO does not block action execution or chunk inference.
    rtc_log_path: str | None = None
    rtc_log_denoise_metrics: bool = True
    rtc_log_denoise_every: int = 1
    rtc_log_actor_summary_s: float = 1.0

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def _reanchor_relative_rtc_prefix(
    prev_actions_absolute: Tensor,
    current_state: Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> Tensor:
    """Convert absolute leftovers into model-space for relative-action RTC policies.

    When a policy uses relative actions, the RTC prefix (leftover actions from
    the previous chunk) is stored in absolute space. Before feeding it back to
    the policy we need to re-express it relative to the *current* robot state
    and then re-normalize.
    """
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
    metrics_logger: AsyncJsonlLogger | None = None,
    rtc_metrics_context: dict[str, Any] | None = None,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        # Only keep .pos joints + camera streams if the policy was trained on positions,
        # not the full pos/vel/torque state the robot exposes.
        observation_features_hw = {
            key: value
            for key, value in robot.observation_features().items()
            if key.endswith(".pos") or isinstance(value, tuple)
        }

        dataset_features = hw_to_dataset_features(observation_features_hw, "observation")
        policy_device = policy.config.device

        # Load preprocessor and postprocessor from pretrained files
        # The stats are embedded in the processor .safetensors files
        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if relative_step is not None:
            if relative_step.action_names is None:
                cfg_names = getattr(cfg.policy, "action_feature_names", None)
                if cfg_names:
                    relative_step.action_names = list(cfg_names)
                else:
                    relative_step.action_names = [
                        k for k in robot.robot.action_features if k.endswith(".pos")
                    ]
            logger.info("[GET_ACTIONS] Relative actions enabled: will re-anchor RTC prefix")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        cycle_id = 0
        metrics_log = metrics_logger.log if metrics_logger is not None else None

        while not shutdown_event.is_set():
            queue_snapshot_before = action_queue.snapshot()
            if queue_snapshot_before["qsize"] <= get_actions_threshold:
                cycle_id += 1
                current_time = time.perf_counter()
                action_index_before_inference = queue_snapshot_before["last_index"]
                prev_actions = action_queue.get_left_over()
                prev_leftover_len = 0 if prev_actions is None else prev_actions.shape[0]

                # Use a recent percentile rather than the historical max so a
                # one-off cold-start spike does not permanently overestimate delay.
                inference_latency = latency_tracker.delay_estimate()
                inference_latency_max = latency_tracker.max()
                inference_latency_p95 = latency_tracker.p95()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs_start = time.perf_counter()
                obs = robot.get_observation()
                obs_read_s = time.perf_counter() - obs_start

                obs_prepare_start = time.perf_counter()
                # Apply robot observation processor
                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]  # Task should be a list, not a string!
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )
                obs_prepare_s = time.perf_counter() - obs_prepare_start

                preprocess_start = time.perf_counter()
                preproceseded_obs = preprocessor(obs_with_policy_features)
                preprocess_s = time.perf_counter() - preprocess_start

                # Re-anchor leftover actions for relative-action policies.
                # We need the *postprocessed* (absolute) leftover, not the original
                # (normalized/relative) one that get_left_over() returns.
                reanchor_s = 0.0
                prev_actions_abs_len = None
                if (
                    prev_actions is not None
                    and relative_step is not None
                    and OBS_STATE in obs_with_policy_features
                ):
                    reanchor_start = time.perf_counter()
                    with action_queue.lock:
                        if action_queue.queue is not None:
                            prev_actions_abs = action_queue.queue[action_queue.last_index :].clone()
                        else:
                            prev_actions_abs = None
                    if prev_actions_abs is not None and prev_actions_abs.numel() > 0:
                        prev_actions_abs_len = prev_actions_abs.shape[0]
                        prev_actions = _reanchor_relative_rtc_prefix(
                            prev_actions_absolute=prev_actions_abs,
                            current_state=obs_with_policy_features[OBS_STATE],
                            relative_step=relative_step,
                            normalizer_step=normalizer_step,
                            policy_device=policy_device,
                        )
                    reanchor_s = time.perf_counter() - reanchor_start

                # Generate actions WITH RTC
                if rtc_metrics_context is not None:
                    rtc_metrics_context.clear()
                    rtc_metrics_context.update({"cycle_id": cycle_id})

                policy_start = time.perf_counter()
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )
                policy_s = time.perf_counter() - policy_start

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                postprocess_start = time.perf_counter()
                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)
                postprocess_s = time.perf_counter() - postprocess_start

                action_index_after_inference = action_queue.get_action_index()
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions too small; "
                        "it should be higher than inference delay + execution horizon."
                    )

                merge_start = time.perf_counter()
                merge_stats = action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
                queue_merge_s = time.perf_counter() - merge_start
                total_s = time.perf_counter() - current_time

                if metrics_log is not None:
                    metrics_log(
                        {
                            "event": "get_actions_cycle",
                            "cycle_id": cycle_id,
                            "fps": fps,
                            "time_per_action_s": time_per_chunk,
                            "queue_threshold": get_actions_threshold,
                            "qsize_before_request": queue_snapshot_before["qsize"],
                            "queue_len_before_request": queue_snapshot_before["queue_len"],
                            "queue_generation_before_request": queue_snapshot_before["generation"],
                            "action_index_before_inference": action_index_before_inference,
                            "action_index_after_inference": action_index_after_inference,
                            "indexes_diff_observed": max(
                                0, action_index_after_inference - action_index_before_inference
                            ),
                            "prev_leftover_len": prev_leftover_len,
                            "prev_actions_abs_len": prev_actions_abs_len,
                            "latency_tracker_max_s": inference_latency_max,
                            "latency_tracker_p95_s": inference_latency_p95,
                            "latency_tracker_estimate_s": inference_latency,
                            "inference_delay_used_steps": inference_delay,
                            "real_delay_steps": new_delay,
                            "chunk_size": original_actions.shape[0],
                            "obs_read_s": obs_read_s,
                            "obs_prepare_s": obs_prepare_s,
                            "preprocess_s": preprocess_s,
                            "reanchor_s": reanchor_s,
                            "policy_s": policy_s,
                            "postprocess_s": postprocess_s,
                            "pre_merge_total_s": new_latency,
                            "queue_merge_s": queue_merge_s,
                            "total_s": total_s,
                            **merge_stats,
                        }
                    )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
    metrics_logger: AsyncJsonlLogger | None = None,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_keys = [k for k in robot.action_features() if k.endswith(".pos")]

        action_count = 0
        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        action_interval = interpolator.get_control_interval(cfg.fps)
        metrics_log = metrics_logger.log if metrics_logger is not None else None
        summary_interval_s = max(0.1, cfg.rtc_log_actor_summary_s)
        last_summary_time = time.perf_counter()
        last_summary_count = 0
        last_policy_action: Tensor | None = None
        last_generation: int | None = None
        policy_actions_fetched_interval = 0
        chunk_boundary_count_interval = 0
        policy_action_jump_l2_sum = 0.0
        policy_action_jump_l2_count = 0
        policy_action_jump_l2_max = 0.0
        policy_action_jump_max_abs = 0.0
        chunk_boundary_jump_l2_max = 0.0
        chunk_boundary_jump_max_abs = 0.0

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            if interpolator.needs_new_action():
                queue_snapshot_before_get = action_queue.snapshot()
                new_action = action_queue.get()
                if new_action is not None:
                    new_action_cpu = new_action.cpu()
                    generation = queue_snapshot_before_get["generation"]
                    jump_l2 = 0.0
                    jump_max_abs = 0.0
                    if last_policy_action is not None:
                        jump = (new_action_cpu - last_policy_action).float()
                        jump_l2 = float(jump.norm().item())
                        jump_max_abs = float(jump.abs().max().item())
                        policy_action_jump_l2_sum += jump_l2
                        policy_action_jump_l2_count += 1
                        policy_action_jump_l2_max = max(policy_action_jump_l2_max, jump_l2)
                        policy_action_jump_max_abs = max(policy_action_jump_max_abs, jump_max_abs)
                    if last_generation is not None and generation != last_generation:
                        chunk_boundary_count_interval += 1
                        chunk_boundary_jump_l2_max = max(chunk_boundary_jump_l2_max, jump_l2)
                        chunk_boundary_jump_max_abs = max(chunk_boundary_jump_max_abs, jump_max_abs)
                    last_generation = generation
                    last_policy_action = new_action_cpu.clone()
                    policy_actions_fetched_interval += 1
                    interpolator.add(new_action_cpu)

            action = interpolator.get()
            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(action_keys)}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

            now = time.perf_counter()
            if metrics_log is not None and now - last_summary_time >= summary_interval_s:
                interval_s = now - last_summary_time
                interval_actions = action_count - last_summary_count
                queue_snapshot = action_queue.snapshot()
                metrics_log(
                    {
                        "event": "actor_summary",
                        "actions_sent_total": action_count,
                        "actions_sent_interval": interval_actions,
                        "actions_sent_hz": interval_actions / interval_s if interval_s > 0 else 0.0,
                        "target_policy_fps": cfg.fps,
                        "target_command_hz": cfg.fps * cfg.interpolation_multiplier,
                        "interpolation_multiplier": cfg.interpolation_multiplier,
                        "queue_remaining": queue_snapshot["qsize"],
                        "queue_last_index": queue_snapshot["last_index"],
                        "queue_len": queue_snapshot["queue_len"],
                        "queue_generation": queue_snapshot["generation"],
                        "policy_actions_fetched_interval": policy_actions_fetched_interval,
                        "chunk_boundary_count_interval": chunk_boundary_count_interval,
                        "policy_action_jump_l2_mean": (
                            policy_action_jump_l2_sum / policy_action_jump_l2_count
                            if policy_action_jump_l2_count > 0
                            else 0.0
                        ),
                        "policy_action_jump_l2_max": policy_action_jump_l2_max,
                        "policy_action_jump_max_abs": policy_action_jump_max_abs,
                        "chunk_boundary_jump_l2_max": chunk_boundary_jump_l2_max,
                        "chunk_boundary_jump_max_abs": chunk_boundary_jump_max_abs,
                    }
                )
                last_summary_time = now
                last_summary_count = action_count
                policy_actions_fetched_interval = 0
                chunk_boundary_count_interval = 0
                policy_action_jump_l2_sum = 0.0
                policy_action_jump_l2_count = 0
                policy_action_jump_l2_max = 0.0
                policy_action_jump_max_abs = 0.0
                chunk_boundary_jump_l2_max = 0.0
                chunk_boundary_jump_max_abs = 0.0

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    # PI models handle their own compilation
    if policy.type == "pi05" or policy.type == "pi0":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")
    metrics_logger = AsyncJsonlLogger(cfg.rtc_log_path) if cfg.rtc_log_path else None
    rtc_metrics_context: dict[str, Any] = {}
    if metrics_logger is not None:
        metrics_logger.log(
            {
                "event": "run_start",
                "policy_path": cfg.policy.pretrained_path,
                "fps": cfg.fps,
                "interpolation_multiplier": cfg.interpolation_multiplier,
                "action_queue_size_to_get_new_actions": cfg.action_queue_size_to_get_new_actions,
                "rtc_enabled": cfg.rtc.enabled,
                "rtc_execution_horizon": cfg.rtc.execution_horizon,
                "rtc_max_guidance_weight": cfg.rtc.max_guidance_weight,
                "rtc_prefix_attention_schedule": cfg.rtc.prefix_attention_schedule.name,
            }
        )
        logger.info(f"RTC diagnostics will be written to: {metrics_logger.path}")

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    get_actions_thread = None
    actor_thread = None

    policy_class = get_policy_class(cfg.policy.type)

    # Load config and set compile_model for pi0/pi05 models
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_pretrained_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    # Turn on RTC
    policy.config.rtc_config = cfg.rtc

    # Init RTC processort, as by default if RTC disabled in the config
    # The processor won't be created
    policy.init_rtc_processor()
    if metrics_logger is not None and cfg.rtc_log_denoise_metrics:
        configured = _configure_rtc_metrics(
            policy,
            metrics_logger,
            cfg.rtc_log_denoise_every,
            rtc_metrics_context,
        )
        logger.info(f"Configured RTC denoise diagnostics on {configured} processor(s)")

    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(
            policy,
            robot_wrapper,
            robot_observation_processor,
            action_queue,
            shutdown_event,
            cfg,
            metrics_logger,
            rtc_metrics_context,
        ),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, cfg, metrics_logger),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()
    logger.info("Started actor thread")

    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(10)

        # Log queue status periodically
        if int(time.time() - start_time) % 5 == 0:
            logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

        if time.time() - start_time > cfg.duration:
            break

    logger.info("Demo duration reached or shutdown requested")

    # Signal shutdown
    shutdown_event.set()

    # Wait for threads to finish
    if get_actions_thread and get_actions_thread.is_alive():
        logger.info("Waiting for chunk requester thread to finish...")
        get_actions_thread.join()

    if actor_thread and actor_thread.is_alive():
        logger.info("Waiting for action executor thread to finish...")
        actor_thread.join()

    # Cleanup robot
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    if metrics_logger is not None:
        metrics_logger.log({"event": "run_end", "duration_s": time.time() - start_time})
        metrics_logger.close()

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
