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

import logging
import math
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread

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


@dataclass(frozen=True)
class ThreadFailure:
    thread_name: str
    error: Exception
    traceback_str: str


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
    preprocessor,
    postprocessor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    failure_queue: Queue[ThreadFailure],
    cfg: RTCDemoConfig,
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
        else:
            chunk_size = getattr(policy.config, "chunk_size", None)
            if chunk_size is not None and get_actions_threshold >= chunk_size:
                clamped_threshold = max(0, chunk_size - 1)
                logger.warning(
                    "[GET_ACTIONS] action_queue_size_to_get_new_actions=%d >= chunk_size=%d; "
                    "clamping to %d to avoid continuous inference.",
                    get_actions_threshold,
                    chunk_size,
                    clamped_threshold,
                )
                get_actions_threshold = clamped_threshold

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs = robot.get_observation()

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

                preproceseded_obs = preprocessor(obs_with_policy_features)

                # Re-anchor leftover actions for relative-action policies.
                # We need the *postprocessed* (absolute) leftover, not the original
                # (normalized/relative) one that get_left_over() returns.
                if (
                    prev_actions is not None
                    and relative_step is not None
                    and OBS_STATE in obs_with_policy_features
                ):
                    with action_queue.lock:
                        if action_queue.queue is not None:
                            prev_actions_abs = action_queue.queue[action_queue.last_index :].clone()
                        else:
                            prev_actions_abs = None
                    if prev_actions_abs is not None and prev_actions_abs.numel() > 0:
                        prev_actions = _reanchor_relative_rtc_prefix(
                            prev_actions_absolute=prev_actions_abs,
                            current_state=obs_with_policy_features[OBS_STATE],
                            relative_step=relative_step,
                            normalizer_step=normalizer_step,
                            policy_device=policy_device,
                        )

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                applied_delay = action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
                queue_size_after_merge = action_queue.qsize()
                logger.info(
                    "[GET_ACTIONS] Inference latency=%.2fs, latency_delay=%d, applied_delay=%d, queue=%d",
                    new_latency,
                    new_delay,
                    applied_delay,
                    queue_size_after_merge,
                )
                if queue_size_after_merge == 0:
                    logger.warning(
                        "[GET_ACTIONS] New chunk has no executable actions after delay compensation. "
                        "Inference is slower than the available queued horizon."
                    )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(tb)
        failure_queue.put(ThreadFailure(thread_name="GET_ACTIONS", error=e, traceback_str=tb))
        shutdown_event.set()


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    failure_queue: Queue[ThreadFailure],
    cfg: RTCDemoConfig,
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

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            if interpolator.needs_new_action():
                new_action = action_queue.get()
                if new_action is not None:
                    interpolator.add(new_action.cpu())

            action = interpolator.get()
            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(action_keys)}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(tb)
        failure_queue.put(ThreadFailure(thread_name="ACTOR", error=e, traceback_str=tb))
        shutdown_event.set()


def _wait_for_initial_action_chunk(
    action_queue: ActionQueue,
    shutdown_event: Event,
    failure_queue: Queue[ThreadFailure],
    log_interval_s: float = 5.0,
) -> ThreadFailure | None:
    """Wait until the first policy chunk is available before starting control."""
    logger.info("[MAIN] Waiting for initial action chunk before starting actor thread")
    wait_start = time.monotonic()
    last_log_time = wait_start

    while not shutdown_event.is_set():
        try:
            return failure_queue.get_nowait()
        except Empty:
            pass

        queue_size = action_queue.qsize()
        if queue_size > 0:
            logger.info(
                "[MAIN] Initial action chunk ready after %.2fs (queue=%d)",
                time.monotonic() - wait_start,
                queue_size,
            )
            return None

        now = time.monotonic()
        if now - last_log_time >= log_interval_s:
            logger.info(
                "[MAIN] Still waiting for initial action chunk... elapsed=%.1fs, queue=%d",
                now - wait_start,
                queue_size,
            )
            last_log_time = now

        time.sleep(0.25)

    return None


def _build_runtime_policy_config(cfg: RTCDemoConfig) -> PreTrainedConfig:
    if cfg.policy is None:
        raise ValueError("Policy configuration is required")

    policy_cfg = deepcopy(cfg.policy)

    if cfg.device is not None:
        policy_cfg.device = cfg.device

    if policy_cfg.type in {"pi05", "pi0"}:
        policy_cfg.compile_model = cfg.use_torch_compile

    return policy_cfg


def _load_policy_processors(
    policy_cfg: PreTrainedConfig,
):
    if policy_cfg.pretrained_path is None:
        raise ValueError("Policy pretrained_path is required to load processors")

    logger.info(f"Loading preprocessor/postprocessor from {policy_cfg.pretrained_path}")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
        },
    )
    logger.info("Preprocessor/postprocessor loaded successfully with embedded stats")
    return preprocessor, postprocessor


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

    runtime_policy_cfg = _build_runtime_policy_config(cfg)

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {runtime_policy_cfg.device}")
    logger.info(
        "Policy runtime config: type=%s, chunk_size=%s, n_action_steps=%s, num_inference_steps=%s, "
        "gradient_checkpointing=%s, dtype=%s, use_relative_actions=%s",
        runtime_policy_cfg.type,
        getattr(runtime_policy_cfg, "chunk_size", None),
        getattr(runtime_policy_cfg, "n_action_steps", None),
        getattr(runtime_policy_cfg, "num_inference_steps", None),
        getattr(runtime_policy_cfg, "gradient_checkpointing", None),
        getattr(runtime_policy_cfg, "dtype", None),
        getattr(runtime_policy_cfg, "use_relative_actions", None),
    )

    # Setup signal handler for graceful shutdown
    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    policy = None
    robot = None
    preprocessor = None
    postprocessor = None
    get_actions_thread = None
    actor_thread = None
    failure_queue: Queue[ThreadFailure] | None = None
    fatal_thread_failure: ThreadFailure | None = None

    try:
        policy_class = get_policy_class(runtime_policy_cfg.type)

        if runtime_policy_cfg.use_peft:
            from peft import PeftConfig, PeftModel

            peft_pretrained_path = runtime_policy_cfg.pretrained_path
            peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

            policy = policy_class.from_pretrained(
                pretrained_name_or_path=peft_config.base_model_name_or_path,
                config=runtime_policy_cfg,
            )
            policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
        else:
            policy = policy_class.from_pretrained(runtime_policy_cfg.pretrained_path, config=runtime_policy_cfg)

        # Turn on RTC
        policy.config.rtc_config = cfg.rtc

        # Init RTC processort, as by default if RTC disabled in the config
        # The processor won't be created
        policy.init_rtc_processor()

        assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

        policy = policy.to(runtime_policy_cfg.device)
        policy.eval()

        # Apply torch.compile to predict_action_chunk method if enabled
        if cfg.use_torch_compile:
            policy = _apply_torch_compile(policy, cfg)

        preprocessor, postprocessor = _load_policy_processors(runtime_policy_cfg)

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
        failure_queue = Queue()

        # Start chunk requester thread
        get_actions_thread = Thread(
            target=get_actions,
            args=(
                policy,
                robot_wrapper,
                robot_observation_processor,
                preprocessor,
                postprocessor,
                action_queue,
                shutdown_event,
                failure_queue,
                cfg,
            ),
            daemon=True,
            name="GetActions",
        )
        get_actions_thread.start()
        logger.info("Started get actions thread")

        fatal_thread_failure = _wait_for_initial_action_chunk(
            action_queue=action_queue,
            shutdown_event=shutdown_event,
            failure_queue=failure_queue,
        )

        if fatal_thread_failure is None and not shutdown_event.is_set():
            # Start action executor thread only after a chunk is ready. This prevents
            # cold-start inference time from consuming the demo duration.
            actor_thread = Thread(
                target=actor_control,
                args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, failure_queue, cfg),
                daemon=True,
                name="Actor",
            )
            actor_thread.start()
            logger.info("Started actor thread")

            logger.info("Started stop by duration thread")

            # Main thread monitors for duration or shutdown
            logger.info(f"Running demo for {cfg.duration} seconds...")
            start_time = time.monotonic()
            last_queue_log_time = start_time

            while not shutdown_event.is_set() and (time.monotonic() - start_time) < cfg.duration:
                if failure_queue is not None:
                    try:
                        fatal_thread_failure = failure_queue.get_nowait()
                    except Empty:
                        pass
                    else:
                        logger.error(
                            f"{fatal_thread_failure.thread_name} thread failed. Stopping demo and cleaning up."
                        )
                        shutdown_event.set()
                        break

                now = time.monotonic()
                if now - last_queue_log_time >= 10:
                    logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")
                    last_queue_log_time = now

                time.sleep(1)

        logger.info("Demo duration reached or shutdown requested")
    finally:
        shutdown_event.set()

        if actor_thread and actor_thread.is_alive():
            logger.info("Waiting for action executor thread to finish...")
            actor_thread.join(timeout=5)
            if actor_thread.is_alive():
                logger.warning("Actor thread did not stop within timeout; continuing cleanup.")

        if get_actions_thread and get_actions_thread.is_alive():
            logger.info("Waiting for chunk requester thread to finish...")
            get_actions_thread.join(timeout=5)
            if get_actions_thread.is_alive():
                logger.warning("Chunk requester thread did not stop within timeout; continuing cleanup.")

        if robot:
            try:
                robot.disconnect()
                logger.info("Robot disconnected")
            except Exception as e:
                logger.warning(f"Robot disconnect failed during cleanup: {e}")

        if fatal_thread_failure is None and failure_queue is not None:
            try:
                fatal_thread_failure = failure_queue.get_nowait()
            except Empty:
                pass

        logger.info("Cleanup completed")

    if fatal_thread_failure is not None:
        raise RuntimeError(
            f"{fatal_thread_failure.thread_name} thread failed: {fatal_thread_failure.error}"
        ) from fatal_thread_failure.error


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
