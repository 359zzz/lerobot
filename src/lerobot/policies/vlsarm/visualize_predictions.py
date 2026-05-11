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

"""Visualize VL-SARM predictions on LeRobot datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sarm.sarm_utils import normalize_stage_tau
from lerobot.policies.vlsarm.modeling_vlsarm import VLSARMRewardModel
from lerobot.policies.vlsarm.processor_vlsarm import make_vlsarm_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VL-SARM progress and stage predictions.")
    parser.add_argument(
        "--dataset-repo-id",
        required=True,
        help="Dataset repo id. For local-only datasets this can be any stable identifier.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional local dataset root (e.g. /home/enine/fold_parquet/5_07_run1).",
    )
    parser.add_argument(
        "--reward-model-path",
        required=True,
        help="Path or Hub id of a trained VL-SARM checkpoint.",
    )
    parser.add_argument(
        "--head-mode",
        default="both",
        choices=["sparse", "dense", "both"],
        help="Which VL-SARM head(s) to visualize.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode indices to visualize. Overrides --num-visualizations when provided.",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="How many leading episodes to visualize when --episode-indices is not provided.",
    )
    parser.add_argument(
        "--num-display-frames",
        type=int,
        default=5,
        help="How many thumbnail frames to display at the bottom of each figure.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Infer every Nth frame and interpolate in between for faster visualization.",
    )
    parser.add_argument(
        "--output-dir",
        default="./vlsarm_viz",
        help="Directory where visualizations will be written.",
    )
    return parser.parse_args()


def to_numpy_image(img: Any) -> np.ndarray:
    """Convert image tensor to numpy uint8 (H, W, C)."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[img.shape[0] // 2]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
    return img


def visualize_episode(
    frames: np.ndarray,
    progress_preds: np.ndarray,
    stage_preds: np.ndarray,
    title: str,
    output_path: Path,
    stage_labels: list[str],
    gt_progress: np.ndarray | None = None,
    gt_stages: np.ndarray | None = None,
) -> None:
    """Create a compact visualization with progress, stage probabilities, and thumbnails."""
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress = fig.add_subplot(gs[0])
    ax_stages = fig.add_subplot(gs[1])
    ax_frames = fig.add_subplot(gs[2])

    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices,
            gt_progress,
            linewidth=2,
            color="#28A745",
            linestyle="--",
            label="Ground Truth",
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(title, fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    ax_frames.axis("off")
    num_sample = len(frames)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    thumb_indices = np.linspace(0, len(progress_preds) - 1, num_sample, dtype=int)
    for i, (frame, idx) in enumerate(zip(frames, thumb_indices, strict=True)):
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[int(np.argmax(stage_preds[idx]))][:16]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {output_path}")


def interpolate_progress(
    computed_indices: np.ndarray,
    computed_values: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    computed_indices = np.asarray(computed_indices)
    computed_values = np.asarray(computed_values)
    all_indices = np.asarray(all_indices)

    mask = np.isfinite(computed_values)
    if mask.sum() == 0:
        return np.full(all_indices.shape, np.nan, dtype=np.float32)
    if mask.sum() == 1:
        return np.full(all_indices.shape, float(computed_values[mask][0]), dtype=np.float32)

    out = np.interp(all_indices, computed_indices[mask], computed_values[mask])
    return out.astype(np.float32)


def set_preprocess_eval(preprocess) -> None:
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()


def empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_vlsarm_resources(
    dataset_repo_id: str,
    reward_model_path: str,
    dataset_root: str | None = None,
    device: str = "cuda",
) -> tuple[LeRobotDataset, VLSARMRewardModel, Any]:
    logging.info(f"Loading VL-SARM model: {reward_model_path}")
    reward_model = VLSARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices

    logging.info(f"Loading dataset metadata: repo_id={dataset_repo_id}, root={dataset_root}")
    temp_dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, download_videos=False)
    fps = temp_dataset.fps

    delta_timestamps = {
        image_key: [idx / fps for idx in delta_indices],
        state_key: [idx / fps for idx in delta_indices],
    }
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=True,
    )

    if reward_model.config.annotation_mode != "single_stage":
        reward_model._load_temporal_proportions(dataset.meta)

    preprocess, _ = make_vlsarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    set_preprocess_eval(preprocess)

    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logging.info(f"Using image key: {image_key}")
    return dataset, reward_model, preprocess


def resolve_task_text(sample: dict[str, Any], fallback: str = "perform the task") -> str:
    task = sample.get("task", fallback)
    if isinstance(task, list) and len(task) > 0:
        return str(task[0])
    return str(task)


def resolve_episode_indices(dataset: LeRobotDataset, args: argparse.Namespace) -> list[int]:
    if args.episode_indices:
        return args.episode_indices
    return list(range(min(args.num_visualizations, dataset.num_episodes)))


def visualize_vlsarm_predictions(
    dataset: LeRobotDataset,
    reward_model: VLSARMRewardModel,
    preprocess,
    episode_indices: list[int],
    head_mode: str,
    output_dir: str | Path,
    num_display_frames: int = 5,
    stride: int = 1,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    device = reward_model.runtime_device
    target_idx = reward_model.config.current_frame_index
    dual_mode = reward_model.config.uses_dual_heads

    schemes_to_viz = []
    if head_mode in ("sparse", "both") or not dual_mode:
        schemes_to_viz.append("sparse")
    if head_mode in ("dense", "both") and dual_mode:
        schemes_to_viz.append("dense")

    if not dual_mode and head_mode in ("dense", "both"):
        logging.warning("Dense head requested but checkpoint is not dual-head. Falling back to sparse only.")

    for episode_idx in episode_indices:
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        num_frames = ep_end - ep_start
        first_sample = dataset[ep_start]
        task = resolve_task_text(first_sample)

        display_indices = set(
            [
                ep_start + int(i * (num_frames - 1) / (num_display_frames - 1))
                for i in range(num_display_frames)
            ]
            if num_frames >= num_display_frames and num_display_frames > 1
            else list(range(ep_start, ep_end))
        )
        viz_frames = {frame_idx: to_numpy_image(dataset[frame_idx][image_key]) for frame_idx in display_indices}

        scheme_data: dict[str, dict[str, Any]] = {}
        for scheme in schemes_to_viz:
            num_stages = getattr(reward_model.config, f"num_{scheme}_stages")
            scheme_data[scheme] = {
                "viz_progress": np.full(num_frames, np.nan, dtype=np.float32),
                "viz_stages": np.full((num_frames, num_stages), np.nan, dtype=np.float32),
                "viz_gt_progress": np.full(num_frames, np.nan, dtype=np.float32),
                "viz_gt_stages": np.full(num_frames, np.nan, dtype=np.float32),
                "target_key": f"{scheme}_targets",
                "num_stages": num_stages,
                "temporal_props": getattr(reward_model.config, f"{scheme}_temporal_proportions"),
                "subtask_names": getattr(reward_model.config, f"{scheme}_subtask_names"),
            }

        frame_indices = list(range(ep_start, ep_end, stride))
        if (ep_end - 1) not in frame_indices:
            frame_indices.append(ep_end - 1)
        frame_indices = sorted(set(frame_indices))

        for frame_idx in tqdm(frame_indices, desc=f"VL-SARM episode {episode_idx}", leave=False):
            local_idx = frame_idx - ep_start
            sample = dataset[frame_idx]

            batch = {
                image_key: sample[image_key],
                "task": task,
                "index": frame_idx,
                "episode_index": episode_idx,
            }
            if state_key in sample:
                batch[state_key] = sample[state_key]

            with torch.no_grad():
                processed = preprocess(batch)
                frame_images = processed["frame_images"].to(device)
                task_texts = processed["task_texts"]
                state_features = processed.get("state_features")
                if state_features is not None:
                    state_features = state_features.to(device)
                lengths = processed.get("lengths")
                frame_delta_indices = processed.get("frame_delta_indices")
                frame_is_rewind = processed.get("frame_is_rewind")

                if lengths is not None:
                    lengths = lengths.to(device)
                if frame_delta_indices is not None:
                    frame_delta_indices = frame_delta_indices.to(device)
                if frame_is_rewind is not None:
                    frame_is_rewind = frame_is_rewind.to(device)

                for scheme in schemes_to_viz:
                    sd = scheme_data[scheme]

                    if stride == 1 and sd["target_key"] in processed:
                        gt_target = float(processed[sd["target_key"]][0, target_idx].cpu().item())
                        sd["viz_gt_stages"][local_idx] = int(gt_target)
                        sd["viz_gt_progress"][local_idx] = normalize_stage_tau(
                            gt_target,
                            num_stages=sd["num_stages"],
                            temporal_proportions=sd["temporal_props"],
                            subtask_names=sd["subtask_names"],
                        )

                    reward, stage_probs = reward_model.calculate_rewards(
                        task_texts=task_texts,
                        frame_images=frame_images,
                        state_features=state_features,
                        lengths=lengths,
                        frame_delta_indices=frame_delta_indices,
                        frame_is_rewind=frame_is_rewind,
                        return_all_frames=True,
                        return_stages=True,
                        head_mode=scheme,
                    )

                    reward_np = reward if isinstance(reward, np.ndarray) else reward.cpu().numpy()
                    stages_np = stage_probs if isinstance(stage_probs, np.ndarray) else stage_probs.cpu().numpy()
                    sd["viz_progress"][local_idx] = reward_np[0, target_idx] if reward_np.ndim == 2 else reward_np[target_idx]
                    sd["viz_stages"][local_idx] = (
                        stages_np[0, target_idx, :] if stages_np.ndim == 3 else stages_np[target_idx, :]
                    )

                del processed, frame_images
                if state_features is not None:
                    del state_features
                if lengths is not None:
                    del lengths
                if frame_delta_indices is not None:
                    del frame_delta_indices
                if frame_is_rewind is not None:
                    del frame_is_rewind

            empty_cuda_cache(device)

        if stride > 1:
            all_local = np.arange(num_frames)
            for scheme in schemes_to_viz:
                sd = scheme_data[scheme]
                valid = np.isfinite(sd["viz_progress"])
                valid_idx = np.where(valid)[0]
                if valid_idx.size >= 1:
                    sd["viz_progress"] = interpolate_progress(valid_idx, sd["viz_progress"][valid_idx], all_local)

                    stage_interp = np.zeros_like(sd["viz_stages"], dtype=np.float32)
                    for stage_i in range(sd["num_stages"]):
                        stage_interp[:, stage_i] = interpolate_progress(
                            valid_idx,
                            sd["viz_stages"][valid_idx, stage_i],
                            all_local,
                        )
                    stage_interp = np.clip(stage_interp, 0.0, 1.0)
                    row_sums = stage_interp.sum(axis=1, keepdims=True)
                    non_zero = row_sums.squeeze(-1) > 0
                    stage_interp[non_zero] = stage_interp[non_zero] / row_sums[non_zero]
                    sd["viz_stages"] = stage_interp
                else:
                    sd["viz_stages"] = np.nan_to_num(sd["viz_stages"], nan=0.0)

        ordered_viz_frames = [viz_frames[idx] for idx in sorted(display_indices)]
        for scheme in schemes_to_viz:
            sd = scheme_data[scheme]
            stage_labels = sd["subtask_names"] or [f"Stage {i + 1}" for i in range(sd["num_stages"])]
            output_path = output_dir / f"vlsarm_prediction_ep{episode_idx}_{scheme}.png"
            visualize_episode(
                frames=np.array(ordered_viz_frames),
                progress_preds=sd["viz_progress"],
                stage_preds=sd["viz_stages"],
                title=f"{task} (Episode {episode_idx}, {scheme})",
                output_path=output_path,
                stage_labels=stage_labels,
                gt_progress=sd["viz_gt_progress"] if not np.all(np.isnan(sd["viz_gt_progress"])) else None,
                gt_stages=sd["viz_gt_stages"] if not np.all(np.isnan(sd["viz_gt_stages"])) else None,
            )

        empty_cuda_cache(device)

    logging.info(f"Visualizations saved to: {output_dir.resolve()}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    dataset, reward_model, preprocess = load_vlsarm_resources(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        dataset_root=args.dataset_root,
        device=args.device,
    )
    episode_indices = resolve_episode_indices(dataset, args)

    visualize_vlsarm_predictions(
        dataset=dataset,
        reward_model=reward_model,
        preprocess=preprocess,
        episode_indices=episode_indices,
        head_mode=args.head_mode,
        output_dir=args.output_dir,
        num_display_frames=args.num_display_frames,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
