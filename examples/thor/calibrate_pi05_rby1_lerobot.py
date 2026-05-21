#!/usr/bin/env python3
"""
FlashRT Thor - Pi0.5 RB-Y1 calibration from a LeRobot v2 dataset.

Usage:
    python examples/thor/calibrate_pi05_rby1_lerobot.py \
        --checkpoint /path/to/pi05_rby1_checkpoint \
        --dataset /path/to/lerobot_v2_dataset \
        --prompt "pick up the object and place it in the target"

    python examples/thor/calibrate_pi05_rby1_lerobot.py \
        --checkpoint /path/to/pi05_rby1_checkpoint \
        --dataset future-robot/rby1-calibration \
        --num-samples 64 \
        --percentile 99.0 \
        --verbose
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import flash_rt
from flash_rt.datasets.lerobot_v2 import (
    DEFAULT_RBY1_IMAGE_KEYS,
    load_calibration_obs,
)


DEFAULT_PROMPT = "pick up the object and place it in the target location"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate FlashRT Pi0.5 RB-Y1 on Thor from LeRobot v2 data"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pi0.5 RB-Y1 checkpoint directory",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "LeRobot v2 local dataset path, or repo_id when using "
            "lerobot==0.1.0"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Task prompt used to initialize the model graph",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of dataset observations to use for calibration",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Activation amax reduction percentile for calibration",
    )
    parser.add_argument(
        "--autotune",
        type=int,
        default=0,
        help="CUDA Graph autotune trials (0=off)",
    )
    parser.add_argument(
        "--hardware",
        default="thor",
        choices=["auto", "thor", "rtx_sm120", "rtx_sm89"],
        help="Backend selection",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size passed to the LeRobot v2 loader",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print loader and calibration details",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading LeRobot v2 calibration observations...")
    obs_list = load_calibration_obs(
        args.dataset,
        n=args.num_samples,
        image_keys=DEFAULT_RBY1_IMAGE_KEYS,
        image_size=args.image_size,
        verbose=args.verbose,
    )
    print(f"Loaded {len(obs_list)} calibration samples from {args.dataset}")

    print("Loading FlashRT Pi0.5 RB-Y1 model...")
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework="torch",
        num_views=4,
        autotune=args.autotune,
        config="pi05_rby1",
        hardware=args.hardware,
    )

    # Initialize prompt and graph capture without running a real inference.
    # Current Pi0.5 Thor set_prompt() performs the built-in calibration pass
    # before capture; the dataset calibration below then overwrites scales and
    # recaptures. Avoiding that startup cost requires a future frontend hook.
    print("Initializing prompt and graphs...")
    model._pipe.set_prompt(args.prompt)

    print(
        f"Running calibration with N={len(obs_list)}, "
        f"percentile={args.percentile:.2f}..."
    )
    model.calibrate(
        obs_list,
        percentile=args.percentile,
        verbose=args.verbose,
    )

    print("Calibration done.")
    print("Calibration cache is under ~/.flash_rt/calibration/")


if __name__ == "__main__":
    main()
