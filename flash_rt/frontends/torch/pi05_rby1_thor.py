"""FlashRT -- RB-Y1 Pi0.5-style Thor torch frontend.

This frontend is intentionally narrow: it reuses the Pi0.5 Thor compute path
for a Torch safetensors checkpoint trained with:

* 4 image views
* 30 action rows per inference chunk
* 44 action dimensions

It supports the standard single-sample inference/calibration path. CFG and
batched Pi0.5 helper paths are not validated for this RB-Y1 shape.
"""

from __future__ import annotations

from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor


class Pi05Rby1TorchFrontendThor(Pi05TorchFrontendThor):
    """Pi0.5 architecture variant for RB-Y1 on Jetson Thor."""

    DEFAULT_NUM_VIEWS = 4
    CHUNK_SIZE = 30
    ACTION_DIM = 44
    DIFFUSION_STEPS = 10
    OUTPUT_ACTION_DIM = None

    def set_rl_mode(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "RB-Y1 Pi0.5 Thor frontend currently supports standard "
            "single-sample inference only; CFG/RL mode is not validated.")

    def set_batched_mode(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "RB-Y1 Pi0.5 Thor frontend currently supports standard "
            "single-sample inference only; batched mode is not validated.")
