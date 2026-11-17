# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import logging
import math
from collections.abc import Sequence

from transformers import TrainerCallback


logger = logging.getLogger(__name__)


class RlcdScheduleCallback(TrainerCallback):
    """
    Run Clef's RLCD term as a phase after cross-entropy training, with a noise scale annealed per epoch.

    Before `start_epoch` the model's `config.rlcd_weight` is 0, so training is plain cross-entropy; from `start_epoch`
    on it is `weight`. The values are a function of `state.epoch` alone, read at the start of every step, so a resumed
    run and every process of a distributed run use the same ones. The model applies the RLCD term in training only, so
    evaluation stays the supervised loss. When training ends, the config keeps the values in effect at the last step,
    so the saved model records what it was trained with: a run that stops before `start_epoch` keeps weight 0.

    Args:
        weight (`float`):
            Weight of the RLCD policy term from `start_epoch` on. Must not be negative.
        start_epoch (`float`, *optional*, defaults to `0.0`):
            Epoch at which RLCD starts. Must not be negative.
        sigmas (`Sequence[float]`, *optional*):
            One noise scale per epoch of the RLCD phase, counted from `start_epoch`: the first applies to
            `[start_epoch, start_epoch + 1)`, and so on, and the last is held for any further epochs. `None` leaves
            `config.rlcd_sigma` alone. Each must be positive.
    """

    def __init__(self, weight: float, *, start_epoch: float = 0.0, sigmas: Sequence[float] | None = None):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight must be a non-negative number, got {weight}")
        if not math.isfinite(start_epoch) or start_epoch < 0:
            raise ValueError(f"start_epoch must be a non-negative number, got {start_epoch}")
        if sigmas is not None:
            sigmas = [float(sigma) for sigma in sigmas]
            if not sigmas:
                raise ValueError("sigmas must not be empty; pass None to keep the config's rlcd_sigma")
            if any(not math.isfinite(sigma) or sigma <= 0 for sigma in sigmas):
                raise ValueError(f"every sigma must be positive, got {sigmas}")
        self.weight = float(weight)
        self.start_epoch = float(start_epoch)
        self.sigmas = sigmas

    def values(self, epoch: float | None) -> tuple[float, float | None]:
        """The weight and the noise scale (`None`: not set by the schedule) at `epoch`; `None` counts as 0."""
        epoch = 0.0 if epoch is None else float(epoch)
        if epoch < self.start_epoch:
            return 0.0, None
        sigma = None
        if self.sigmas is not None:
            sigma = self.sigmas[min(int(math.floor(epoch - self.start_epoch)), len(self.sigmas) - 1)]
        return self.weight, sigma

    @staticmethod
    def _config(model):
        config = getattr(getattr(model, "module", model), "config", None)
        if config is None or not hasattr(config, "rlcd_weight"):
            raise ValueError("RlcdScheduleCallback needs a model whose config has `rlcd_weight` (a Clef model)")
        return config

    def _apply(self, state, model):
        config = self._config(model)
        weight, sigma = self.values(state.epoch)
        changes = []
        if config.rlcd_weight != weight:
            changes.append(f"rlcd_weight {config.rlcd_weight} -> {weight}")
            config.rlcd_weight = weight
        if sigma is not None and config.rlcd_sigma != sigma:
            changes.append(f"rlcd_sigma {config.rlcd_sigma} -> {sigma}")
            config.rlcd_sigma = sigma
        if changes and state.is_world_process_zero:
            logger.info("RLCD schedule at epoch %s: %s", state.epoch, ", ".join(changes))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._config(model)  # raises for a model without RLCD
        self._apply(state, model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._apply(state, model)
