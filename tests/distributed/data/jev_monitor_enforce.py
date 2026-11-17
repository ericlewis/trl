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

"""
Train a one-layer model with the Jev monitor in `"enforce"` mode on CPU, with a local judge that always says stop.
Launched by `tests/distributed/test_distributed.py` with two processes: `python jev_monitor_enforce.py OUTPUT_DIR
SCENARIO` under `torchrun`. `SCENARIO` is `stop`, or `walk_error` (checking the stop checkpoint raises on the main
process). Each rank writes what it saw to `OUTPUT_DIR/rank<rank>.json`; rank `r` keeps its monitor files in
`OUTPUT_DIR/monitor<r>`.
"""

import json
import os
import sys

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from trl.experimental.decision.jev_monitor import JevMonitorCallback, StopRule
from trl.experimental.decision.jev_monitor import callback as callback_module


class Rows(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        return {"x": torch.full((4,), i / 16), "labels": torch.tensor(float(i % 2))}


class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def forward(self, x, labels=None):
        y = self.layer(x).squeeze(-1)
        return {"loss": ((y - labels) ** 2).mean(), "logits": y}


def main(output_dir, scenario):
    set_seed(0)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        logging_steps=1,
        disable_tqdm=True,
        max_steps=12,
    )
    rank = args.process_index
    calls = []

    def judge(digest, questions):
        calls.append(digest["step"])
        return {"stop": {"type": "noul", "noul": 0.99}}

    monitor = JevMonitorCallback(
        judge,
        mode="enforce",
        stop_when=[StopRule("stop")],
        patience=2,
        output_dir=os.path.join(output_dir, f"monitor{rank}"),
    )

    class WaitForVerdict(TrainerCallback):  # runs before the monitor: each verdict is ready when it applies
        def on_evaluate(self, *args, **kwargs):
            monitor.wait(5)

    trainer = Trainer(
        model=Regression(), args=args, train_dataset=Rows(), eval_dataset=Rows(), callbacks=[WaitForVerdict()]
    )
    monitor.attach(trainer)
    if scenario == "walk_error" and rank == 0:

        def walk_error(path):
            raise PermissionError(13, "Permission denied", str(path))

        callback_module._has_weights = walk_error
    trainer.train()
    result = {
        "rank": rank,
        "world_size": args.world_size,
        "global_step": trainer.state.global_step,
        "judge_calls": calls,
        "writes_checkpoints": args.should_save,
    }
    with open(os.path.join(output_dir, f"rank{rank}.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
