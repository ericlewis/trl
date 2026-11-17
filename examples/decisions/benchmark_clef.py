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

# /// script
# dependencies = [
#     "trl[peft]",
#     "transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/50058/head",
#     "accelerate @ git+https://github.com/huggingface/accelerate.git",
# ]
# ///

"""
Benchmark training of the Clef example: the model, data and trainer are built by the example's own `build_trainer`,
so what is measured is what `clef_typed_decisions.py` runs.

Without `--config`, a synthetic recipe runs: a tiny random Qwen2 backbone with a new Clef head on 256 generated records,
on CUDA when there is one and on the CPU otherwise. With `--config`, the example's settings (its YAML) are used as they
are, except that training runs `--steps` steps without evaluating, saving or reporting, and evaluation runs once after.

The JSON summary has the fields of the baseline benchmark's `summary.json` (`startup_seconds`, `trainer_wall_seconds`,
`global_steps`, `head_parameters`, `peak_allocated_gib`, `peak_reserved_gib`, `steady_step_p50`, `steady_step_p95`,
`window_records_per_second`, `window_steps`, `gpu`, `torch`), plus `build_seconds`, `first_step_seconds`,
`steady_records_per_second`, `eval_seconds`, `peak_memory_bytes`, the `steps` (index and seconds, to recompute any
statistic) and the `recipe` (every resolved setting, and the backbone as it was built). Steps after the first 20 are
steady; the window is steps 21 to 320. A step's time is the wall-clock time from the end of the previous step (the
first from the start of training), so it includes loading its batch; the CUDA device is synchronised at each boundary.
`startup_seconds` runs from the start of the build to the start of training, `build_seconds` covers the build alone,
`head_parameters` counts every parameter outside the backbone (the `lm_head` of the lexical prior included), and
`peak_*` are CUDA only (`None` elsewhere).

python examples/decisions/benchmark_clef.py --steps 120

python examples/decisions/benchmark_clef.py --config examples/decisions/clef_typed_decisions.yaml --steps 320 \
    --output output/clef-benchmark.json
"""

import argparse
import json
import math
import random
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainerCallback, set_seed


try:
    from examples.decisions import clef_typed_decisions as clef
except ImportError:  # run as a script: `examples` is a folder of the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.decisions import clef_typed_decisions as clef

from trl import TrlParser
from trl.experimental.decision import DecisionConfig


STEADY_AFTER = 20  # steps before this one warm up (compilation, allocator, caches)
WINDOW = (21, 320)  # the steps of `window_records_per_second`, as in the baseline benchmark
SYNTHETIC_RECORDS = 256
SYNTHETIC_TOKENIZER = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
GIB = 1024**3

QUESTION_POOL = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle the ticket?",
        "criteria": {"billing": "Payments and invoices", "shipping": "Deliveries", "support": None, "sales": None},
    },
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "This week", "Today"]},
    "language": {"type": "choice", "instructions": "Which language is it in?", "criteria": {"en": None, "fr": None}},
    "abusive": {"type": "noul", "instructions": "Is the message abusive?"},
}
WORDS = "order invoice parcel late broken refund charge account password delivery thanks urgent today week".split()


def write_synthetic_recipe(folder: Path) -> dict[str, Any]:
    """
    Write a tiny random causal backbone (with the tokenizer of `SYNTHETIC_TOKENIZER`) and `SYNTHETIC_RECORDS`
    generated records under `folder`, and return the settings of the example that train a new Clef head on them.
    """
    folder = Path(folder)
    tokenizer = AutoTokenizer.from_pretrained(SYNTHETIC_TOKENIZER)
    set_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer) + 2,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    Qwen2ForCausalLM(config).save_pretrained(folder / "backbone")
    tokenizer.save_pretrained(folder / "backbone")
    generator = random.Random(0)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    with open(folder / "data" / "train.jsonl", "w", encoding="utf-8") as file:
        for index in range(SYNTHETIC_RECORDS):
            names = generator.sample(sorted(QUESTION_POOL), generator.randint(1, 3))
            questions = {name: QUESTION_POOL[name] for name in names}
            gold = {}
            for name, question in questions.items():
                if question["type"] == "noul":
                    gold[name] = generator.random() < 0.5
                elif question["type"] == "choice":
                    gold[name] = generator.choice(sorted(question["criteria"]))
                else:
                    gold[name] = generator.randrange(len(question["criteria"]))
            state = f"ticket {index}: " + " ".join(generator.choices(WORDS, k=generator.randint(8, 240)))
            row = {"state": state, "questions": questions, "gold": gold}
            file.write(json.dumps({key: json.dumps(value) for key, value in row.items()}) + "\n")
    return {
        "model_name_or_path": str(folder / "backbone"),
        "dtype": "float32",
        "dataset_name": str(folder / "data"),
        "dataset_config": None,
        "validation_fraction": 0.05,
        "mode": "full",
        "head_type": "joint",
        "head_hidden_size": 32,
        "head_intermediate_size": 64,
        "head_num_layers": 1,
        "head_num_evidence_layers": 1,
        "head_num_attention_heads": 2,
        "max_length": 384,
        "truncation": "state",
        "permute_fields": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 1e-3,
        "head_learning_rate": 3e-3,
        "bf16": False,
        "gradient_checkpointing": False,
        # CUDA when there is one; never another accelerator, so that the recipe means the same everywhere else
        "use_cpu": not torch.cuda.is_available(),
        "seed": 3407,
    }


class StepTimer(TrainerCallback):
    """
    Wall-clock seconds of every optimizer step, from the end of the previous one (the first from the start of
    training), so that each includes loading its batch; the CUDA device is synchronised at each boundary.
    """

    def __init__(self):
        self.steps: list[dict[str, float]] = []
        self.train_start: float | None = None
        self._last: float | None = None

    @staticmethod
    def _now(args) -> float:
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        return time.perf_counter()

    def on_train_begin(self, args, state, control, **kwargs):
        self._last = self.train_start = self._now(args)

    def on_step_end(self, args, state, control, **kwargs):
        now = self._now(args)
        self.steps.append({"step": state.global_step, "seconds": now - self._last})
        self._last = now


def parse_settings(settings: dict[str, Any]):
    """The example's script, training and model arguments from a mapping of its settings (its YAML)."""
    parser = TrlParser((clef.ClefScriptArguments, DecisionConfig, clef.ClefModelArguments))
    return parser.parse_dict(settings)


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def describe_recipe(run, script_args, training_args, model_args, synthetic: bool) -> dict[str, Any]:
    """Every resolved setting, and the model as it was built and trained."""
    model = run.trainer.model
    backbone = model.base_model
    if hasattr(backbone, "get_base_model"):  # a PEFT backbone in `mode: lora`
        backbone = backbone.get_base_model()
    total = parameter_count(model)
    head = total - parameter_count(model.base_model)
    use_fla_kernels = getattr(backbone, "_use_fla_kernels", None)
    return {
        "synthetic": synthetic,
        "script": asdict(script_args),
        "model": asdict(model_args),
        "training": training_args.to_dict(),
        "records_per_step": training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size,
        "device": str(training_args.device),
        "backbone": type(backbone).__name__,
        "parameter_dtype": str(next(backbone.parameters()).dtype).removeprefix("torch."),
        "attn_implementation": getattr(backbone.config, "_attn_implementation", None),
        "activation_checkpointing": bool(getattr(backbone, "is_gradient_checkpointing", False)),
        "head_type": model.config.head_type,
        "head_parameters": head,
        "total_parameters": total,
        "use_fla_kernels": None if use_fla_kernels is None else bool(use_fla_kernels),
        "backbone_weights": str(getattr(model, "backbone_weights", None)),
        "build_report": run.reports,
    }


def summarize(steps: list[dict[str, float]], records_per_step: int) -> dict[str, Any]:
    """The throughput and step-time statistics of `steps`; `None` where there are no steady steps."""
    steady = [step["seconds"] for step in steps if step["step"] > STEADY_AFTER]
    window = [step["seconds"] for step in steps if WINDOW[0] <= step["step"] <= WINDOW[1]]

    def rate(seconds):
        return records_per_step * len(seconds) / sum(seconds) if seconds else None

    return {
        "first_step_seconds": steps[0]["seconds"] if steps else None,
        "steady_records_per_second": rate(steady),
        "steady_step_p50": float(np.percentile(steady, 50)) if steady else None,
        "steady_step_p95": float(np.percentile(steady, 95)) if steady else None,
        "window_records_per_second": rate(window),
        "window_steps": len(window),
    }


def run_benchmark(config: dict[str, Any] | None = None, *, steps: int = 120) -> dict[str, Any]:
    """
    Train the Clef example for `steps` steps with the settings of `config` (the example's YAML as a mapping), or the
    synthetic recipe when it is `None`, then evaluate once, and return the summary described in the module docstring.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    with tempfile.TemporaryDirectory() as folder:
        synthetic = config is None
        settings = write_synthetic_recipe(Path(folder)) if synthetic else dict(config)
        settings |= {
            "max_steps": steps,
            "output_dir": str(Path(folder) / "output"),
            "eval_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "push_to_hub": False,
            "report_to": "none",
        }
        script_args, training_args, model_args = parse_settings(settings)
        timer = StepTimer()
        cuda = training_args.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(training_args.device)
        start = time.perf_counter()
        run = clef.build_trainer(script_args, training_args, model_args)
        build_seconds = time.perf_counter() - start
        run.trainer.add_callback(timer)
        train_start = time.perf_counter()
        run.trainer.train()
        trainer_wall_seconds = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        run.trainer.evaluate()
        if cuda:
            torch.cuda.synchronize(training_args.device)
        eval_seconds = time.perf_counter() - eval_start
        recipe = describe_recipe(run, script_args, training_args, model_args, synthetic)
        peak_allocated = torch.cuda.max_memory_allocated(training_args.device) if cuda else None
        peak_reserved = torch.cuda.max_memory_reserved(training_args.device) if cuda else None
        summary = {
            "build_seconds": build_seconds,
            # From the start of the build to the start of training: the build, then the trainer's own set-up
            "startup_seconds": timer.train_start - start,
            "trainer_wall_seconds": trainer_wall_seconds,
            "eval_seconds": eval_seconds,
            "global_steps": run.trainer.state.global_step,
            "head_parameters": recipe["head_parameters"],
            "peak_memory_bytes": peak_allocated,
            "peak_allocated_gib": None if peak_allocated is None else peak_allocated / GIB,
            "peak_reserved_gib": None if peak_reserved is None else peak_reserved / GIB,
            **summarize(timer.steps, recipe["records_per_step"]),
            "gpu": torch.cuda.get_device_name(training_args.device) if cuda else None,
            "torch": torch.__version__,
            "steps": timer.steps,
            "recipe": recipe,
        }
        run.trainer.end()
    return summary


def to_json(value: Any) -> str:
    """JSON with the settings that are not JSON (enums, paths) as strings, and no NaN."""

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(entry) for entry in item]
        return item

    return json.dumps(finite(value), default=str, allow_nan=False)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", help="The example's YAML; the synthetic recipe when left out.")
    parser.add_argument("--steps", type=int, default=120, help="Training steps.")
    parser.add_argument("--output", help="Also write the JSON summary to this file.")
    args = parser.parse_args(argv)
    config = None
    if args.config is not None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = run_benchmark(config, steps=args.steps)
    text = to_json(summary)
    if args.output is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


if __name__ == "__main__":
    main()
