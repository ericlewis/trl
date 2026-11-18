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

# Needs a Transformers build with Clef training support (Clef's native loss and batched forward, head choice, RLCD) on
# top of Transformers #50058, which adds Clef without them; replace the transformers line with such a build.
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

The settings are parsed exactly as the example parses its `--config` (so `1e-4` in a YAML file is a float), and
`dataloader_drop_last` is forced on, so that every step holds `records_per_step` records and the rates are exact.

The JSON summary has the fields of the reference benchmark's `summary.json`, under the same names. The per-step
`seconds` and the rates computed from them (`steady_*`, `window_records_per_second`) have the reference's meanings;
`trainer_wall_seconds` and `startup_seconds` do not, and the summary's `notes` say so. Every boundary of a timing
synchronises the CUDA device. A step is steady when its index is above `--warmup_steps_excluded` (20 by default).

- `startup_seconds`: from the start of the process (its creation time when `psutil` is installed, else the import of
  this module) to just before `trainer.train()`. Called as a function, `run_benchmark` counts from the given
  `process_start`, or from its own call; `recipe.startup_origin` says which. Its origin is not the reference's, so it
  is not comparable with the reference's `startup_seconds`.
- `build_seconds`: the example's `build_trainer` alone (model, tokenized splits, trainer).
- `trainer_wall_seconds`: `trainer.train()`, which here never evaluates or saves (evaluation and saving are off). The
  reference's covers its evaluations and saves too, so the two are not comparable. `eval_seconds`: one
  `trainer.evaluate()` after training.
- `global_steps`: the optimizer steps taken.
- `steps`: per step, `seconds` from `on_step_begin` to `on_step_end` (forward, backward, optimizer update; the trainer
  fetches and collates the batches of a step before `on_step_begin`), and `data_seconds` from the previous step's end
  (the first: from `on_train_begin`) to `on_step_begin` (the previous step's logging, saving or evaluation, then
  fetching and collating this step's batches).
- `first_step_seconds`: `data_seconds + seconds` of step 1, from `on_train_begin` to the end of the first step.
- `steady_step_p50`, `steady_step_p95`: percentiles of the steady steps' `seconds`.
- `steady_records_per_second`: records per step × steady steps / sum of their `seconds`.
- `window_records_per_second`: the same over the window, the steady steps up to step 320; `window_steps` is its
  length. With the default warm-up the window is steps 21 to 320, as in the reference benchmark.
- `steady_data_p50`: median steady `data_seconds`; `steady_records_per_second_with_data`: steady throughput counting
  `seconds + data_seconds`, so that the cost of collation (permuted prompts are re-rendered) stays visible.
- `head_parameters`: the decision head alone, without the backbone or the `lm_head` table of the lexical prior (in
  `recipe.lexical_prior_parameters`, 0 when tied to the backbone's embeddings, `None` when there is none).
- `peak_allocated_gib`, `peak_reserved_gib`, `peak_memory_bytes` (allocated): CUDA only, `None` elsewhere. The peaks are
  reset just before `trainer.train()` and read at every step end, so they cover training (with any evaluation during
  it) but not loading nor the final evaluation.
- `gpu`, `torch`: the device name and the torch version.
- `recipe`: every resolved setting (`model_init_kwargs` other than a few harmless keys redacted, since they may hold a
  token), and the backbone as it was built (class, floating-point dtypes, the attention implementation the backbone
  uses, activation checkpointing, head type, parameter counts, KLM's `_use_fla_kernels`).
- `notes`: what `trainer_wall_seconds` and `startup_seconds` cover, and that they are not comparable with the
  reference's.

Under torchrun only the main process prints and writes the summary.

python examples/decisions/benchmark_clef.py --steps 120

python examples/decisions/benchmark_clef.py --config examples/decisions/clef_typed_decisions.yaml --steps 320 \
    --output output/clef-benchmark.json
"""

import time


# Before the heavy imports: the start of the process when psutil cannot tell it
MODULE_IMPORT_TIME = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from accelerate import PartialState  # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainerCallback, set_seed  # noqa: E402


try:
    from examples.decisions import clef_typed_decisions as clef
except ImportError:  # run as a script: `examples` is a folder of the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.decisions import clef_typed_decisions as clef

from trl import TrlParser  # noqa: E402
from trl.experimental.decision import DecisionConfig  # noqa: E402


WINDOW_END = 320  # the last step of `window_records_per_second`, as in the reference benchmark
# The `model_init_kwargs` the recipe prints; the others (a token, say) are redacted
SHOWN_MODEL_INIT_KWARGS = ("dtype", "revision", "attn_implementation", "trust_remote_code")
NOTES = {
    "trainer_wall_seconds": "trainer.train() alone, which runs no evaluation and no save here; the reference's includes "
    "its evaluations and saves, so the two are not comparable",
    "startup_seconds": "from the start of the process (see recipe.startup_origin) to just before trainer.train(); its "
    "origin differs from the reference's, so the two are not comparable",
}
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


def write_synthetic_recipe(folder: Path, records: int = SYNTHETIC_RECORDS) -> dict[str, Any]:
    """
    Write a tiny random causal backbone (with the tokenizer of `SYNTHETIC_TOKENIZER`) and `records` generated records
    under `folder`, and return the settings of the example that train a new Clef head on them.
    """
    folder = Path(folder)
    # Explicit, so that a tokenizer whose repository ships code is never run nor asked about on stdin
    tokenizer = AutoTokenizer.from_pretrained(SYNTHETIC_TOKENIZER, trust_remote_code=False)
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
        for index in range(records):
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
    For every optimizer step: `seconds` from `on_step_begin` to `on_step_end` (forward, backward and optimizer update;
    the trainer fetches and collates the step's batches before `on_step_begin`), and `data_seconds` from the end of
    the previous step (the first: from `on_train_begin`) to `on_step_begin` (the previous step's logging, saving or
    evaluation, then fetching and collating this step's batches). The CUDA device is synchronised at every boundary,
    and the peak CUDA memory is read at every step end.
    """

    def __init__(self):
        self.steps: list[dict[str, float]] = []
        self.train_start: float | None = None
        self.peak_allocated: int | None = None
        self.peak_reserved: int | None = None
        self._begin: float | None = None
        self._last: float | None = None

    @staticmethod
    def _now(args) -> float:
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        return time.perf_counter()

    def on_train_begin(self, args, state, control, **kwargs):
        self._last = self.train_start = self._now(args)

    def on_step_begin(self, args, state, control, **kwargs):
        self._begin = self._now(args)

    def on_step_end(self, args, state, control, **kwargs):
        now = self._now(args)
        self.steps.append(
            {"step": state.global_step, "seconds": now - self._begin, "data_seconds": self._begin - self._last}
        )
        self._last = now
        if args.device.type == "cuda":
            self.peak_allocated = max(self.peak_allocated or 0, torch.cuda.max_memory_allocated(args.device))
            self.peak_reserved = max(self.peak_reserved or 0, torch.cuda.max_memory_reserved(args.device))


def process_start() -> tuple[float, str]:
    """The `time.time()` the process started at, and how it was found."""
    try:
        import psutil

        return psutil.Process().create_time(), "process creation"
    except Exception:  # psutil missing, or the platform does not tell
        return MODULE_IMPORT_TIME, "benchmark module import"


def parse_settings(settings: dict[str, Any]):
    """
    The example's script, training and model arguments from a mapping of its settings (its YAML), parsed exactly as
    the example parses `--config`: the settings become the defaults of its parser, which converts them to the type of
    each field (YAML reads `1e-4` as a string).
    """
    parser = TrlParser((clef.ClefScriptArguments, DecisionConfig, clef.ClefModelArguments))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "settings.yaml"
        path.write_text(yaml.safe_dump(json.loads(json.dumps(settings, default=str))), encoding="utf-8")
        return parser.parse_args_and_config(["--config", str(path)])


def describe_recipe(run, script_args, training_args, model_args, synthetic: bool) -> dict[str, Any]:
    """Every resolved setting, and the model as it was built and trained."""
    model = run.trainer.model
    backbone = model.base_model
    if hasattr(backbone, "get_base_model"):  # a PEFT backbone in `mode: lora`
        backbone = backbone.get_base_model()
    # Parameters by identity, so a table tied to the backbone's embeddings is counted once, as the backbone's
    parameters = {id(parameter): parameter for parameter in model.parameters()}
    in_backbone = {id(parameter) for parameter in model.base_model.parameters()}
    lm_head = getattr(model, "lm_head", None)
    in_lexical_prior = {id(parameter) for parameter in lm_head.parameters()} - in_backbone if lm_head else set()
    total = sum(parameter.numel() for parameter in parameters.values())
    lexical_prior = sum(parameters[key].numel() for key in in_lexical_prior) if lm_head is not None else None
    head = sum(parameter.numel() for key, parameter in parameters.items() if key not in in_backbone | in_lexical_prior)
    use_fla_kernels = getattr(backbone, "_use_fla_kernels", None)
    dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in backbone.parameters()
            if parameter.is_floating_point()
        }
    )
    training = training_args.to_dict()
    if training.get("model_init_kwargs"):
        training["model_init_kwargs"] = {
            key: value if key in SHOWN_MODEL_INIT_KWARGS else "<redacted>"
            for key, value in training["model_init_kwargs"].items()
        }
    return {
        "synthetic": synthetic,
        "script": asdict(script_args),
        "model": asdict(model_args),
        "training": training,
        "records_per_step": training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size,
        "device": str(training_args.device),
        "backbone": type(backbone).__name__,
        "parameter_dtypes": dtypes,
        # What the backbone uses; the head keeps its own default
        "attn_implementation": getattr(backbone.config, "_attn_implementation", None),
        "activation_checkpointing": bool(getattr(backbone, "is_gradient_checkpointing", False)),
        "head_type": model.config.head_type,
        "head_parameters": head,
        "lexical_prior_parameters": lexical_prior,
        "total_parameters": total,
        "use_fla_kernels": None if use_fla_kernels is None else bool(use_fla_kernels),
        "backbone_weights": str(getattr(model, "backbone_weights", None)),
        "build_report": run.reports,
    }


def summarize(steps: list[dict[str, float]], records_per_step: int, warmup_steps_excluded: int) -> dict[str, Any]:
    """The throughput and step-time statistics of `steps`; `None` where there are no steady steps."""
    steady = [step for step in steps if step["step"] > warmup_steps_excluded]
    window = [step for step in steps if warmup_steps_excluded < step["step"] <= WINDOW_END]

    def rate(seconds):
        return records_per_step * len(seconds) / sum(seconds) if seconds else None

    def percentile(values, q):
        return float(np.percentile(values, q)) if values else None

    return {
        "first_step_seconds": steps[0]["data_seconds"] + steps[0]["seconds"] if steps else None,
        "steady_records_per_second": rate([step["seconds"] for step in steady]),
        "steady_step_p50": percentile([step["seconds"] for step in steady], 50),
        "steady_step_p95": percentile([step["seconds"] for step in steady], 95),
        "steady_data_p50": percentile([step["data_seconds"] for step in steady], 50),
        "steady_records_per_second_with_data": rate([step["seconds"] + step["data_seconds"] for step in steady]),
        "window_records_per_second": rate([step["seconds"] for step in window]),
        "window_steps": len(window),
    }


def run_benchmark(
    config: dict[str, Any] | None = None,
    *,
    steps: int = 120,
    warmup_steps_excluded: int = 20,
    process_start: tuple[float, str] | None = None,
    synthetic_records: int = SYNTHETIC_RECORDS,
) -> dict[str, Any]:
    """
    Train the Clef example for `steps` steps with the settings of `config` (the example's YAML as a mapping), or the
    synthetic recipe when it is `None`, then evaluate once, and return the summary described in the module docstring.

    `warmup_steps_excluded` steps are left out of the steady statistics. `process_start` is the `time.time()` that
    `startup_seconds` counts from, and how it was found; by default, the call of this function.
    `synthetic_records` is the number of records the synthetic recipe generates.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if warmup_steps_excluded < 0:
        raise ValueError("warmup_steps_excluded must not be negative")
    origin_time, origin = process_start or (time.time(), "run_benchmark call")
    with tempfile.TemporaryDirectory() as folder:
        synthetic = config is None
        settings = write_synthetic_recipe(Path(folder), synthetic_records) if synthetic else dict(config)
        settings |= {
            "max_steps": steps,
            "output_dir": str(Path(folder) / "output"),
            "eval_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "push_to_hub": False,
            "report_to": "none",
            # Every step holds records_per_step records, so the rates are exact
            "dataloader_drop_last": True,
        }
        script_args, training_args, model_args = parse_settings(settings)
        timer = StepTimer()
        cuda = training_args.device.type == "cuda"
        start = time.perf_counter()
        run = clef.build_trainer(script_args, training_args, model_args)
        build_seconds = time.perf_counter() - start
        run.trainer.add_callback(timer)
        if cuda:  # the peaks cover training (and any evaluation during it), not loading nor the final evaluation
            torch.cuda.reset_peak_memory_stats(training_args.device)
        startup_seconds = time.time() - origin_time
        train_start = time.perf_counter()
        run.trainer.train()
        trainer_wall_seconds = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        run.trainer.evaluate()
        if cuda:
            torch.cuda.synchronize(training_args.device)
        eval_seconds = time.perf_counter() - eval_start
        recipe = describe_recipe(run, script_args, training_args, model_args, synthetic)
        recipe["warmup_steps_excluded"] = warmup_steps_excluded
        recipe["startup_origin"] = origin
        summary = {
            "build_seconds": build_seconds,
            "startup_seconds": startup_seconds,
            "trainer_wall_seconds": trainer_wall_seconds,
            "eval_seconds": eval_seconds,
            "global_steps": run.trainer.state.global_step,
            "head_parameters": recipe["head_parameters"],
            "peak_memory_bytes": timer.peak_allocated,
            "peak_allocated_gib": None if timer.peak_allocated is None else timer.peak_allocated / GIB,
            "peak_reserved_gib": None if timer.peak_reserved is None else timer.peak_reserved / GIB,
            **summarize(timer.steps, recipe["records_per_step"], warmup_steps_excluded),
            "gpu": torch.cuda.get_device_name(training_args.device) if cuda else None,
            "torch": torch.__version__,
            "steps": timer.steps,
            "recipe": recipe,
            "notes": dict(NOTES),
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
    parser.add_argument(
        "--warmup_steps_excluded", type=int, default=20, help="Steps left out of the steady statistics."
    )
    args = parser.parse_args(argv)
    config = None
    if args.config is not None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = run_benchmark(
        config, steps=args.steps, warmup_steps_excluded=args.warmup_steps_excluded, process_start=process_start()
    )
    if PartialState().is_main_process:  # one summary under torchrun
        text = to_json(summary)
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(text)
    return summary


if __name__ == "__main__":
    main()
