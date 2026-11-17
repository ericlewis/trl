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

import copy
import inspect
import json
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from .client import SystemOneClient, validate_questions
from .digest import build_digest, json_safe
from .ledger import Ledger


MODES = ("advise", "enforce")
# A learned logit scale above this (a temperature below 1/100) is reported as a sentinel; it never stops training.
SCALE_LIMIT = math.log(100.0)
# The metrics of `compute_calibration_metrics`, and the loss: what the digest reports by default.
DEFAULT_METRICS = (
    "loss",
    "accuracy",
    "kl",
    "brier",
    "ece",
    "mean_confidence",
    "soft_nll",
    "confidence_gap",
    "selection_score",
)
DEFAULT_OBJECTIVE = {"selection_score": 1.0}
_HISTORY_LIMIT = 256

# Generic questions about the digest: they name no model, head or dataset.
DEFAULT_QUESTIONS = {
    "stop": {
        "type": "noul",
        "instructions": (
            "Judging only from this digest of the run's evaluations, has further training stopped paying off, so "
            "that it is better to stop now and keep the best evaluation so far?"
        ),
    },
    "trend": {
        "type": "choice",
        "instructions": "Over the latest evaluations, where is the objective heading?",
        "criteria": {
            "improving": "The objective is still getting better.",
            "flat": "The objective has stopped moving.",
            "worsening": "The objective is getting worse.",
        },
    },
    "health": {
        "type": "score",
        "instructions": "How healthy does the run look from this digest?",
        "criteria": [
            "Broken: metrics are missing, non-finite or diverging.",
            "Struggling: metrics are unstable or getting worse.",
            "Healthy: metrics are finite and steady or improving.",
        ],
    },
}

_NOT_READY = object()


@dataclass(frozen=True)
class StopRule:
    """
    A condition on the judge's answers under which [`JevMonitorCallback`] stops training in `"enforce"` mode.

    The rule holds for a verdict when the probability it reads is at least `threshold`: `answers[question]["noul"]` for
    a `noul` question (`option` is `None`), else `answers[question]["probabilities"][option]` for a `choice` question
    (an option of its criteria) or a `score` question (a level, `"0"`, `"1"`, ...).

    Args:
        question (`str`):
            Name of the question.
        option (`str`, *optional*):
            Option whose probability is read; `None` for a `noul` question.
        threshold (`float`, *optional*, defaults to `0.9`):
            Probability, in `(0, 1]`, at or above which the rule holds.
    """

    question: str
    option: str | None = None
    threshold: float = 0.9

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question:
            raise ValueError(f"StopRule.question must be a question name, got {self.question!r}")
        if self.option is not None and not isinstance(self.option, str):
            raise ValueError(f"StopRule.option must be a string or None, got {self.option!r}")
        threshold = self.threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError(f"StopRule.threshold must be in (0, 1], got {threshold!r}")


def _probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _default_broadcast(value):
    """Rank 0's `value` on every process; the value itself when there is a single process."""
    from accelerate import PartialState

    if PartialState().num_processes <= 1:
        return value
    from accelerate.utils import broadcast_object_list

    return broadcast_object_list([value], from_process=0)[0]


# Files that hold weights in a saved checkpoint (`training_args.bin` does not count)
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _has_weights(directory: str | os.PathLike) -> bool:
    """Whether `directory` holds at least one non-empty weights file, at any depth."""
    directory = Path(directory)
    if not directory.is_dir():
        return False
    for path in directory.rglob("*"):
        if path.suffix in _WEIGHT_SUFFIXES and path.name != "training_args.bin" and path.is_file():
            if path.stat().st_size > 0:
                return True
    return False


def _is_nonfinite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return not math.isfinite(float(value))
    except (OverflowError, ValueError):
        return True


class JevMonitorCallback(TrainerCallback):
    """
    Lets a decision model behind a System One API (the judge) watch a training run, and optionally stop it.

    At every evaluation during training, the callback summarises the run as a small digest of its evaluation metrics
    (never training data; see [`~experimental.decision.jev_monitor.build_digest`]) and hands it to a background thread,
    which asks the judge `questions` about it. Nothing waits for the judge: the verdict on the evaluation at step S is
    applied at the next evaluation, and if it has not arrived by then it is recorded as missing and discarded when it
    comes. The verdict on the last evaluation of a run is recorded but never applied.

    In `"advise"` mode (the default) the callback never touches the Trainer's control flow: it only records what it saw
    and would have done in `ledger.jsonl`. In `"enforce"` mode it may set `control.should_training_stop`, and nothing
    else, when a stop rule has held for `patience` consecutive verdicts (a missing or malformed answer neither breaks
    nor extends a streak), or at once when an evaluation metric is NaN or infinite, or at the next evaluation once
    gradients were non-finite on `patience_gradients` consecutive checks of optimizer steps that were taken. A step the
    optimizer skipped (a loss-scale overflow under fp16, normal operation) is recorded as a `note`, never a sentinel;
    without an attached trainer to tell, under fp16, every non-finite check is a `note`. One clean check clears the
    gradient sentinel. Before stopping, every process calls `trainer.save_model` to
    `<args.output_dir>/monitor-stop-<step>`: the save is collective under FSDP and DeepSpeed, and processes that do not
    write return at once under DDP. The main process then checks that the directory holds a non-empty weights file, and
    only then is the stop enforced and `intervened.json` written; otherwise training goes on and the reason is
    recorded. It never sets `should_save`, `should_evaluate` or `should_epoch_stop`. A learned logit scale above
    `log(100)` is recorded, never a reason to stop.

    Only the main process talks to the judge and writes files. The decision to stop is sent from it to every process
    with `broadcast`, so all processes stop at the same step. Evaluations outside `train()`, or whose metrics do not
    carry `metric_prefix`, are ignored; an evaluation that comes after a training step began and before it ended means
    `train()` failed, and ends the monitor's run. Several evaluations at one step (a dict of evaluation datasets) are
    merged into one record of that step, their metrics named without the prefix (`<dataset>_<metric>`); the judge is
    sent the merged digest if its request has not started yet, else the digest of the first. Set `metric_prefix` (for
    example `"eval_validation"`) to watch one dataset only. When training resumes, ledger records after the resumed
    step are dropped and the streaks are rebuilt from the rest. No hook raises into the Trainer: errors are recorded in
    the ledger.

    Limits: under FSDP the main process sees only its own shard of the gradients, and under DeepSpeed ZeRO the
    gradients may not be available to callbacks, so the gradient check may see nothing; a zero-dimensional scale
    parameter that is sharded is not found. The main process may finish up to `finish_timeout` seconds after the
    others. Resuming does not apply the verdict on the evaluation of the checkpoint's own step. A reused output
    directory keeps the earlier ledger records up to the step training starts or resumes from. An evaluation is
    recognised as outside training only once a training step has begun, so a failure while loading a batch is not
    detected until the next evaluation is followed by no training.

    Args:
        backend ([`~experimental.decision.jev_monitor.SystemOneClient`] or `Callable[[dict, dict], dict | None]`, *optional*):
            The judge: a client, or a function of `(digest, questions)` returning the answers mapping or `None`. `None`
            asks nothing: only the sentinels run.
        questions (`dict`, *optional*):
            Typed questions about the digest; defaults to a small generic set (`DEFAULT_QUESTIONS`).
        mode (`str`, *optional*, defaults to `"advise"`):
            `"advise"` or `"enforce"`. `"enforce"` needs [`~JevMonitorCallback.attach`].
        stop_when (`Sequence[StopRule]`, *optional*):
            Rules under which training stops in `"enforce"` mode; any one of them is enough.
        patience (`int`, *optional*, defaults to `3`):
            Consecutive verdicts a rule must hold for.
        metric_prefix (`str`, *optional*, defaults to `"eval"`):
            Prefix of the evaluation metrics to watch.
        output_dir (`str` or `Path`, *optional*):
            Where `ledger.jsonl` and `intervened.json` go; defaults to `<args.output_dir>/jev_monitor`.
        grad_norm_every (`int`, *optional*):
            Optimizer steps between gradient checks; defaults to `args.logging_steps`.
        patience_gradients (`int`, *optional*, defaults to `2`):
            Consecutive checks with non-finite gradients, on optimizer steps that were taken, before they can stop
            training.
        budget_bytes (`int`, *optional*, defaults to `8192`):
            Size bound of the digest.
        control_ledger (`str` or `Path`, *optional*):
            Ledger of a matched control run; its evaluations up to the current step are summarised in the digest.
        metrics (`Sequence[str]`, *optional*):
            Metrics (without prefix) the digest reports; defaults to the loss and those of
            [`~experimental.decision.compute_calibration_metrics`].
        objective (`dict[str, float]`, *optional*):
            Weights of the metrics in the run's objective; defaults to `{"selection_score": 1.0}`.
        broadcast (`Callable`, *optional*):
            Sends rank 0's value to every process and returns it; defaults to `accelerate`'s `broadcast_object_list`
            when there are several processes.
        finish_timeout (`float`, *optional*, defaults to `5.0`):
            Seconds `on_train_end` waits for the last verdict.
    """

    def __init__(
        self,
        backend: SystemOneClient | Callable[[dict, dict], dict | None] | None = None,
        questions: dict | None = None,
        *,
        mode: str = "advise",
        stop_when: Sequence[StopRule] = (),
        patience: int = 3,
        metric_prefix: str = "eval",
        output_dir: str | os.PathLike | None = None,
        grad_norm_every: int | None = None,
        patience_gradients: int = 2,
        budget_bytes: int = 8192,
        control_ledger: str | os.PathLike | None = None,
        metrics: Sequence[str] | None = None,
        objective: dict[str, float] | None = None,
        broadcast: Callable[[Any], Any] | None = None,
        finish_timeout: float = 5.0,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if isinstance(backend, SystemOneClient):
            ask = backend.ask
        elif backend is None or callable(backend):
            ask = backend
        else:
            raise ValueError("backend must be a SystemOneClient, a callable or None")
        questions = copy.deepcopy(DEFAULT_QUESTIONS if questions is None else questions)
        validate_questions(questions)
        stop_when = tuple(stop_when)
        for rule in stop_when:
            self._check_rule(rule, questions)
        if stop_when and ask is None:
            raise ValueError("stop_when needs a backend to ask")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be a positive integer, got {patience!r}")
        if isinstance(patience_gradients, bool) or not isinstance(patience_gradients, int) or patience_gradients < 1:
            raise ValueError(f"patience_gradients must be a positive integer, got {patience_gradients!r}")
        if grad_norm_every is not None and (
            isinstance(grad_norm_every, bool) or not isinstance(grad_norm_every, int) or grad_norm_every < 1
        ):
            raise ValueError(f"grad_norm_every must be a positive integer or None, got {grad_norm_every!r}")
        if not isinstance(metric_prefix, str) or not metric_prefix:
            raise ValueError("metric_prefix must be a non-empty string")
        if not isinstance(finish_timeout, (int, float)) or not 0 <= finish_timeout < math.inf:
            raise ValueError(f"finish_timeout must be a non-negative number of seconds, got {finish_timeout!r}")

        self.backend = backend
        self.questions = questions
        self.mode = mode
        self.stop_when = stop_when
        self.patience = patience
        self.metric_prefix = metric_prefix
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.grad_norm_every = grad_norm_every
        self.patience_gradients = patience_gradients
        self.budget_bytes = int(budget_bytes)
        self.control_ledger = None if control_ledger is None else Path(control_ledger)
        self.metrics = tuple(DEFAULT_METRICS if metrics is None else metrics)
        self.objective = dict(DEFAULT_OBJECTIVE if objective is None else objective)
        self.finish_timeout = float(finish_timeout)
        self.ledger: Ledger | None = None if self.output_dir is None else Ledger(self.output_dir / "ledger.jsonl")
        self._ask = ask
        self._broadcast = broadcast or _default_broadcast
        self._trainer = None

        # Worker: one request slot (a newer request replaces one not yet started), results by evaluation step
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None
        self._job: tuple[int, int, dict] | None = None  # (generation, step, digest)
        self._generation = 0  # one per `train()`: results of an earlier run are dropped
        self._busy = False
        self._closing = False
        self._active = False  # worker results are recorded only while this is set
        self._results: dict[int, Any] = {}
        self._discarded: set[int] = set()
        self._reset()

    # -- set-up ------------------------------------------------------------------------------------------------

    @staticmethod
    def _check_rule(rule, questions: dict) -> None:
        if not isinstance(rule, StopRule):
            raise ValueError(f"stop_when takes StopRule instances, got {rule!r}")
        question = questions.get(rule.question)
        if not isinstance(question, dict):
            raise ValueError(f"StopRule names the question {rule.question!r}, which is not one of {sorted(questions)}")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "noul":
            if rule.option is not None:
                raise ValueError(f"StopRule on the noul question {rule.question!r} reads `noul`: leave option None")
            return
        options = list(criteria) if kind == "choice" else [str(level) for level in range(len(criteria or []))]
        if rule.option not in options:
            raise ValueError(
                f"StopRule on the {kind} question {rule.question!r} needs an option among {options}, "
                f"got {rule.option!r}"
            )

    def _reset(self) -> None:
        self._training = False
        self._rank0 = True
        self._stopped = False
        self._model = None
        self._every = 1
        self._groups: dict[str, list] | None = None
        self._pending_step: int | None = None
        self._history: list[dict] = []
        self._streaks: list[list[dict]] = [[] for _ in self.stop_when]
        self._grad_norms: dict[str, float] = {}
        self._grad_check: dict | None = None  # non-finite check waiting for the optimizer step's outcome
        self._grad_streak = 0
        self._grad_sentinel: dict | None = None
        self._armed = True  # no training step has begun without ending since the last one ended
        self._last_eval_step: int | None = None
        self._high_scales: set[str] = set()
        self._control_warned = False

    def attach(self, trainer) -> "JevMonitorCallback":
        """Give the callback the trainer it saves with before an enforced stop, and add it to the trainer's callbacks."""
        self._trainer = trainer
        if self not in trainer.callback_handler.callbacks:
            trainer.add_callback(self)
        return self

    def _record(self, kind: str, step: int, payload: dict) -> None:
        """Ledger a record from the main process; never raises."""
        if not self._rank0 or self.ledger is None:
            return
        try:
            self.ledger.append(kind, step, payload)
        except Exception:  # noqa: BLE001 - Ledger.append does not raise; this is a last guard
            pass

    def _error(self, where: str, step: Any, error: BaseException) -> None:
        try:
            step = int(step)
        except (TypeError, ValueError, OverflowError):
            step = 0
        self._record("error", step, {"where": where, "error": f"{type(error).__name__}: {error}"[:500]})

    # -- worker ------------------------------------------------------------------------------------------------

    def _start_worker(self) -> None:
        with self._cond:
            self._closing = False
            self._active = True
            self._generation += 1
            self._job = None
            self._results.clear()
            self._discarded.clear()
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._work, name="jev-monitor", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._closing:
                    self._cond.wait()
                if self._job is None:
                    return
                generation, step, digest = self._job
                self._job, self._busy = None, True
            try:
                self._answer(generation, step, digest)
            except Exception:  # noqa: BLE001 - the worker must survive anything
                pass
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _answer(self, generation: int, step: int, digest: dict) -> None:
        started = time.monotonic()
        error = None
        try:
            answers = self._ask(digest, self.questions)
        except Exception as e:  # noqa: BLE001 - a broken backend is a missing answer
            answers, error = None, f"{type(e).__name__}: {e}"[:500]
        if answers is not None and not isinstance(answers, dict):
            answers, error = None, f"the backend returned a {type(answers).__name__}, not a mapping"
        if answers is None and error is None:
            error = getattr(self.backend, "last_error", None) or "the backend gave no answer"
        payload = {"answers": answers, "error": error, "seconds": round(time.monotonic() - started, 3)}
        # Recorded under the lock, so nothing is written once `on_train_end` has closed the worker
        with self._cond:
            if not self._active or generation != self._generation:  # training ended, or another run started
                return
            late = step in self._discarded
            self._discarded.discard(step)
            if not late:
                self._results[step] = answers
            self._record("late" if late else "answer", step, payload)

    def wait(self, timeout: float = 5.0) -> bool:
        """
        Wait up to `timeout` seconds for the background request to finish. Returns `True` if nothing is pending.
        """
        with self._cond:
            return self._cond.wait_for(lambda: self._job is None and not self._busy, timeout)

    def close(self) -> None:
        """Stop the background thread; it records nothing more. Idempotent."""
        with self._cond:
            self._closing = True
            self._active = False
            self._job = None
            self._cond.notify_all()

    # -- hooks -------------------------------------------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.mode == "enforce" and self._trainer is None:
            raise RuntimeError("JevMonitorCallback(mode='enforce') needs the trainer: call attach(trainer) first")
        self._reset()
        self._training = True  # on every process, whatever fails below, so that all take the same path
        self._rank0 = bool(getattr(state, "is_world_process_zero", True))
        self._model = model
        step = int(getattr(state, "global_step", 0) or 0)
        try:
            every = self.grad_norm_every
            if every is None:
                logging_steps = float(getattr(args, "logging_steps", 1) or 1)
                if logging_steps < 1:  # a ratio of the planned steps
                    logging_steps = math.ceil(logging_steps * max(int(getattr(state, "max_steps", 1) or 1), 1))
                every = int(logging_steps)
            self._every = max(int(every), 1)
            if self.ledger is None:
                self.ledger = Ledger(Path(args.output_dir) / "jev_monitor" / "ledger.jsonl")
            if not self._rank0:
                return
            self._resume(step)
            if self._ask is not None:
                self._start_worker()
            self._record(
                "start",
                step,
                {
                    "mode": self.mode,
                    "questions": sorted(self.questions),
                    "stop_when": [asdict(rule) for rule in self.stop_when],
                    "patience": self.patience,
                    "grad_norm_every": self._every,
                },
            )
        except Exception as e:  # noqa: BLE001
            self._error("on_train_begin", step, e)

    def _resume(self, step: int) -> None:
        records = self.ledger.read()
        if not records:
            return
        self.ledger.truncate_after(step)
        records = [r for r in records if isinstance(r.get("step"), int) and r["step"] <= step]
        keys = [(rule.question, rule.option, rule.threshold) for rule in self.stop_when]
        for record in records:
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            if record.get("kind") == "eval" and isinstance(payload.get("metrics"), dict):
                if self._history and self._history[-1]["step"] == record["step"]:  # several datasets, one step
                    self._history[-1].update(payload["metrics"])
                else:
                    self._history.append({"step": record["step"], **payload["metrics"]})
            elif record.get("kind") == "verdict" and isinstance(payload.get("rules"), list):
                for entry in payload["rules"]:
                    if not isinstance(entry, dict):
                        continue
                    key = (entry.get("question"), entry.get("option"), entry.get("threshold"))
                    if key in keys:
                        self._update_streak(keys.index(key), payload.get("for_step"), entry.get("probability"))
        self._history = self._history[-_HISTORY_LIMIT:]
        self._record("resume", step, {"from_step": step, "kept_records": len(records)})

    def on_step_begin(self, args, state, control, **kwargs):
        self._armed = False

    def on_step_end(self, args, state, control, **kwargs):
        self._armed = True

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if not self._training or not self._rank0:
            return
        self._grad_check = None
        try:
            step = int(state.global_step) + 1  # the step this optimizer update completes
            if step % self._every:
                return
            norms = self._gradient_norms(model if model is not None else self._model)
            bad = sorted(name for name, value in norms.items() if not math.isfinite(value))
            if bad:  # judged in `on_optimizer_step`, once it is known whether the step was taken
                self._grad_check = {"step": step, "groups": bad}
            elif norms:
                self._grad_norms = {name: math.sqrt(value) for name, value in norms.items()}
                self._grad_streak, self._grad_sentinel = 0, None
        except Exception as e:  # noqa: BLE001
            self._error("on_pre_optimizer_step", getattr(state, "global_step", 0), e)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self._training or not self._rank0 or self._grad_check is None:
            return
        check, self._grad_check = self._grad_check, None
        try:
            skipped = self._step_skipped(args)
            if skipped is not False:
                reason = (
                    "the optimizer skipped this step (loss-scale overflow)"
                    if skipped
                    else "a gradient scaler may be active and no trainer is attached to tell whether the step was taken"
                )
                self._record("note", check["step"], {"note": "nonfinite_gradients", **check, "reason": reason})
                return
            self._grad_streak += 1
            sentinel = {"sentinel": "nonfinite_gradients", **check, "consecutive": self._grad_streak}
            self._record("sentinel", check["step"], sentinel)
            if self._grad_streak >= self.patience_gradients and self._grad_sentinel is None:
                self._grad_sentinel = sentinel
        except Exception as e:  # noqa: BLE001
            self._error("on_optimizer_step", getattr(state, "global_step", 0), e)

    def _step_skipped(self, args) -> bool | None:
        """Whether the optimizer skipped the step; `None` when a gradient scaler may be active and nothing can tell."""
        accelerator = getattr(self._trainer, "accelerator", None)
        if accelerator is not None:
            return bool(accelerator.optimizer_step_was_skipped)
        return None if getattr(args, "fp16", False) else False

    def _parameter_groups(self, model) -> dict[str, list]:
        """Trainable parameters by group: `backbone` (the model's `base_model`) and `head` (everything else)."""
        if self._groups is None:
            unwrapped = getattr(model, "module", model)
            base = getattr(unwrapped, "base_model", None)
            trainable = [p for p in unwrapped.parameters() if p.requires_grad]
            if base is None or base is unwrapped or not isinstance(base, torch.nn.Module):
                self._groups = {"model": trainable}
            else:
                backbone = {id(p) for p in base.parameters()}
                self._groups = {
                    "backbone": [p for p in trainable if id(p) in backbone],
                    "head": [p for p in trainable if id(p) not in backbone],
                }
        return self._groups

    def _gradient_norms(self, model) -> dict[str, float]:
        """Squared gradient norm of each group: summed on device, then read with one `.item()` per group."""
        if model is None:
            return {}
        norms = {}
        for name, params in self._parameter_groups(model).items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if not grads:
                continue
            if hasattr(torch.nn.utils, "get_total_norm"):
                total = torch.nn.utils.get_total_norm(grads, 2.0, error_if_nonfinite=False)
            else:  # older torch
                device = grads[0].device
                total = torch.stack([torch.linalg.vector_norm(g.float()).to(device) for g in grads]).norm()
            norms[name] = total.float().square().item()
        return norms

    def _read_scales(self, model) -> dict[str, float]:
        """The model's zero-dimensional parameters named `...logit_scale`, read with one transfer."""
        if model is None:
            return {}
        unwrapped = getattr(model, "module", model)
        named = [(n, p) for n, p in unwrapped.named_parameters() if p.dim() == 0 and n.endswith("logit_scale")]
        if not named:
            return {}
        device = named[0][1].device
        values = torch.stack([p.detach().float().to(device) for _, p in named]).tolist()
        return {name: value for (name, _), value in zip(named, values, strict=True)}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self._training or not isinstance(metrics, dict):
            return
        prefix = self.metric_prefix + "_"
        if not any(isinstance(key, str) and key.startswith(prefix) for key in metrics):
            return
        if not self._armed:  # a training step began and never ended: `train()` failed, this evaluation is outside it
            self._training = False
            self.close()
            return
        step = int(state.global_step)
        proposal = None
        if self._rank0:
            try:
                proposal = self._observe(args, state, metrics, step, kwargs.get("model"))
            except Exception as e:  # noqa: BLE001
                self._error("on_evaluate", step, e)
        if self.mode == "enforce":
            self._decide(args, control, step, proposal)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._training:
            return
        try:
            if self._rank0 and self._ask is not None:
                idle = self.wait(self.finish_timeout)
                with self._cond:
                    received = self._pending_step in self._results
                self._record(
                    "end",
                    int(state.global_step),
                    {"last_eval_step": self._pending_step, "last_answer_received": received, "worker_idle": idle},
                )
            elif self._rank0:
                self._record("end", int(state.global_step), {"last_eval_step": None})
        except Exception as e:  # noqa: BLE001
            self._error("on_train_end", getattr(state, "global_step", 0), e)
        finally:
            self._training = False
            self.close()

    # -- evaluation (main process) -----------------------------------------------------------------------------

    def _observe(self, args, state, metrics: dict, step: int, model) -> dict | None:
        """Apply the previous verdict, check the sentinels, send this evaluation's digest; return a stop proposal."""
        same_step = self._last_eval_step == step
        self._last_eval_step = step
        if not same_step:
            self._apply_previous(step)
        prefix = self.metric_prefix + "_"

        nonfinite = sorted(k for k, v in metrics.items() if k.startswith(prefix) and _is_nonfinite_number(v))
        if nonfinite:
            self._record("sentinel", step, {"sentinel": "nonfinite_metrics", "metrics": nonfinite})
        scales = self._read_scales(model if model is not None else self._model)
        high = {name for name, value in scales.items() if not value <= SCALE_LIMIT}
        for name in sorted(high - self._high_scales):
            self._record(
                "sentinel",
                step,
                {"sentinel": "scale_above_limit", "name": name, "value": scales[name], "limit": SCALE_LIMIT},
            )
        self._high_scales = high

        entry = {}
        for key, value in metrics.items():
            name = key[len(prefix) :] if key.startswith(prefix) else None
            if name and any(name == metric or name.endswith("_" + metric) for metric in self.metrics):
                entry[name] = value
        entry = json_safe(entry)
        if same_step and self._history and self._history[-1].get("step") == step:
            self._history[-1] = {**self._history[-1], **entry}
        else:
            self._history = (self._history + [{"step": step, **entry}])[-_HISTORY_LIMIT:]
        digest = build_digest(
            self._history,
            step=step,
            max_steps=getattr(state, "max_steps", 0) or 0,
            objective=self.objective,
            scales={**scales, **{f"grad_norm_{group}": value for group, value in self._grad_norms.items()}},
            sentinels={
                "nonfinite_metrics": bool(nonfinite),
                "nonfinite_gradients": self._grad_streak > 0,
                "scale_above_limit": bool(high),
            },
            control=self._control_rows(step),
            budget_bytes=self.budget_bytes,
        )
        self._record("eval", step, {"metrics": entry})

        if self._ask is not None:
            with self._cond:
                if same_step:
                    # Send the merged digest if this step's request has not started; else it covers the first part
                    if self._job is not None and self._job[1] == step:
                        self._job = (self._generation, step, digest)
                else:
                    if self._job is not None:  # an older request never started: it was already recorded missing
                        self._discarded.discard(self._job[1])
                    self._job = (self._generation, step, digest)
                self._cond.notify_all()
            self._pending_step = step

        if self._stopped:
            return None
        proposal = None
        if nonfinite:
            proposal = {"reason": "sentinel", "sentinel": {"sentinel": "nonfinite_metrics", "metrics": nonfinite}}
        elif self._grad_sentinel is not None:
            proposal = {"reason": "sentinel", "sentinel": self._grad_sentinel}
        else:
            for rule, streak in zip(self.stop_when, self._streaks, strict=True):
                if len(streak) >= self.patience:
                    proposal = {"reason": "rule", "rule": asdict(rule), "verdicts": streak[-self.patience :]}
                    break
        if proposal is None:
            return None
        proposal["step"] = step
        if self.mode != "enforce":
            self._record("advice", step, proposal)
            return None
        proposal["checkpoint"] = str(Path(args.output_dir) / f"monitor-stop-{step}")
        return proposal

    def _apply_previous(self, step: int) -> None:
        previous, self._pending_step = self._pending_step, None
        if previous is None:
            return
        with self._cond:
            answers = self._results.pop(previous, _NOT_READY)
            if answers is _NOT_READY:
                self._discarded.add(previous)
        if answers is _NOT_READY:
            self._record("missing", step, {"for_step": previous, "reason": "no answer by the next evaluation"})
            return
        if answers is None:
            self._record("missing", step, {"for_step": previous, "reason": "the backend gave no answer"})
            return
        rules = []
        for index, rule in enumerate(self.stop_when):
            probability = self._rule_probability(rule, answers)
            status = self._update_streak(index, previous, probability)
            rules.append({**asdict(rule), "probability": probability, "status": status})
            rules[-1]["streak"] = len(self._streaks[index])
            if status == "missing":
                self._record(
                    "missing",
                    step,
                    {
                        "for_step": previous,
                        "rule": asdict(rule),
                        "reason": "the rule's answer is missing or malformed",
                    },
                )
        self._record("verdict", step, {"for_step": previous, "answers": answers, "rules": rules})

    def _rule_probability(self, rule: StopRule, answers: dict) -> float | None:
        answer = answers.get(rule.question)
        if not isinstance(answer, dict) or answer.get("type") != self.questions[rule.question]["type"]:
            return None
        if rule.option is None:
            return _probability(answer.get("noul"))
        probabilities = answer.get("probabilities")
        return _probability(probabilities.get(rule.option)) if isinstance(probabilities, dict) else None

    def _update_streak(self, index: int, for_step: Any, probability: Any) -> str:
        probability = _probability(probability)
        if probability is None:
            return "missing"
        if probability >= self.stop_when[index].threshold:
            self._streaks[index].append({"for_step": for_step, "probability": probability})
            return "agree"
        self._streaks[index].clear()
        return "disagree"

    def _control_rows(self, step: int) -> list[dict] | None:
        if self.control_ledger is None:
            return None
        problem = None
        try:
            records = Ledger(self.control_ledger).read()
            rows = [
                {"step": r["step"], **r["payload"]["metrics"]}
                for r in records
                if r.get("kind") == "eval"
                and isinstance(r.get("step"), int)
                and r["step"] <= step
                and isinstance(r.get("payload"), dict)
                and isinstance(r["payload"].get("metrics"), dict)
            ]
            if not records:
                problem = "the control ledger is missing or empty"
        except Exception as e:  # noqa: BLE001
            rows, problem = [], f"cannot read the control ledger: {type(e).__name__}: {e}"
        if problem is not None and not self._control_warned:
            self._control_warned = True
            self._record("warning", step, {"warning": problem[:500], "path": str(self.control_ledger)})
        return rows or None

    # -- decision (every process) ------------------------------------------------------------------------------

    def _decide(self, args, control, step: int, proposal: dict | None) -> None:
        """Every process: receive rank 0's decision, save, and set `should_training_stop` identically."""
        try:
            decision = self._broadcast(proposal if self._rank0 else None)
        except Exception as e:  # noqa: BLE001
            self._error("broadcast", step, e)
            return
        if not isinstance(decision, dict) or self._stopped:
            return
        checkpoint = decision.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            checkpoint = str(Path(args.output_dir) / f"monitor-stop-{step}")
        saved, error = self._save(checkpoint)
        if saved and self._rank0 and not _has_weights(checkpoint):
            saved, error = False, f"save_model wrote no weights file to {checkpoint}"
        try:
            saved = self._broadcast(saved if self._rank0 else None)
        except Exception as e:  # noqa: BLE001
            self._error("broadcast", step, e)
            return
        if saved is not True:
            self._record(
                "error",
                step,
                {"where": "save", "error": error or "the checkpoint was not saved", "checkpoint": checkpoint},
            )
            return
        record = {**decision, "step": step, "checkpoint": checkpoint, "mode": self.mode}
        if self._rank0 and self.ledger is not None:
            try:
                directory = self.ledger.path.parent
                directory.mkdir(parents=True, exist_ok=True)
                tmp = directory / "intervened.json.tmp"
                tmp.write_text(json.dumps(json_safe(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")
                os.replace(tmp, directory / "intervened.json")
            except Exception as e:  # noqa: BLE001 - the checkpoint exists: the stop stands
                self._error("intervened.json", step, e)
            self._record("intervened", step, record)
        self._stopped = True
        control.should_training_stop = True

    def _save(self, checkpoint: str) -> tuple[bool, str | None]:
        """
        Save with the attached trainer. Every process calls it: `save_model` gathers sharded weights collectively and
        writes from the main process only. `_internal_call` keeps it from pushing to the Hub.
        """
        try:
            save_model = self._trainer.save_model
            try:
                internal = "_internal_call" in inspect.signature(save_model).parameters
            except (TypeError, ValueError):
                internal = False
            if internal:
                save_model(checkpoint, _internal_call=True)
            else:
                save_model(checkpoint)
            return True, None
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"[:500]
