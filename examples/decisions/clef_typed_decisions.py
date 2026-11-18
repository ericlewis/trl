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
Train a Clef decision model on typed decisions: one prompt per record, every question answered jointly.

`model_name_or_path` is a Clef checkpoint, loaded as it is, or a backbone for a new head: a causal LM (KLM-1.6, Qwen)
reads its last token with the ChatML prompt; a bidirectional encoder (BERT, ModernBERT, or a reranker, unwrapped to
its base model) reads its first token with the plain prompt between [CLS] and [SEP]. `global_pooling` and
`prompt_style` override that choice; `head_type: cosine` (with `cosine_width`, `cosine_initial_scale`) selects the
small cosine head instead of the joint one, and is what the example's configuration trains. The resolved setup is
printed at start-up. `rlcd_weight` (with
`rlcd_sigma`, `rlcd_samples`, `rlcd_normalize_advantage`) adds Clef's RLCD policy term to the training loss. RLCD is
meant as a second phase after cross-entropy training: start this script again from the saved checkpoint with
`rlcd_weight` set, or in one run give `rlcd_start_epoch` and/or `rlcd_sigmas` (one noise scale per epoch of the RLCD
phase), which keep the weight at 0 until that epoch.

Backbone weights are never newly initialised and remote code is never run. A checkpoint whose weights sit under a
prefix (a remote-code package such as an UltraBERT reranker, with `backbone.<name>` keys) has its backbone read from
its safetensors under the one prefix that holds all of them, and its other tensors dropped; the start-up line names
where the weights came from. A backbone with weights that no single prefix provides is refused.

python examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml

torchrun --nproc_per_node=2 \
    examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
"""

import json
import os
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Features, Sequence, Value, load_dataset
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForDecision,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.decision import Question
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES
from transformers.utils import cached_file

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.decision import (
    DataCollatorForDecision,
    DecisionConfig,
    DecisionTrainer,
    RlcdScheduleCallback,
    compute_calibration_metrics,
)


def clef_decision_config(prompt_style: str = "chatml"):
    """
    Clef's decision config with its ChatML prompt (or its plain one, `prompt_style="plain"`): any number of options,
    since the head reads no label token.
    """
    from transformers import ClefConfig

    return ClefConfig.prompt_decision_config(prompt_style)


def common_prefix_length(first: list, second: list) -> int:
    """The number of leading items `first` and `second` have in common."""
    return next(
        (index for index, (a, b) in enumerate(zip(first, second, strict=False)) if a != b),
        min(len(first), len(second)),
    )


def decoded_state(state: Any) -> Any:
    """`state` decoded when it is the JSON string of an object or an array, else `state` as it is."""
    if isinstance(state, str):
        try:
            value = json.loads(state)
        except ValueError:
            return state
        if isinstance(value, (dict, list)):
            return value
    return state


class SchemaTooLongError(ValueError):
    """A prompt is longer than `max_length` even without its state, so cutting the state cannot make it fit."""


@dataclass
class DataCollatorForClef(DataCollatorForDecision):
    """
    Batch records for Clef: one prompt per record, with all of its questions answered jointly.

    Each example contains a `request` (a dictionary or JSON string), or `state` and `questions`, plus the targets of
    all questions. A `state` that is the JSON string of an object or an array (as datasets store it) is decoded, so that
    the prompt renders the record; any other string is kept as it is. Targets are `targets` (a mapping of question
    name to `target_index`, `target_probs`, `p_true` or `score`), or `gold` / `answers` (a mapping of question name to
    a gold label or probabilities). Positional targets (`target_index`, `target_probs`) follow the template's option
    order, which sorts choice options by key (`target_index: 0` of criteria `{"zeta", "alpha"}` is `alpha`); `gold`
    names the option instead. Nullable questions are not supported, since Clef has no coverage output.

    For repeated training or evaluation, preprocess once with `dataset.map(collator.tokenize)`. The collator also
    accepts these prepared examples and only pads them when building batches. A batch holds the questions of all its
    records one after the other: `labels` and `option_mask` have one row per question, in record order then question
    order, `hard_labels` holds the gold label of each question (the option its `label` or `target_index` names, else
    its target's first most probable option, which accuracy is counted against), and `rows_per_example` holds the
    number of questions of each record. With it, `DecisionTrainer` gathers
    whole records across processes, and its evaluation predictions have `max_questions` slots per record (the most
    questions of a record in the batch, over all processes), with the unused slots at `-100`.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer used by the model. Padding is applied on the right, with the pad token or else the EOS token.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model. The template must render all questions in one prompt.
        max_length (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens of a prompt.
        truncation (`str`, *optional*, defaults to `"error"`):
            What to do with a record whose prompt is longer than `max_length`, or whose request or targets are
            invalid:

            - `"error"`: raise.
            - `"skip"`: mark it as skipped under a reason that does not depend on the record.
            - `"state"`: as Clef does, cut the end of the state of an over-length prompt so that it is exactly
              `max_length` tokens, keeping everything else of the prompt (its questions and options) and the targets;
              the record is marked `truncated`. A record whose prompt is too long even without its state is skipped
              (`"schema longer than max_length (N tokens)"`), as are invalid records.

            Only the state is ever cut, since cutting anything else would remove options or questions.
        permute_fields (`bool`, *optional*, defaults to `False`):
            Keep the request of each record and re-render it with its questions in a new random order every time it
            is batched. Only records tokenized by a collator with `permute_fields` are permuted, so evaluation uses
            a separate collator without it. A re-render that fails, or that no longer matches the targets, raises; so
            does an over-length one, unless `truncation="state"` cuts its state.
        seed (`int`, *optional*, defaults to `3407`):
            Seed of the permutations, combined with the seed of the dataloader worker and the process rank.
    """

    max_length: int = 2048
    truncation: str = "error"
    permute_fields: bool = False
    seed: int = 3407
    _generator: random.Random | None = field(default=None, init=False, repr=False)
    _generator_key: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.truncation not in {"error", "skip", "state"}:
            raise ValueError("truncation must be error, skip or state")

    @staticmethod
    def _request(example: dict[str, Any]) -> dict[str, Any]:
        """
        The request of `example`, with its `questions` decoded and its `state` decoded when it is the JSON string of an
        object or an array (as datasets store it), so that the template renders the record, as the CLM example does. A
        state that is any other string is kept as it is.
        """
        request = example["request"] if "request" in example else {k: example[k] for k in ("state", "questions")}
        request = json.loads(request) if isinstance(request, str) else dict(request)
        if isinstance(request["questions"], str):
            request["questions"] = json.loads(request["questions"])
        request["state"] = decoded_state(request["state"])
        return request

    def _features(self, input_ids, marks, questions, labels, hard_labels):
        return {
            "input_ids": input_ids,
            "marks": marks,
            "num_options": [len(question.options) for question in questions],
            "question_type_ids": [Question.questions_types.index(question.type) for question in questions],
            "labels": labels,
            "hard_labels": hard_labels,
            "is_choice": [question.type == "choice" for question in questions],
        }

    def _skip(self, reason: str, error: Exception) -> dict[str, Any]:
        """
        Raise `error`, or with `truncation="skip"` or `"state"` return a skipped record with the empty lists and
        `reason`.
        """
        if self.truncation == "error":
            raise error
        keys = ("input_ids", "marks", "num_options", "question_type_ids", "labels", "hard_labels", "is_choice")
        empty = {key: [] for key in keys}
        skipped = {**empty, "skip": True, "reason": reason, "truncated": False}
        return {**skipped, **({"request": ""} if self.permute_fields else {})}

    def _render(self, request: dict[str, Any]) -> tuple[list[int], list[int], list]:
        """The `input_ids`, `marks` and questions of the one prompt of `request`."""
        prompts = self.decision_config.prepare_inputs(request, self.tokenizer)
        if len(prompts) != 1:
            raise ValueError(f"The template renders {len(prompts)} prompts; Clef needs all questions in one")
        inputs, questions = prompts[0]
        return inputs["input_ids"][0].tolist(), inputs["marks"].tolist(), questions

    def _cut_state(self, request, input_ids, marks) -> tuple[list[int], list[int]]:
        """
        Cut the end of the state of the over-length prompt `input_ids` of `request` so that it is exactly `max_length`
        tokens, and move `marks` with it. Raises a `SchemaTooLongError` when the prompt is too long without its state.

        The template renders the state between `<|split|>` sentinels, so the tokens before and after it do not depend
        on it: the request rendered with an empty state is exactly those tokens, and the prompt must start and end with
        them (with its marks after the state), or this raises rather than cutting by guess. The state starts at the one
        split of the prompt into that beginning, the state and that end. When several splits fit (the state begins
        with the tokens that follow it, or ends with those before it), two renderings whose states differ in their
        first token tell where it starts.
        """
        empty, empty_marks, _ = self._render({**request, "state": ""})
        if len(empty) > self.max_length:
            raise SchemaTooLongError(
                f"A record is longer than max_length even without its state ({len(empty)} > {self.max_length} "
                "tokens); increase max_length"
            )
        state_length = len(input_ids) - len(empty)
        # The starts of the state for which the prompt is the empty rendering with `state_length` tokens inserted
        latest = common_prefix_length(input_ids, empty)
        earliest = len(empty) - common_prefix_length(input_ids[::-1], empty[::-1])
        start = earliest
        if earliest < latest:
            first, _, _ = self._render({**request, "state": "a"})
            second, _, _ = self._render({**request, "state": "b"})
            start = common_prefix_length(first, second)
        if (
            state_length <= 0
            or not earliest <= start <= latest
            or [mark - state_length for mark in marks] != empty_marks
            or any(mark < start for mark in empty_marks)
        ):
            raise ValueError(
                "The template does not render the state on its own between <|split|> sentinels, so the state of an "
                "over-length prompt cannot be cut; use truncation='error' or 'skip'"
            )
        end, overflow = start + state_length, len(input_ids) - self.max_length
        return input_ids[: end - overflow] + input_ids[end:], [mark - overflow for mark in marks]

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render and tokenize one record with the native Transformers template.

        Returns `input_ids`, `marks`, `num_options`, `question_type_ids`, `labels` (the target distribution of each
        question over its options), `is_choice`, `skip`, `reason` and `truncated` (whether the state was cut), plus the
        `request` when `permute_fields`. A record that is too long, whose request is invalid, that asks a nullable
        question or whose targets cannot be used raises, or with `truncation="skip"` returns `skip=True`, the empty
        lists and one of the reasons `"longer than max_length (N tokens)"`, `"invalid request"`, `"nullable
        question"` and `"unusable targets"`. With `truncation="state"`, an over-length record has the end of its state
        cut instead, and one that is too long without its state is skipped with `"schema longer than max_length (N
        tokens)"`.
        """
        try:
            request = self._request(example)
        except (ValueError, KeyError, TypeError) as error:  # not JSON, or no state or questions
            return self._skip("invalid request", error)
        # Before validation, which rejects the `allow_null` field
        requested = request["questions"]
        for name, question in requested.items() if isinstance(requested, dict) else ():
            if isinstance(question, dict) and question.get("allow_null"):
                error = ValueError(
                    f"Question {name!r} is nullable; Clef has no coverage output, so it cannot be trained"
                )
                return self._skip("nullable question", error)
        # The request is rendered (and its questions validated) once, then the targets are read for its questions
        try:
            input_ids, marks, questions = self._render(request)
        except ValueError as error:
            return self._skip("invalid request", error)
        try:
            targets, hard_labels = self.prepare_labeled_targets(example, questions)
        except ValueError as error:
            return self._skip("unusable targets", error)
        for question in questions:
            if asdict(question).get("allow_null", False):
                error = ValueError(f"Question {question.name!r} is nullable; Clef has no coverage output")
                return self._skip("nullable question", error)
        if self.permute_fields and [question.name for question in questions] != list(request["questions"]):
            raise ValueError("permute_fields needs a template that renders the questions in the order of the request")
        truncated = len(input_ids) > self.max_length
        if truncated and self.truncation != "state":
            error = ValueError(
                f"A record is longer than max_length ({len(input_ids)} > {self.max_length} tokens); increase "
                "max_length or set truncation='state' or 'skip'"
            )
            return self._skip(f"longer than max_length ({self.max_length} tokens)", error)
        if truncated:
            try:
                input_ids, marks = self._cut_state(request, input_ids, marks)
            except SchemaTooLongError as error:
                return self._skip(f"schema longer than max_length ({self.max_length} tokens)", error)
        # The null column is dropped: Clef has no null output
        features = self._features(input_ids, marks, questions, [list(target[1:]) for target in targets], hard_labels)
        features |= {"skip": False, "reason": "", "truncated": truncated}
        if self.permute_fields:
            features["request"] = json.dumps(request)
        return features

    def _shuffler(self) -> random.Random:
        info = torch.utils.data.get_worker_info()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.environ.get("RANK", 0))
        key = (rank, None if info is None else info.seed)
        if self._generator is None or self._generator_key != key:
            self._generator, self._generator_key = random.Random(f"{self.seed}/{key[0]}/{key[1]}"), key
        return self._generator

    def _permute(self, feature: dict[str, Any]) -> dict[str, Any]:
        """
        Re-render the request of a record with its questions in a new order, each keeping its target. When the drawn
        order is the original one (always so for a single question), the tokenized record is returned as it is: it is
        what the re-render would give.
        """
        request = json.loads(feature["request"])
        names = list(request["questions"])
        if len(names) != len(feature["labels"]):
            raise ValueError("The request and the labels of a record have different numbers of questions")
        order = list(range(len(names)))
        self._shuffler().shuffle(order)
        if order == sorted(order):
            return feature
        permuted = {**request, "questions": {names[i]: request["questions"][names[i]] for i in order}}
        try:
            input_ids, marks, questions = self._render(permuted)
        except Exception as error:
            raise ValueError(f"Re-rendering a request with permuted questions failed: {error}") from error
        if [question.name for question in questions] != [names[i] for i in order]:
            raise ValueError("The permuted prompt does not list the questions in the permuted order")
        labels = [feature["labels"][i] for i in order]
        hard_labels = [feature["hard_labels"][i] for i in order]
        if [len(question.options) for question in questions] != [len(row) for row in labels]:
            raise ValueError("The options of a permuted question no longer match its labels")
        if len(input_ids) > self.max_length:
            if self.truncation != "state":
                raise ValueError(f"A permuted prompt is longer than max_length ({self.max_length} tokens)")
            input_ids, marks = self._cut_state(permuted, input_ids, marks)
        return {**feature, **self._features(input_ids, marks, questions, labels, hard_labels)}

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" not in features[0]:
            features = [self.tokenize(example) for example in features]
        if any(feature.get("skip") for feature in features):
            raise ValueError("Skipped records must be filtered out before batching")
        if any("hard_labels" not in feature for feature in features):
            raise ValueError(
                "Records tokenized without `hard_labels` (the gold label of each question); tokenize again"
            )
        if self.permute_fields:
            features = [self._permute(feature) if feature.get("request") else feature for feature in features]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        def pad(rows, value, dtype=torch.long):
            width = max(len(row) for row in rows)
            return torch.tensor([list(row) + [value] * (width - len(row)) for row in rows], dtype=dtype)

        input_ids = pad([feature["input_ids"] for feature in features], pad_token_id)
        attention_mask = pad([[1] * len(feature["input_ids"]) for feature in features], 0)
        num_options = [option for feature in features for option in feature["num_options"]]
        max_options = max(num_options)
        option_mask = torch.arange(max_options)[None, :] < torch.tensor(num_options)[:, None]
        labels = pad([row for feature in features for row in feature["labels"]], 0.0, torch.float)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marks": pad([feature["marks"] for feature in features], -1),
            "num_options": pad([feature["num_options"] for feature in features], 0),
            "question_type_ids": pad([feature["question_type_ids"] for feature in features], 0),
            "labels": labels,
            "hard_labels": torch.tensor([label for feature in features for label in feature["hard_labels"]]),
            "option_mask": option_mask,
            "is_choice": torch.tensor([choice for feature in features for choice in feature["is_choice"]]),
            "rows_per_example": torch.tensor([len(feature["num_options"]) for feature in features]),
        }


@dataclass
class ClefScriptArguments(ScriptArguments):
    dataset_name: str = field(default="LocalLLaMA/typed-decisions", metadata={"help": "Typed decision dataset."})
    dataset_config: str | None = field(default="all", metadata={"help": "Dataset subset."})
    dataset_revision: str | None = field(default=None, metadata={"help": "Dataset revision."})
    dataset_test_split: str | None = field(
        default=None, metadata={"help": "Evaluation split. Defaults to a holdout from the training split."}
    )
    validation_fraction: float = field(
        default=0.05,
        metadata={
            "help": "Fraction of records held out when there is no evaluation split, drawn with eval_sample_seed."
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": "Limit training records to a random sample of this many rows, drawn with seed before "
            "tokenization (skipped records make the realised count smaller; the build report shows it)."
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": "Limit evaluation records to a random sample of this many rows, drawn with eval_sample_seed "
            "before tokenization (skipped records make the realised count smaller; the build report shows it)."
        },
    )
    eval_sample_seed: int = field(
        default=0,
        metadata={
            "help": "Seed of the evaluation holdout and of the max_eval_samples sample. It is not the run's seed, so "
            "runs with different seeds evaluate on the same records."
        },
    )

    mode: str = field(default="full", metadata={"help": "Train head, lora, or full model."})
    head_type: str | None = field(
        default=None,
        metadata={
            "help": "Decision head of a new model: joint (the model's default) or cosine (what the example's "
            "configuration sets). A Clef checkpoint keeps its own head."
        },
    )
    global_pooling: str | None = field(
        default=None,
        metadata={
            "help": "Token of a new model's global state: last or first. Defaults to the backbone's attention: last for "
            "causal backbones, first for bidirectional ones (encoders)."
        },
    )
    prompt_style: str | None = field(
        default=None,
        metadata={
            "help": "Template of a new model: chatml or plain. Defaults to chatml for causal backbones and plain "
            "(between [CLS] and [SEP]) for bidirectional ones."
        },
    )
    cosine_width: int | None = field(default=None, metadata={"help": "Width of the cosine head's projections."})
    cosine_initial_scale: float | None = field(
        default=None, metadata={"help": "Initial temperature of the cosine head."}
    )
    head_hidden_size: int | None = field(
        default=None, metadata={"help": "Width of the head. Defaults to the config's."}
    )
    head_intermediate_size: int | None = field(default=None, metadata={"help": "Feed-forward width of the head."})
    head_num_layers: int | None = field(default=None, metadata={"help": "Number of layers of the head."})
    head_num_evidence_layers: int | None = field(default=None, metadata={"help": "Layers in which options read."})
    head_num_attention_heads: int | None = field(default=None, metadata={"help": "Attention heads of the head."})
    label_smoothing: float | None = field(default=None, metadata={"help": "Loss label smoothing. Defaults to 0."})
    brier_weight: float | None = field(default=None, metadata={"help": "Weight of the Brier score. Defaults to 0."})
    ordinal_weight: float | None = field(default=None, metadata={"help": "Weight of the ordinal term. Defaults to 0."})
    rlcd_weight: float | None = field(
        default=None, metadata={"help": "Weight of the RLCD policy term, applied in training only. Defaults to 0."}
    )
    rlcd_sigma: float | None = field(
        default=None, metadata={"help": "Standard deviation of the RLCD noise on the logits. Defaults to 0.4."}
    )
    rlcd_samples: int | None = field(
        default=None, metadata={"help": "Noisy samples per question of the RLCD term. Defaults to 32."}
    )
    rlcd_normalize_advantage: bool | None = field(
        default=None,
        metadata={
            "help": "Divide the RLCD advantages by their standard deviation (Laya's original). Defaults to False."
        },
    )
    rlcd_start_epoch: float | None = field(
        default=None,
        metadata={
            "help": "Run cross-entropy only until this epoch, then switch RLCD on with rlcd_weight (which must be "
            "positive). The model is built with rlcd_weight 0."
        },
    )
    rlcd_sigmas: list[float] | None = field(
        default=None,
        metadata={
            "help": "One RLCD noise scale per epoch of the RLCD phase, counted from rlcd_start_epoch (0 if unset); the "
            "last is held for later epochs, e.g. [0.4, 0.3, 0.2, 0.1]. Needs a positive rlcd_weight."
        },
    )
    max_length: int = field(default=2048, metadata={"help": "Maximum prompt token length."})
    truncation: str = field(
        default="error",
        metadata={
            "help": "Over-length or invalid records: error; skip and count them; or state, which cuts the end of the "
            "state of an over-length record to fit (as Clef does) and skips the rest."
        },
    )
    permute_fields: bool = field(default=False, metadata={"help": "Shuffle the question order of training prompts."})


@dataclass
class ClefModelArguments(ModelConfig):
    model_name_or_path: str = field(
        default="MinimaLabs/KLM-1.6",
        metadata={"help": "Clef checkpoint, or a causal LM or an encoder to use as the backbone of a new head."},
    )
    lora_task_type: str = field(default="FEATURE_EXTRACTION", metadata={"help": "PEFT task type for the backbone."})


def is_causal(config) -> bool:
    """
    Whether a backbone attends causally: a model type with a causal language modeling class that is not a masked
    language model (BERT and RoBERTa have both and are encoders), or a config marked as a decoder.
    """
    model_type = config.model_type
    is_decoder_type = (
        model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and model_type not in MODEL_FOR_MASKED_LM_MAPPING_NAMES
    )
    return is_decoder_type or bool(getattr(config, "is_decoder", False))


@dataclass
class BackboneWeights:
    """Where the weights of a backbone came from: `direct` or `prefix "<p>"`, and the checkpoint tensors left out."""

    source: str
    dropped: list[str]

    def __str__(self):
        names = ", ".join(self.dropped[:10]) + (", ..." if len(self.dropped) > 10 else "")
        return f"{self.source} (dropped {len(self.dropped)} non-backbone tensors{': ' + names if names else ''})"


def without_remote_code(model_kwargs):
    """`model_kwargs` with `trust_remote_code=False`: this script never runs the code of a checkpoint."""
    if model_kwargs.get("trust_remote_code"):
        raise ValueError("This example never runs remote code; remove trust_remote_code")
    return {**model_kwargs, "trust_remote_code": False}


def loading_problems(info) -> list[str]:
    """The keys that `from_pretrained` initialised anew: missing from the checkpoint, or of another shape."""
    mismatched = [key[0] if isinstance(key, tuple) else key for key in info.get("mismatched_keys", ())]
    return sorted({*info.get("missing_keys", ()), *mismatched})


def checkpoint_shapes(name, revision=None) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The file and shape of every tensor of the safetensors weights of `name`, read from their headers only."""
    files = []
    single = cached_file(name, "model.safetensors", revision=revision, _raise_exceptions_for_missing_entries=False)
    if single is not None:
        files = [single]
    else:
        index = cached_file(
            name, "model.safetensors.index.json", revision=revision, _raise_exceptions_for_missing_entries=False
        )
        if index is not None:
            shards = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
            files = [cached_file(name, shard, revision=revision) for shard in shards]
    shapes = {}
    for file in files:
        with safe_open(file, framework="pt") as weights:
            for key in weights.keys():
                shapes[key] = (file, tuple(weights.get_slice(key).get_shape()))
    return shapes


def repair_backbone(name, backbone, missing, revision=None) -> BackboneWeights:
    """
    Load the weights of `backbone` from the safetensors of `name` under the single key prefix that holds every one of
    them with its shape (`backbone.`, `model.`, ...), strictly. Raises a `ValueError` naming the missing keys when no
    prefix holds them all, or when several do.
    """
    expected = {key: tuple(value.shape) for key, value in backbone.state_dict().items()}
    shapes = checkpoint_shapes(name, revision)
    reference = next(iter(expected))
    # Any prefix that holds every key holds the first: the candidates are the prefixes of that one
    candidates = sorted(
        key[: -len(reference)]
        for key in shapes
        if key.endswith("." + reference) and key[: -len(reference)] not in ("", ".")
    )
    # The weights each candidate lacks, or holds with another shape
    absent = {
        prefix: [key for key, shape in expected.items() if shapes.get(prefix + key, (None, None))[1] != shape]
        for prefix in candidates
    }
    resolved = [prefix for prefix in candidates if not absent[prefix]]
    if len(resolved) > 1:
        prefixes = ", ".join(f'"{prefix}"' for prefix in resolved)
        raise ValueError(
            f"The weights of {name} are ambiguous: the prefixes {prefixes} each hold a whole "
            f"{type(backbone).__name__}. Save the backbone on its own and train from that."
        )
    if not resolved:
        where = "under the names of its class"
        if candidates:  # the keys that the closest prefix lacks
            closest = min(candidates, key=lambda prefix: len(absent[prefix]))
            missing, where = absent[closest], f'under the closest prefix "{closest}"'
        listed = ", ".join(missing[:10]) + (", ..." if len(missing) > 10 else "")
        raise ValueError(
            f"{len(missing)} of the {len(expected)} weights of the {type(backbone).__name__} backbone are missing "
            f"from {name} {where} (or have another shape), and no single key prefix of its safetensors holds them "
            f"all: {listed}. Refusing to train from newly initialised weights."
        )
    (prefix,) = resolved
    state = {}
    for key in expected:
        file, _ = shapes[prefix + key]
        with safe_open(file, framework="pt") as weights:
            state[key] = weights.get_tensor(prefix + key)
    backbone.load_state_dict(state, strict=True)
    used = {prefix + key for key in expected}
    return BackboneWeights(f'prefix "{prefix}"', sorted(key for key in shapes if key not in used))


def check_per_layer_settings(name, config, revision=None):
    """
    Refuse a checkpoint whose `config.json` lists `attention_windows` or `rope_theta` per layer (as remote-code
    UltraBERT packages do) with values that differ within a kind of layer, or with a length other than the number of
    layers. Transformers' native config reads such lists as one window and one RoPE base per kind of layer (`null`
    windows are global layers), so it would build another architecture without a warning.
    """
    path = cached_file(name, "config.json", revision=revision, _raise_exceptions_for_missing_entries=False)
    if path is None:
        return
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lists = {key: raw[key] for key in ("attention_windows", "rope_theta") if isinstance(raw.get(key), list)}
    if not lists:
        return
    kinds = getattr(config, "layer_types", None)
    if kinds is None:  # the kinds the windows give, or else one kind
        windows = lists.get("attention_windows")
        kinds = ["full_attention" if window is None else "sliding_attention" for window in windows or []]
        kinds = kinds or ["all"] * len(lists["rope_theta"])
    for key, values in lists.items():
        if len(values) != len(kinds):
            raise ValueError(
                f"{name} sets {key} per layer for {len(values)} layers, but has {len(kinds)}; refusing to guess "
                "which value each layer takes"
            )
        for kind in dict.fromkeys(kinds):
            found = list(dict.fromkeys(value for value, other in zip(values, kinds, strict=False) if other == kind))
            if len(found) > 1:
                raise ValueError(
                    f"{name} sets {key} per layer with different values within its {kind} layers ({found}): "
                    "Transformers reads one value per kind of layer, so it would build another architecture"
                )
            if key == "attention_windows" and (found[0] is None) != (kind == "full_attention"):
                raise ValueError(
                    f"{name} sets {key} per layer with {found[0]} for its {kind} layers, which does not match "
                    "the kind of these layers"
                )


def load_backbone(name, config, model_kwargs):
    """
    The backbone of a new head, the table its `lm_head` starts from, and where its weights came from: the base model of
    a causal LM with its output embeddings, or a bidirectional encoder with its input embeddings. A checkpoint that only
    registers a sequence classification class (a reranker) is loaded as that and unwrapped to its `base_model`.

    No weight of the backbone is newly initialised: when the checkpoint does not provide one under the names of the
    loaded class, the backbone is read from its safetensors under the one key prefix that holds all of its weights
    (see `repair_backbone`), or a `ValueError` is raised. Remote code is never run.
    """
    model_kwargs = without_remote_code(model_kwargs)
    check_per_layer_settings(name, config, revision=model_kwargs.get("revision"))
    architectures = config.architectures or []
    if is_causal(config):
        loaded, info = AutoModelForCausalLM.from_pretrained(name, output_loading_info=True, **model_kwargs)
    elif architectures and all(architecture.endswith("ForSequenceClassification") for architecture in architectures):
        loaded, info = AutoModelForSequenceClassification.from_pretrained(
            name, output_loading_info=True, **model_kwargs
        )
        if loaded.base_model is loaded or not hasattr(loaded.base_model, "get_input_embeddings"):
            raise ValueError(
                f"{name} registers only {architectures}, whose base model cannot be unwrapped into a backbone"
            )
    else:
        loaded, info = AutoModel.from_pretrained(name, output_loading_info=True, **model_kwargs)
    backbone = loaded.base_model
    # The keys of the loaded class that belong to its backbone, and the others (a task head, an `lm_head`)
    wrapper = "" if backbone is loaded else f"{loaded.base_model_prefix}."
    problems = loading_problems(info)
    missing = [key[len(wrapper) :] for key in problems if key.startswith(wrapper)]
    head = [key for key in problems if not key.startswith(wrapper)]
    if is_causal(config) and head:  # the output embeddings the lexical prior starts from
        raise ValueError(f"{name} is missing weights of its causal LM outside its backbone: {', '.join(head[:10])}")
    if missing:
        weights = repair_backbone(name, backbone, missing, revision=model_kwargs.get("revision"))
    else:
        dropped = sorted(info.get("unexpected_keys", ()))
        if wrapper and not is_causal(config):  # the task head of the loaded class is not kept
            dropped += sorted(key for key in loaded.state_dict() if not key.startswith(wrapper) and key not in head)
        weights = BackboneWeights("direct", dropped)
    embeddings = loaded.get_output_embeddings().weight if is_causal(config) else backbone.get_input_embeddings().weight
    return backbone, embeddings, weights


LOSS_SETTINGS = (
    "label_smoothing",
    "brier_weight",
    "ordinal_weight",
    "rlcd_weight",
    "rlcd_sigma",
    "rlcd_samples",
    "rlcd_normalize_advantage",
)


# The `from_pretrained` arguments a new head honours: the backbone is loaded with them, then the model is built afresh
# around it with its dtype and attention implementation. Any other argument would be dropped, so it is refused.
# `attn_implementation` is the backbone's, on both paths: Clef's own layers keep their default.
NEW_HEAD_MODEL_KWARGS = frozenset({"revision", "dtype", "attn_implementation", "trust_remote_code"})


def rlcd_schedule(script_args) -> RlcdScheduleCallback | None:
    """
    The callback that runs RLCD from `rlcd_start_epoch` with the noise scales `rlcd_sigmas`, or `None` when neither is
    given. `rlcd_weight` is then the weight of the phase and must be positive.
    """
    if script_args.rlcd_start_epoch is None and script_args.rlcd_sigmas is None:
        return None
    if not script_args.rlcd_weight or script_args.rlcd_weight <= 0:
        raise ValueError("rlcd_start_epoch and rlcd_sigmas schedule RLCD, so rlcd_weight must be positive")
    if script_args.rlcd_sigmas is not None and script_args.rlcd_sigma is not None:
        raise ValueError("rlcd_sigmas sets the noise scale of every epoch of the RLCD phase; remove rlcd_sigma")
    return RlcdScheduleCallback(
        script_args.rlcd_weight,
        start_epoch=script_args.rlcd_start_epoch or 0.0,
        sigmas=script_args.rlcd_sigmas,
    )


def build_model(script_args, model_args, model_kwargs):
    """
    A Clef checkpoint is loaded as it is; any other model becomes the backbone of a new head. A causal backbone reads
    its last token with the ChatML prompt, a bidirectional one (an encoder) its first with the plain prompt, unless
    `global_pooling` or `prompt_style` say otherwise.
    """
    from transformers import ClefConfig, ClefForDecision

    name, revision = model_args.model_name_or_path, model_args.model_revision
    losses = {key: getattr(script_args, key) for key in LOSS_SETTINGS if getattr(script_args, key) is not None}
    if rlcd_schedule(script_args) is not None:  # the schedule turns RLCD on: the model starts as plain cross-entropy
        losses["rlcd_weight"] = 0.0
    head_settings = {
        key: getattr(script_args, key)
        for key in ("head_type", "global_pooling", "prompt_style", "cosine_width", "cosine_initial_scale")
        if getattr(script_args, key) is not None
    }
    # The sizes of the head, under the names of `ClefConfig`
    sizes = {
        key: getattr(script_args, f"head_{value}")
        for key, value in {
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "num_hidden_layers": "num_layers",
            "num_evidence_layers": "num_evidence_layers",
            "num_attention_heads": "num_attention_heads",
        }.items()
        if getattr(script_args, f"head_{value}") is not None
    }
    model_kwargs = without_remote_code(model_kwargs)
    config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=False)
    is_checkpoint = (
        config.model_type == "clef"
        or cached_file(name, "joint_head_config.json", revision=revision, _raise_exceptions_for_missing_entries=False)
        is not None
    )
    if is_checkpoint:
        if isinstance(model_kwargs.get("attn_implementation"), str):  # the backbone's; Clef's head keeps its own
            model_kwargs = {
                **model_kwargs,
                "attn_implementation": {"text_config": model_kwargs["attn_implementation"]},
            }
        model, info = AutoModelForDecision.from_pretrained(name, output_loading_info=True, **model_kwargs)
        problems = loading_problems(info)
        if problems:
            raise ValueError(
                f"{len(problems)} weights of the Clef checkpoint {name} are missing or of another shape: "
                f"{', '.join(problems[:10])}{', ...' if len(problems) > 10 else ''}"
            )
        changed = {
            key: value for key, value in {**head_settings, **sizes}.items() if getattr(model.config, key) != value
        }
        if changed:
            raise ValueError(
                f"{name} is a Clef checkpoint; its head settings and sizes cannot change ({changed}). Remove them, or "
                "train a new head from its backbone."
            )
        check_cosine_settings(script_args, model.config.head_type)
        # `setattr` skips the checks of a new config: the loss settings it ends up with are checked by building one
        loss_keys = [key for key in LOSS_SETTINGS if hasattr(model.config, key)]
        ClefConfig(**{**{key: getattr(model.config, key) for key in loss_keys}, **losses})
        for key, value in losses.items():
            setattr(model.config, key, value)
        model.backbone_weights = "Clef checkpoint"
    else:
        unsupported = sorted(set(model_kwargs) - NEW_HEAD_MODEL_KWARGS)
        if unsupported:
            raise ValueError(
                f"model_init_kwargs {unsupported} cannot be honoured for a new head on {name}: its model is built "
                f"afresh around the loaded backbone, which keeps only {sorted(NEW_HEAD_MODEL_KWARGS)}. Remove them, "
                "or train from a Clef checkpoint, which is loaded with all of them."
            )
        backbone, embeddings, weights = load_backbone(name, config, model_kwargs)
        check_cosine_settings(script_args, head_settings.get("head_type", "joint"))
        if (losses.get("rlcd_weight") or 0) > 0:
            warnings.warn(
                f"rlcd_weight={losses['rlcd_weight']} on a new head with no schedule runs RLCD from the first step. "
                "RLCD is meant as a phase after cross-entropy training: train with cross-entropy first and start again "
                "from the saved checkpoint, or set rlcd_start_epoch.",
                stacklevel=2,
            )
        text_config = backbone.config
        # The attention the backbone was loaded with (the one asked for, or the default it resolved to): building a
        # config around `text_config` resets it
        attention = text_config._attn_implementation
        causal = is_causal(text_config)
        defaults = {"global_pooling": "last" if causal else "first", "prompt_style": "chatml" if causal else "plain"}
        config = ClefConfig(
            text_config=text_config,
            backbone_hidden_size=text_config.hidden_size,
            **sizes,
            **losses,
            **{**defaults, **head_settings},
        )
        # Set on the backbone only: Clef's own layers keep their default (they do not support every kernel)
        model = ClefForDecision._from_config(config, attn_implementation={"text_config": attention})
        model.model.load_state_dict(backbone.state_dict())
        # The head stays in float32; the backbone keeps the dtype it was loaded in
        model.model.to(backbone.dtype)
        if hasattr(model, "lm_head"):  # the cosine head, and encoders without a causal LM class, have none
            with torch.no_grad():
                model.lm_head.weight.copy_(embeddings)
            model.lm_head.to(backbone.dtype)
        model.backbone_weights = weights
        del backbone, embeddings
    if model.decision_config.chat_template is None:  # the template of the model's own prompt style
        model.decision_config = clef_decision_config(model.config.prompt_style)
    return model


def check_cosine_settings(script_args, head_type):
    """Refuse `cosine_width` and `cosine_initial_scale` for a head that is not the cosine head, which ignores them."""
    given = [key for key in ("cosine_width", "cosine_initial_scale") if getattr(script_args, key) is not None]
    if given and head_type != "cosine":
        raise ValueError(f"{given} only apply to head_type: cosine, and this head is {head_type}; remove them")


def position_limit(backbone) -> int | None:
    """
    The longest sequence a backbone with a learned table of absolute positions (BERT, RoBERTa) can read, or `None`
    when it has none (rotary or relative positions).
    """
    for name, module in backbone.named_modules():
        if name.split(".")[-1] == "position_embeddings" and isinstance(module, torch.nn.Embedding):
            # RoBERTa-style tables count positions from after the padding index
            offset = 0 if module.padding_idx is None else module.padding_idx + 1
            return module.num_embeddings - offset
    return None


def describe(model) -> str:
    """
    The resolved setup of a Clef model: the attention of its backbone, its pooling, prompt style and head, and where
    `build_model` read the weights of its backbone from.
    """
    config = model.config
    attention = "causal" if is_causal(config.text_config) else "bidirectional"
    weights = getattr(model, "backbone_weights", None)
    return (
        f"Clef setup: {attention} backbone ({config.text_config.model_type}), global_pooling={config.global_pooling}, "
        f"prompt_style={config.prompt_style}, head_type={config.head_type}"
        + ("" if weights is None else f", backbone weights: {weights}")
    )


def sample_rows(dataset, count, seed):
    """
    `count` rows of `dataset` drawn at random with `seed`, the same for the same seed; the whole dataset when `count`
    is `None` or not smaller. The first rows of a split are not a fair sample of it: datasets are often sorted.
    """
    if count is None or count >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(count))


def prepare_split(dataset, collator, training_args, desc):
    """Tokenize once, drop the skipped records, and report what was built."""
    features = {
        "input_ids": Sequence(Value("int32")),
        "marks": Sequence(Value("int32")),
        "num_options": Sequence(Value("int32")),
        "question_type_ids": Sequence(Value("int32")),
        "labels": Sequence(Sequence(Value("float32"))),
        "hard_labels": Sequence(Value("int32")),
        "is_choice": Sequence(Value("bool")),
        "skip": Value("bool"),
        "reason": Value("string"),
        "truncated": Value("bool"),
    }
    if collator.permute_fields:
        features["request"] = Value("string")
    with training_args.main_process_first(desc=f"Tokenizing {desc} dataset"):
        tokenized = dataset.map(
            collator.tokenize,
            remove_columns=dataset.column_names,
            features=Features(features),
            desc=f"Tokenizing {desc} dataset",
        )
        reasons = Counter(reason for reason in tokenized["reason"] if reason)
        truncated = sum(tokenized["truncated"])
        tokenized = tokenized.filter(lambda skip: not skip, input_columns="skip")
        tokenized = tokenized.remove_columns(["skip", "reason", "truncated"])
    report = {
        "rows": len(dataset),
        "questions": sum(len(counts) for counts in tokenized["num_options"]),
        "skipped": dict(reasons),
        # The records whose state was cut to fit (truncation="state"); nothing else of a prompt is ever cut
        "truncated": truncated,
    }
    if training_args.process_index == 0:
        print(f"Build report ({desc}): {json.dumps(report)}")
    return tokenized, report


@dataclass
class ClefRun:
    """What `build_trainer` builds: the trainer, the LoRA backbone of `mode: lora` (else `None`), the build reports."""

    trainer: DecisionTrainer
    lora_backbone: Any
    reports: dict[str, dict]


def build_trainer(script_args, training_args, model_args) -> ClefRun:
    """Everything the script does before training: the model, the tokenized splits and the trainer."""
    set_seed(training_args.seed)
    schedule = rlcd_schedule(script_args)
    if script_args.mode not in {"head", "lora", "full"}:
        raise ValueError("mode must be head, lora, or full")
    # Batches hold whole records (rows_per_example), so they are never dispatched or split across processes
    config = training_args.accelerator_config
    if config.dispatch_batches or config.split_batches:
        raise ValueError(
            "The Clef example batches whole records, which dispatch_batches and split_batches would break apart; set "
            "both to false in accelerator_config"
        )
    config.dispatch_batches = config.split_batches = False
    model_kwargs = dict(training_args.model_init_kwargs or {})
    model_kwargs.setdefault("revision", model_args.model_revision)
    model_kwargs.setdefault("dtype", model_args.dtype)
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = build_model(script_args, model_args, model_kwargs)
    limit = position_limit(model.model)
    if limit is not None and script_args.max_length > limit:
        raise ValueError(
            f"max_length is {script_args.max_length} tokens, but the {model.config.text_config.model_type} backbone "
            f"reads at most {limit} positions; set max_length to {limit} or less"
        )
    if training_args.process_index == 0:
        print(describe(model))
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.mode == "full":
        model.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
        if hasattr(model, "lm_head"):  # the cosine head and encoders have none
            model.lm_head.requires_grad_(False)
    lora_backbone = None
    if script_args.mode == "lora":
        peft_config = get_peft_config(model_args)
        if peft_config is None:
            raise ValueError("mode='lora' requires use_peft=True")
        from peft import get_peft_model

        # Adapters go on the backbone only. The head is trained in full next to them as ordinary parameters: PEFT's
        # `modules_to_save` matches names by suffix, so it cannot cleanly wrap the head's `scorer` (and its
        # `residual_scorer`) and the resulting adapter does not reload.
        model.model = lora_backbone = get_peft_model(model.model, peft_config)

    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, revision=script_args.dataset_revision
    )
    train_dataset = dataset[script_args.dataset_train_split]
    if script_args.dataset_test_split is None:
        # The holdout and its sample do not depend on the run's seed: runs with different seeds evaluate alike
        split = train_dataset.train_test_split(
            test_size=script_args.validation_fraction, seed=script_args.eval_sample_seed
        )
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        eval_dataset = dataset[script_args.dataset_test_split]
    train_dataset = sample_rows(train_dataset, script_args.max_train_samples, training_args.seed)
    eval_dataset = sample_rows(eval_dataset, script_args.max_eval_samples, script_args.eval_sample_seed)

    # Training batches permute the questions; evaluation never does: it has its own collator, for tokenizing and for
    # batching
    collator_kwargs = {"max_length": script_args.max_length, "truncation": script_args.truncation}
    collator = DataCollatorForClef(
        tokenizer,
        model.decision_config,
        permute_fields=script_args.permute_fields,
        seed=training_args.seed,
        **collator_kwargs,
    )
    eval_collator = DataCollatorForClef(tokenizer, model.decision_config, **collator_kwargs)
    train_dataset, train_report = prepare_split(train_dataset, collator, training_args, "train")
    eval_dataset, eval_report = prepare_split(eval_dataset, eval_collator, training_args, "eval")
    # Say so here when nothing is left, rather than fail later in the sampler of an empty dataset
    splits = [("training", train_dataset, train_report)]
    if training_args.eval_strategy != "no":
        splits.append(("evaluation", eval_dataset, eval_report))
    for name, split, report in splits:
        if len(split) == 0:
            raise ValueError(
                f"No {name} record is left: all {report['rows']} were skipped ({report['skipped']}). Raise max_length "
                f"(now {script_args.max_length}) if the prompts are longer than it, or check the rows."
            )

    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        eval_data_collator=eval_collator,
        compute_metrics=compute_calibration_metrics,
    )
    if schedule is not None:
        trainer.add_callback(schedule)
    return ClefRun(trainer, lora_backbone, {"train": train_report, "eval": eval_report})


def main(script_args, training_args, model_args):
    run = build_trainer(script_args, training_args, model_args)
    trainer, lora_backbone, model = run.trainer, run.lora_backbone, run.trainer.model
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if lora_backbone is not None:
        # The adapter alone, for the original backbone, then merged into the backbone: the output is a native Clef
        # checkpoint that `AutoModelForDecision` and the decision pipeline load as they are
        if training_args.should_save:
            lora_backbone.save_pretrained(Path(training_args.output_dir) / "adapter")
        model.model = lora_backbone.merge_and_unload()
    trainer.save_model(training_args.output_dir)
    trainer.save_metrics("train", result.metrics)
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    trainer.end()
    return trainer


if __name__ == "__main__":
    parser = TrlParser((ClefScriptArguments, DecisionConfig, ClefModelArguments))
    main(*parser.parse_args_and_config())
