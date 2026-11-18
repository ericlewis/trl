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

`model_name_or_path` is a Clef checkpoint, loaded as it is, or the backbone of a new head (a causal LM, an encoder or a
reranker package), loaded with no weight newly initialised and with its own code only with `trust_remote_code`. RLCD
(`rlcd_weight`) is a second phase after cross-entropy training. The README describes the main settings; `--help` lists
them all.

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
from transformers import AutoConfig, AutoModelForDecision, AutoTokenizer, set_seed
from transformers.decision import Question
from transformers.utils import cached_file

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.decision import (
    DataCollatorForDecision,
    DecisionConfig,
    DecisionTrainer,
    RlcdScheduleCallback,
    compute_calibration_metrics,
    is_causal,
    load_decision_backbone,
    needs_remote_code,
)
from trl.experimental.decision.backbone import check_max_length, loading_problems, raw_config, remote_code_setting
from trl.experimental.decision.data_collator import SchemaTooLongError, cut_state, decode_request, skipped_record


def clef_decision_config(prompt_style: str = "chatml"):
    """Clef's decision config with its ChatML (or `"plain"`) prompt; it takes any number of options."""
    from transformers import ClefConfig

    return ClefConfig.prompt_decision_config(prompt_style)


@dataclass
class DataCollatorForClef(DataCollatorForDecision):
    """
    Batch records for Clef: one prompt per record, with all of its questions answered jointly.

    Each example has a `request` (or `state` and `questions`) and the targets of all of its questions (`targets`, `gold`
    or `answers`). A batch holds the questions of its records one after the other, with `rows_per_example`.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer used by the model. Padding is applied on the right, with the pad token or else the EOS token.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model. The template must render all questions in one prompt.
        max_length (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens of a prompt.
        truncation (`str`, *optional*, defaults to `"error"`):
            Over-length or invalid records: raise (`"error"`), mark them skipped (`"skip"`), or cut the end of the
            state of an over-length prompt to `max_length` tokens and skip the rest (`"state"`).
        permute_fields (`bool`, *optional*, defaults to `False`):
            Re-render each record with its questions in a new random order every time it is batched.
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

    _request = staticmethod(decode_request)

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
        """Raise `error`, or unless `truncation="error"` return a skipped record with `reason`."""
        keys = ("input_ids", "marks", "num_options", "question_type_ids", "labels", "hard_labels", "is_choice")
        skipped = skipped_record(error, reason, keys, raise_error=self.truncation == "error")
        return {**skipped, **({"request": ""} if self.permute_fields else {})}

    def _render(self, request: dict[str, Any]) -> tuple[list[int], list[int], list]:
        """The `input_ids`, `marks` and questions of the one prompt of `request`."""
        prompts = self.decision_config.prepare_inputs(request, self.tokenizer)
        if len(prompts) != 1:
            raise ValueError(f"The template renders {len(prompts)} prompts; Clef needs all questions in one")
        inputs, questions = prompts[0]
        return inputs["input_ids"][0].tolist(), inputs["marks"].tolist(), questions

    def _cut_state(self, request, input_ids, marks) -> tuple[list[int], list[int]]:
        """Cut the end of the state of an over-length prompt to `max_length` tokens (see `cut_state`)."""
        return cut_state(self._render, request, input_ids, marks, self.max_length)

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render and tokenize one record; a record that cannot be used raises or is skipped (see `truncation`)."""
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
        """Re-render a record with its questions in a new order, each keeping its target (as it is when unchanged)."""
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
        default=0.05, metadata={"help": "Holdout fraction without an evaluation split."}
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "Random sample of training rows (seed)."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Random sample of evaluation rows."})
    eval_sample_seed: int = field(default=0, metadata={"help": "Seed of the holdout and of max_eval_samples."})

    mode: str = field(default="full", metadata={"help": "Train head, lora, or full model."})
    head_learning_rate: float | None = field(
        default=None, metadata={"help": "Learning rate of the head. Defaults to learning_rate."}
    )
    head_type: str | None = field(default=None, metadata={"help": "Head of a new model: joint or cosine."})
    global_pooling: str | None = field(default=None, metadata={"help": "last or first. Defaults by attention."})
    prompt_style: str | None = field(default=None, metadata={"help": "chatml or plain. Defaults by attention."})
    cosine_width: int | None = field(default=None, metadata={"help": "Width of the cosine head's projections."})
    cosine_initial_scale: float | None = field(default=None, metadata={"help": "Initial scale of the cosine head."})
    head_hidden_size: int | None = field(default=None, metadata={"help": "Width of the head."})
    head_intermediate_size: int | None = field(default=None, metadata={"help": "Feed-forward width of the head."})
    head_num_layers: int | None = field(default=None, metadata={"help": "Number of layers of the head."})
    head_num_evidence_layers: int | None = field(default=None, metadata={"help": "Layers in which options read."})
    head_num_attention_heads: int | None = field(default=None, metadata={"help": "Attention heads of the head."})
    label_smoothing: float | None = field(default=None, metadata={"help": "Loss label smoothing. Defaults to 0."})
    brier_weight: float | None = field(default=None, metadata={"help": "Weight of the Brier score. Defaults to 0."})
    ordinal_weight: float | None = field(default=None, metadata={"help": "Weight of the ordinal term. Defaults to 0."})
    rlcd_weight: float | None = field(default=None, metadata={"help": "Weight of the RLCD term. Defaults to 0."})
    rlcd_sigma: float | None = field(default=None, metadata={"help": "RLCD noise on the logits. Defaults to 0.4."})
    rlcd_samples: int | None = field(default=None, metadata={"help": "RLCD samples per question. Defaults to 32."})
    rlcd_normalize_advantage: bool | None = field(
        default=None, metadata={"help": "Divide RLCD advantages by their standard deviation. Defaults to False."}
    )
    rlcd_start_epoch: float | None = field(
        default=None, metadata={"help": "Epoch at which RLCD starts, after cross-entropy training."}
    )
    rlcd_sigmas: list[float] | None = field(
        default=None, metadata={"help": "One RLCD noise scale per epoch of the RLCD phase, e.g. [0.4, 0.3, 0.2, 0.1]."}
    )
    max_length: int = field(default=2048, metadata={"help": "Maximum prompt token length."})
    truncation: str = field(
        default="error", metadata={"help": "Over-length or invalid records: error, skip or state."}
    )
    permute_fields: bool = field(default=False, metadata={"help": "Shuffle the question order of training prompts."})


@dataclass
class ClefModelArguments(ModelConfig):
    model_name_or_path: str = field(
        default="MinimaLabs/KLM-1.6", metadata={"help": "Clef checkpoint, or the backbone of a new head."}
    )
    lora_task_type: str = field(default="FEATURE_EXTRACTION", metadata={"help": "PEFT task type for the backbone."})
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Run the code of a model that Transformers does not have. Off by default."}
    )

    def __post_init__(self):
        super().__post_init__()
        # A config file gives its value as it is (`trust_remote_code: 1`); the command line parses it as a boolean
        if self.trust_remote_code in (0, 1) and not isinstance(self.trust_remote_code, bool):
            self.trust_remote_code = bool(self.trust_remote_code)
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError(f"trust_remote_code must be true or false, not {self.trust_remote_code!r}")


RLCD_SETTINGS = ("rlcd_weight", "rlcd_sigma", "rlcd_samples", "rlcd_normalize_advantage")
LOSS_SETTINGS = ("label_smoothing", "brier_weight", "ordinal_weight", *RLCD_SETTINGS)
# The sizes of the head under the names of `ClefConfig`; the script's are `head_<name>`, `head_num_layers` for layers
HEAD_SIZES = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_evidence_layers", "num_attention_heads")


def rlcd_schedule(script_args) -> RlcdScheduleCallback | None:
    """The callback that runs RLCD after cross-entropy training from `rlcd_start_epoch`, or `None` without one."""
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
    """A Clef checkpoint as it is, or a new head on any other model as its backbone."""
    from transformers import ClefConfig, ClefForDecision

    name, revision = model_args.model_name_or_path, model_args.model_revision
    losses = {key: getattr(script_args, key) for key in LOSS_SETTINGS if getattr(script_args, key) is not None}
    if rlcd_schedule(script_args) is not None:  # the schedule turns RLCD on: the model starts as plain cross-entropy
        losses["rlcd_weight"] = 0.0
    names = ("head_type", "global_pooling", "prompt_style", "cosine_width", "cosine_initial_scale")
    head_settings = {key: getattr(script_args, key) for key in names if getattr(script_args, key) is not None}
    sizes = {key: getattr(script_args, "head_" + key.replace("hidden_layers", "layers")) for key in HEAD_SIZES}
    sizes = {key: value for key, value in sizes.items() if value is not None}
    trust_remote_code, model_kwargs = remote_code_setting(model_args, model_kwargs)
    # Clef is a model of Transformers: classes a Clef checkpoint names for itself never run (its backbone's may)
    saved = raw_config(name, revision=revision)
    if saved.get("model_type") == ClefConfig.model_type and saved.get("auto_map"):
        raise ValueError(
            f"{name} is a Clef checkpoint whose config.json names classes of its own code ({saved['auto_map']}). "
            "Clef is a model of Transformers and this example never runs a Clef checkpoint's own classes; remove "
            "the top-level auto_map from its config.json to load it with Transformers' classes"
        )
    # Raises before any code of the checkpoint is imported, unless it is needed and allowed
    remote_code = needs_remote_code(name, trust_remote_code, revision=revision)
    config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=remote_code)
    is_checkpoint = (
        config.model_type == "clef"
        or cached_file(name, "joint_head_config.json", revision=revision, _raise_exceptions_for_missing_entries=False)
        is not None
    )
    if is_checkpoint:
        attention = model_kwargs.get("attn_implementation")
        if isinstance(attention, str):  # the backbone's; Clef's head keeps its own
            model_kwargs = {**model_kwargs, "attn_implementation": {"text_config": attention}}
        # Settings of the backbone's config (a raised context) are saved in its `text_config`: changing one is refused
        saved_backbone = saved.get("text_config") or {}
        backbone_settings = {
            key: model_kwargs.pop(key) for key in list(model_kwargs) if key in saved_backbone and key not in saved
        }
        changed = {key: value for key, value in backbone_settings.items() if saved_backbone[key] != value}
        if changed:
            kept = {key: saved_backbone[key] for key in changed}
            raise ValueError(
                f"{name} is a Clef checkpoint whose backbone was saved with {kept}; model_init_kwargs {changed} "
                "cannot change it. Remove them, or train a new head from the backbone with them."
            )
        model, info = AutoModelForDecision.from_pretrained(
            name, output_loading_info=True, **model_kwargs, trust_remote_code=remote_code
        )
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
        # The model is built around the backbone as loaded, with every `model_init_kwargs` entry
        backbone = load_decision_backbone(name, trust_remote_code, model_kwargs, config=config)
        check_cosine_settings(script_args, head_settings.get("head_type", "joint"))
        if (losses.get("rlcd_weight") or 0) > 0:
            warnings.warn(
                f"rlcd_weight={losses['rlcd_weight']} on a new head with no schedule runs RLCD from the first step. "
                "RLCD is meant as a phase after cross-entropy training: train with cross-entropy first and start again "
                "from the saved checkpoint, or set rlcd_start_epoch.",
                stacklevel=2,
            )
        causal = backbone.causal
        defaults = {"global_pooling": "last" if causal else "first", "prompt_style": "chatml" if causal else "plain"}
        config = ClefConfig(
            text_config=backbone.model.config,
            backbone_hidden_size=backbone.hidden_size,
            backbone_hidden_module=backbone.hidden_module,
            **sizes,
            **losses,
            **{**defaults, **head_settings},
        )
        # The head is new, in float32, with Clef's default attention (its layers do not support every kernel)
        model = ClefForDecision(config, model=backbone.model)
        # The cosine head, encoders without a causal LM class and a backbone of its own code have none
        if hasattr(model, "lm_head"):
            with torch.no_grad():
                model.lm_head.weight.copy_(backbone.embeddings)
            model.lm_head.to(backbone.model.dtype)
        model.backbone_weights = backbone.weights
        del backbone
    if model.decision_config.chat_template is None:  # the template of the model's own prompt style
        model.decision_config = clef_decision_config(model.config.prompt_style)
    return model


def check_cosine_settings(script_args, head_type):
    """Refuse `cosine_width` and `cosine_initial_scale` for a head that is not the cosine head, which ignores them."""
    given = [key for key in ("cosine_width", "cosine_initial_scale") if getattr(script_args, key) is not None]
    if given and head_type != "cosine":
        raise ValueError(f"{given} only apply to head_type: cosine, and this head is {head_type}; remove them")


def describe(model) -> str:
    """The start-up line: the backbone's attention, pooling, prompt style, head and where its weights came from."""
    config = model.config
    attention = "causal" if is_causal(config.text_config) else "bidirectional"
    weights = getattr(model, "backbone_weights", None)
    module = getattr(config, "backbone_hidden_module", None)
    return (
        f"Clef setup: {attention} backbone ({config.text_config.model_type}), global_pooling={config.global_pooling}, "
        f"prompt_style={config.prompt_style}, head_type={config.head_type}"
        + ("" if module is None else f", backbone_hidden_module={module}")
        + ("" if weights is None else f", backbone weights: {weights}")
    )


def sample_rows(dataset, count, seed):
    """`count` rows of `dataset` drawn at random with `seed` (datasets are often sorted), or all of them."""
    if count is None or count >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(count))


def prepare_split(dataset, collator, training_args, desc):
    """Tokenize once, drop the skipped records, and report what was built."""
    integers = Sequence(Value("int32"))
    features = {"input_ids": integers, "marks": integers, "num_options": integers, "question_type_ids": integers}
    features |= {"labels": Sequence(Sequence(Value("float32"))), "hard_labels": integers}
    features |= {"is_choice": Sequence(Value("bool")), "skip": Value("bool"), "reason": Value("string")}
    features["truncated"] = Value("bool")
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
        # The records whose state was cut to fit (truncation="state")
        "truncated": truncated,
    }
    if training_args.process_index == 0:
        print(f"Build report ({desc}): {json.dumps(report)}")
    return tokenized, report


def set_head_learning_rate(trainer, head_learning_rate):
    """The backbone (with any LoRA adapters) at `learning_rate`, the head at `head_learning_rate`, as in Jeva."""
    model, args = trainer.model, trainer.args
    backbone_parameters = {id(parameter) for parameter in model.base_model.parameters()}
    decay_parameters = trainer.get_decay_parameter_names(model)
    parameter_groups = []
    for backbone in (True, False):
        for decay in (True, False):
            parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and (id(parameter) in backbone_parameters) == backbone
                and (name in decay_parameters and parameter.dim() > 0) == decay
            ]
            if parameters:
                lr = args.learning_rate if backbone else head_learning_rate
                weight_decay = args.weight_decay if decay else 0.0
                parameter_groups.append({"params": parameters, "lr": lr, "weight_decay": weight_decay})
    optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(args, model)
    try:
        from transformers.trainer_optimizer import is_optimizer_factory
    except ImportError:  # Transformers versions without optimizer factories
        is_optimizer_factory = lambda optimizer_cls: False  # noqa: E731
    # Optimizers that build their own parameter groups, or take the model, would drop these
    if is_optimizer_factory(optimizer_cls) or {"params", "model", "optimizer_dict"} & set(optimizer_kwargs):
        raise ValueError(f"head_learning_rate is not supported with the {args.optim} optimizer")
    optimizer_kwargs["params"] = parameter_groups
    trainer.optimizer_cls_and_kwargs = (optimizer_cls, optimizer_kwargs)


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
    trust_remote_code, model_kwargs = remote_code_setting(model_args, model_kwargs)
    model = build_model(script_args, model_args, model_kwargs)
    if training_args.activation_checkpointing and not model.model.supports_gradient_checkpointing:
        raise ValueError(
            f"activation_checkpointing (or gradient_checkpointing) is on, but the {type(model.model).__name__} "
            "backbone does not support it; set activation_checkpointing: false"
        )
    if script_args.mode == "lora" and getattr(model.config, "backbone_hidden_module", None) is not None:
        # The name is resolved against the backbone, which PEFT would wrap: the first step would fail
        raise ValueError(
            f"mode: lora cannot put adapters on the {type(model.model).__name__} backbone, whose hidden states are read "
            f"from an inner layer (backbone_hidden_module={model.config.backbone_hidden_module}); use mode: full or "
            "mode: head"
        )
    warning = check_max_length(model.model, model.config.text_config, script_args.max_length)
    if training_args.process_index == 0:
        if warning is not None:
            print(warning)
        print(describe(model))
    name, revision = model_args.model_name_or_path, model_args.model_revision
    remote_code = needs_remote_code(name, trust_remote_code, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=remote_code)
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

        # Adapters on the backbone only; the head trains in full (`modules_to_save` cannot wrap its `scorer` cleanly)
        model.model = lora_backbone = get_peft_model(model.model, peft_config)

    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, revision=script_args.dataset_revision
    )
    train_dataset = dataset[script_args.dataset_train_split]
    if script_args.dataset_test_split is None:
        # The holdout and its sample do not depend on the run's seed: runs with different seeds evaluate alike
        split = train_dataset.train_test_split(script_args.validation_fraction, seed=script_args.eval_sample_seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        eval_dataset = dataset[script_args.dataset_test_split]
    train_dataset = sample_rows(train_dataset, script_args.max_train_samples, training_args.seed)
    eval_dataset = sample_rows(eval_dataset, script_args.max_eval_samples, script_args.eval_sample_seed)

    # Training batches permute the questions; evaluation never does: it has its own collator
    collator_kwargs = {"max_length": script_args.max_length, "truncation": script_args.truncation}
    permuting = {"permute_fields": script_args.permute_fields, "seed": training_args.seed}
    collator = DataCollatorForClef(tokenizer, model.decision_config, **permuting, **collator_kwargs)
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
    if script_args.head_learning_rate is not None:
        set_head_learning_rate(trainer, script_args.head_learning_rate)
    if schedule is not None:
        trainer.add_callback(schedule)
    return ClefRun(trainer, lora_backbone, {"train": train_report, "eval": eval_report})


def main(script_args, training_args, model_args):
    run = build_trainer(script_args, training_args, model_args)
    trainer, lora_backbone, model = run.trainer, run.lora_backbone, run.trainer.model
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if lora_backbone is not None:
        # The adapter alone in `adapter/`, then merged: the output is a native Clef checkpoint
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
