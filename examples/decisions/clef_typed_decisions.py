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
Train a Clef decision model on typed decisions: one prompt per record, every question answered jointly.

`model_name_or_path` is a Clef checkpoint, loaded as it is, or a backbone for a new head: a causal LM (KLM-1.6, Qwen)
reads its last token with the ChatML prompt; a bidirectional encoder (BERT, ModernBERT, or a reranker, unwrapped to
its base model) reads its first token with the plain prompt between [CLS] and [SEP]. `global_pooling` and
`prompt_style` override that choice; `head_type: cosine` (with `cosine_width`, `cosine_initial_scale`) selects the
small cosine head instead of the joint one. The resolved setup is printed at start-up.

python examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml

torchrun --nproc_per_node=2 \
    examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
"""

import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Features, Sequence, Value, load_dataset
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
    compute_calibration_metrics,
)


def clef_decision_config():
    """Clef's decision config with its ChatML prompt: any number of options, since the head reads no label token."""
    from transformers import ClefConfig

    return ClefConfig.prompt_decision_config("chatml")


@dataclass
class DataCollatorForClef(DataCollatorForDecision):
    """
    Batch records for Clef: one prompt per record, with all of its questions answered jointly.

    Each example contains a `request` (a dictionary or JSON string), or `state` and `questions`, plus the targets of
    all questions: `targets` (a mapping of question name to `target_index`, `target_probs`, `p_true` or `score`), or
    `gold` / `answers` (a mapping of question name to a gold label or probabilities). Nullable questions are not
    supported, since Clef has no coverage output.

    For repeated training or evaluation, preprocess once with `dataset.map(collator.tokenize)`. The collator also
    accepts these prepared examples and only pads them when building batches. A batch holds the questions of all its
    records one after the other: `labels` and `option_mask` have one row per question, in record order then question
    order, and `rows_per_example` holds the number of questions of each record. With it, `DecisionTrainer` gathers
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
            invalid: reject it (`"error"`) or mark it as skipped (`"skip"`) under a reason that does not depend on
            the record. Prompts are never cut, since that would remove options or questions.
        permute_fields (`bool`, *optional*, defaults to `False`):
            Keep the request of each record and re-render it with its questions in a new random order every time it
            is batched. Only records tokenized by a collator with `permute_fields` are permuted, so evaluation uses
            a separate collator without it. A re-render that fails, or that no longer matches the targets, raises.
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
        if self.truncation not in {"error", "skip"}:
            raise ValueError("truncation must be error or skip")

    @staticmethod
    def _request(example: dict[str, Any]) -> dict[str, Any]:
        request = example["request"] if "request" in example else {k: example[k] for k in ("state", "questions")}
        request = json.loads(request) if isinstance(request, str) else dict(request)
        if isinstance(request["questions"], str):
            request["questions"] = json.loads(request["questions"])
        return request

    def _features(self, inputs, questions, labels):
        return {
            "input_ids": inputs["input_ids"][0].tolist(),
            "marks": inputs["marks"].tolist(),
            "num_options": [len(question.options) for question in questions],
            "question_type_ids": [Question.questions_types.index(question.type) for question in questions],
            "labels": labels,
            "is_choice": [question.type == "choice" for question in questions],
        }

    def _skip(self, reason: str, error: Exception) -> dict[str, Any]:
        """Raise `error`, or with `truncation="skip"` return a skipped record with the empty lists and `reason`."""
        if self.truncation == "error":
            raise error
        empty = {key: [] for key in ("input_ids", "marks", "num_options", "question_type_ids", "labels", "is_choice")}
        return {**empty, "skip": True, "reason": reason, **({"request": ""} if self.permute_fields else {})}

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render and tokenize one record with the native Transformers template.

        Returns `input_ids`, `marks`, `num_options`, `question_type_ids`, `labels` (the target distribution of each
        question over its options), `is_choice`, `skip` and `reason`, plus the `request` when `permute_fields`. A
        record that is too long, whose request is invalid, that asks a nullable question or whose targets cannot be
        used raises, or with `truncation="skip"` returns `skip=True`, the empty lists and one of the reasons
        `"longer than max_length (N tokens)"`, `"invalid request"`, `"nullable question"` and `"unusable targets"`.
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
            prompts = self.decision_config.prepare_inputs(request, self.tokenizer)
            if len(prompts) != 1:
                raise ValueError(f"The template renders {len(prompts)} prompts; Clef needs all questions in one")
        except ValueError as error:
            return self._skip("invalid request", error)
        inputs, questions = prompts[0]
        try:
            targets = self.prepare_targets(example, questions)
        except ValueError as error:
            return self._skip("unusable targets", error)
        for question in questions:
            if asdict(question).get("allow_null", False):
                error = ValueError(f"Question {question.name!r} is nullable; Clef has no coverage output")
                return self._skip("nullable question", error)
        if self.permute_fields and [question.name for question in questions] != list(request["questions"]):
            raise ValueError("permute_fields needs a template that renders the questions in the order of the request")
        length = inputs["input_ids"].shape[-1]
        if length > self.max_length:
            error = ValueError(
                f"A record is longer than max_length ({length} > {self.max_length} tokens); increase max_length or "
                "set truncation='skip'"
            )
            return self._skip(f"longer than max_length ({self.max_length} tokens)", error)
        # The null column is dropped: Clef has no null output
        features = self._features(inputs, questions, [list(target[1:]) for target in targets])
        features |= {"skip": False, "reason": ""}
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
        """Re-render the request of a record with its questions in a new order, each keeping its target."""
        request = json.loads(feature["request"])
        names = list(request["questions"])
        if len(names) != len(feature["labels"]):
            raise ValueError("The request and the labels of a record have different numbers of questions")
        order = list(range(len(names)))
        self._shuffler().shuffle(order)
        permuted = {**request, "questions": {names[i]: request["questions"][names[i]] for i in order}}
        try:
            prompts = self.decision_config.prepare_inputs(permuted, self.tokenizer)
        except Exception as error:
            raise ValueError(f"Re-rendering a request with permuted questions failed: {error}") from error
        if len(prompts) != 1:
            raise ValueError("The template must render all questions in one prompt")
        inputs, questions = prompts[0]
        if [question.name for question in questions] != [names[i] for i in order]:
            raise ValueError("The permuted prompt does not list the questions in the permuted order")
        labels = [feature["labels"][i] for i in order]
        if [len(question.options) for question in questions] != [len(row) for row in labels]:
            raise ValueError("The options of a permuted question no longer match its labels")
        if inputs["input_ids"].shape[-1] > self.max_length:
            raise ValueError(f"A permuted prompt is longer than max_length ({self.max_length} tokens)")
        return {**feature, **self._features(inputs, questions, labels)}

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" not in features[0]:
            features = [self.tokenize(example) for example in features]
        if any(feature.get("skip") for feature in features):
            raise ValueError("Skipped records must be filtered out before batching")
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
        default=0.05, metadata={"help": "Fraction of records held out when there is no evaluation split."}
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "Limit training records."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Limit evaluation records."})

    mode: str = field(default="full", metadata={"help": "Train head, lora, or full model."})
    head_type: str | None = field(
        default=None,
        metadata={
            "help": "Decision head of a new model: joint (the default) or cosine. A Clef checkpoint keeps its own head."
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
    max_length: int = field(default=2048, metadata={"help": "Maximum prompt token length."})
    truncation: str = field(
        default="error", metadata={"help": "Over-length or invalid records: error, or skip and count them."}
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


def load_backbone(name, config, model_kwargs):
    """
    The backbone of a new head, and the table its `lm_head` starts from (`None` without one): the base model of a
    causal LM with its output embeddings, or a bidirectional encoder with its input embeddings. A checkpoint that only
    registers a sequence classification class (a reranker) is loaded as that and unwrapped to its `base_model`.
    """
    if is_causal(config):
        causal_lm = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
        return causal_lm.base_model, causal_lm.get_output_embeddings().weight
    architectures = config.architectures or []
    if architectures and all(architecture.endswith("ForSequenceClassification") for architecture in architectures):
        classifier = AutoModelForSequenceClassification.from_pretrained(name, **model_kwargs)
        backbone = classifier.base_model
        if backbone is classifier or not hasattr(backbone, "get_input_embeddings"):
            raise ValueError(
                f"{name} registers only {architectures}, whose base model cannot be unwrapped into a backbone"
            )
    else:
        backbone = AutoModel.from_pretrained(name, **model_kwargs)
    return backbone, backbone.get_input_embeddings().weight


def build_model(script_args, model_args, model_kwargs):
    """
    A Clef checkpoint is loaded as it is; any other model becomes the backbone of a new head. A causal backbone reads
    its last token with the ChatML prompt, a bidirectional one (an encoder) its first with the plain prompt, unless
    `global_pooling` or `prompt_style` say otherwise.
    """
    from transformers import ClefConfig, ClefForDecision

    name, revision = model_args.model_name_or_path, model_args.model_revision
    losses = {
        key: getattr(script_args, key)
        for key in ("label_smoothing", "brier_weight", "ordinal_weight")
        if getattr(script_args, key) is not None
    }
    head_settings = {
        key: getattr(script_args, key)
        for key in ("head_type", "global_pooling", "prompt_style", "cosine_width", "cosine_initial_scale")
        if getattr(script_args, key) is not None
    }
    config = AutoConfig.from_pretrained(name, revision=revision)
    is_checkpoint = (
        config.model_type == "clef"
        or cached_file(name, "joint_head_config.json", revision=revision, _raise_exceptions_for_missing_entries=False)
        is not None
    )
    if is_checkpoint:
        model = AutoModelForDecision.from_pretrained(name, **model_kwargs)
        changed = {key: value for key, value in head_settings.items() if getattr(model.config, key) != value}
        if changed:
            raise ValueError(f"{name} is a Clef checkpoint; its head settings cannot change ({changed})")
        for key, value in losses.items():
            setattr(model.config, key, value)
    else:
        backbone, embeddings = load_backbone(name, config, model_kwargs)
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
        text_config = backbone.config
        causal = is_causal(text_config)
        defaults = {"global_pooling": "last" if causal else "first", "prompt_style": "chatml" if causal else "plain"}
        config = ClefConfig(
            text_config=text_config,
            backbone_hidden_size=text_config.hidden_size,
            **sizes,
            **losses,
            **{**defaults, **head_settings},
        )
        model = ClefForDecision(config)
        model.model.load_state_dict(backbone.state_dict())
        # The head stays in float32; the backbone keeps the dtype it was loaded in
        model.model.to(backbone.dtype)
        if hasattr(model, "lm_head"):  # the cosine head, and encoders without a causal LM class, have none
            with torch.no_grad():
                model.lm_head.weight.copy_(embeddings)
            model.lm_head.to(backbone.dtype)
        del backbone, embeddings
    if model.decision_config.chat_template is None:
        model.decision_config = clef_decision_config()
    return model


def describe(model) -> str:
    """The resolved setup of a Clef model: the attention of its backbone, its pooling, prompt style and head."""
    config = model.config
    attention = "causal" if is_causal(config.text_config) else "bidirectional"
    return (
        f"Clef setup: {attention} backbone ({config.text_config.model_type}), global_pooling={config.global_pooling}, "
        f"prompt_style={config.prompt_style}, head_type={config.head_type}"
    )


def prepare_split(dataset, collator, training_args, desc):
    """Tokenize once, drop the skipped records, and report what was built."""
    features = {
        "input_ids": Sequence(Value("int32")),
        "marks": Sequence(Value("int32")),
        "num_options": Sequence(Value("int32")),
        "question_type_ids": Sequence(Value("int32")),
        "labels": Sequence(Sequence(Value("float32"))),
        "is_choice": Sequence(Value("bool")),
        "skip": Value("bool"),
        "reason": Value("string"),
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
        tokenized = tokenized.filter(lambda skip: not skip, input_columns="skip").remove_columns(["skip", "reason"])
    report = {
        "rows": len(dataset),
        "questions": sum(len(counts) for counts in tokenized["num_options"]),
        "skipped": dict(reasons),
        # Prompts are never cut: an over-length record is rejected or skipped
        "truncated": 0,
    }
    if training_args.process_index == 0:
        print(f"Build report ({desc}): {json.dumps(report)}")
    return tokenized, report


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    if script_args.mode not in {"head", "lora", "full"}:
        raise ValueError("mode must be head, lora, or full")
    training_args.accelerator_config.dispatch_batches = False
    training_args.accelerator_config.split_batches = False
    model_kwargs = dict(training_args.model_init_kwargs or {})
    model_kwargs.setdefault("revision", model_args.model_revision)
    model_kwargs.setdefault("dtype", model_args.dtype)
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = build_model(script_args, model_args, model_kwargs)
    if training_args.process_index == 0:
        print(describe(model))
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
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
        split = train_dataset.train_test_split(test_size=script_args.validation_fraction, seed=training_args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(script_args.max_train_samples, len(train_dataset))))
    if script_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(script_args.max_eval_samples, len(eval_dataset))))

    # Training batches permute the questions; evaluation never does, so it has its own collator
    collator_kwargs = {"max_length": script_args.max_length, "truncation": script_args.truncation}
    collator = DataCollatorForClef(
        tokenizer,
        model.decision_config,
        permute_fields=script_args.permute_fields,
        seed=training_args.seed,
        **collator_kwargs,
    )
    eval_collator = DataCollatorForClef(tokenizer, model.decision_config, **collator_kwargs)
    train_dataset, _ = prepare_split(train_dataset, collator, training_args, "train")
    eval_dataset, _ = prepare_split(eval_dataset, eval_collator, training_args, "eval")

    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_calibration_metrics,
    )
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
