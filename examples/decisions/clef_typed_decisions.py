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
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForDecision, AutoTokenizer, set_seed
from transformers.decision import DecisionConfig as TransformersDecisionConfig
from transformers.decision import Question
from transformers.utils import cached_file

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.decision import (
    DataCollatorForDecision,
    DecisionConfig,
    DecisionTrainer,
    compute_calibration_metrics,
)


def clef_decision_config():
    from transformers.models.clef.configuration_clef import CLEF_TEMPLATE

    return TransformersDecisionConfig(noul_options=["true", "false"], sort_choices=True, chat_template=CLEF_TEMPLATE)


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
            What to do with a record whose prompt is longer than `max_length`: reject it (`"error"`) or mark it
            as skipped (`"skip"`). Prompts are never cut, since that would remove options or questions.
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

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render and tokenize one record with the native Transformers template.

        Returns `input_ids`, `marks`, `num_options`, `question_type_ids`, `labels` (the target distribution of each
        question over its options), `is_choice`, `skip` and `reason`, plus the `request` when `permute_fields`. A
        record that is too long raises, or with `truncation="skip"` returns `skip=True` and the empty lists.
        """
        request = self._request(example)
        for name, question in request["questions"].items():
            if isinstance(question, dict) and question.get("allow_null"):
                raise ValueError(
                    f"Question {name!r} is nullable; Clef has no coverage output, so it cannot be trained"
                )
        inputs, questions, targets = self.prepare_request(example)
        for question in questions:
            if asdict(question).get("allow_null", False):
                raise ValueError(f"Question {question.name!r} is nullable; Clef has no coverage output")
        if self.permute_fields and [question.name for question in questions] != list(request["questions"]):
            raise ValueError("permute_fields needs a template that renders the questions in the order of the request")
        length = inputs["input_ids"].shape[-1]
        if length > self.max_length:
            reason = f"longer than max_length ({length} > {self.max_length} tokens)"
            if self.truncation == "error":
                raise ValueError(f"A record is {reason}; increase max_length or set truncation='skip'")
            empty = {
                key: [] for key in ("input_ids", "marks", "num_options", "question_type_ids", "labels", "is_choice")
            }
            return {**empty, "skip": True, "reason": reason, **({"request": ""} if self.permute_fields else {})}
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
    head_type: str = field(default="joint", metadata={"help": "Decision head: joint."})
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
    truncation: str = field(default="error", metadata={"help": "Overlength policy: error or skip."})
    permute_fields: bool = field(default=False, metadata={"help": "Shuffle the question order of training prompts."})


@dataclass
class ClefModelArguments(ModelConfig):
    model_name_or_path: str = field(
        default="MinimaLabs/KLM-1.6", metadata={"help": "Clef checkpoint, or a causal LM to use as the backbone."}
    )
    lora_task_type: str = field(default="FEATURE_EXTRACTION", metadata={"help": "PEFT task type for the backbone."})


def build_model(script_args, model_args, model_kwargs):
    """A Clef checkpoint is loaded as it is; any other model becomes the backbone of a new head."""
    from transformers import ClefConfig, ClefForDecision

    name, revision = model_args.model_name_or_path, model_args.model_revision
    losses = {
        key: getattr(script_args, key)
        for key in ("label_smoothing", "brier_weight", "ordinal_weight")
        if getattr(script_args, key) is not None
    }
    is_checkpoint = (
        AutoConfig.from_pretrained(name, revision=revision).model_type == "clef"
        or cached_file(name, "joint_head_config.json", revision=revision, _raise_exceptions_for_missing_entries=False)
        is not None
    )
    if is_checkpoint:
        model = AutoModelForDecision.from_pretrained(name, **model_kwargs)
        for key, value in losses.items():
            setattr(model.config, key, value)
    else:
        if script_args.head_type != "joint":
            raise ValueError(f"head_type {script_args.head_type!r} is not available; this build has only 'joint'")
        causal_lm = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
        backbone = causal_lm.base_model
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
        config = ClefConfig(text_config=text_config, backbone_hidden_size=text_config.hidden_size, **sizes, **losses)
        model = ClefForDecision(config)
        model.model.load_state_dict(backbone.state_dict())
        with torch.no_grad():
            model.lm_head.weight.copy_(causal_lm.get_output_embeddings().weight)
        # The head stays in float32; the backbone keeps the dtype it was loaded in
        model.model.to(causal_lm.dtype)
        model.lm_head.to(causal_lm.dtype)
        del causal_lm, backbone
    if model.decision_config.chat_template is None:
        model.decision_config = clef_decision_config()
    return model


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
    if training_args.gradient_checkpointing and not model.supports_gradient_checkpointing:
        raise ValueError(f"{type(model).__name__} does not support gradient checkpointing; set it to false")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.mode == "full":
        model.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
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
