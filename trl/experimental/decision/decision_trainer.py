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

# Jeva: System One Models from First Principles
# Copyright 2026 Qi Li and Xinchao Wang.

import json
import os
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import torch.nn.functional as F
from accelerate.utils import recursively_apply
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForDecision,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    set_seed,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_peft_available,
    is_sagemaker_mp_enabled,
    logging,
)

from ...trainer.base_trainer import _BaseTrainer
from .decision_config import DecisionConfig


if is_peft_available():
    from peft import PeftModel, get_peft_model


logger = logging.get_logger(__name__)


def _to_model_layout(model: PreTrainedModel, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a state dict saved by `model.save_pretrained` back to the keys and tensors of `model`.

    Saving reverts the model's weight conversions (renamings, fused projections, ...) to the layout of its original
    checkpoints; this applies them again, with the same conversions saving used, without loading anything into the
    model. Keys that no conversion matches are kept as they are.

    This mirrors private internals of `transformers.core_model_loading`, read at Transformers commit 02b9cd260c (the
    Clef training branch on top of #50058): the conversions `revert_weight_conversion` reverts when saving, and the key
    renaming and tensor conversion of `convert_and_load_state_dict_in_model` (`rename_source_key`, `dot_natural_key`,
    `WeightRenaming` / `WeightConverter.add_tensor` and `.convert`). Check it against them when Transformers changes;
    the checkpoint-restore tests fail if it drifts.
    """
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import (
        PrefixChange,
        WeightConverter,
        WeightRenaming,
        dot_natural_key,
        rename_source_key,
    )

    # The conversions `revert_weight_conversion` reverts when saving
    conversions = getattr(model, "_weight_conversions", None)
    if conversions is None:
        conversions = get_model_conversion_mapping(model, add_legacy=False)
        conversions = [conversion for conversion in conversions if not isinstance(conversion, PrefixChange)]
    if not conversions:
        return state_dict
    renamings = [conversion for conversion in conversions if isinstance(conversion, WeightRenaming)]
    converters = [conversion for conversion in conversions if isinstance(conversion, WeightConverter)]
    converter_by_pattern = {pattern: converter for converter in converters for pattern in converter.source_patterns}
    expected = model.state_dict()
    transforms = {}
    for key, tensor in sorted(state_dict.items(), key=lambda item: dot_natural_key(item[0])):
        target, pattern = rename_source_key(key, renamings, converters, model.base_model_prefix, expected)
        if target not in expected and key in expected:
            target, pattern = key, None
        if pattern is None:
            transform = transforms.setdefault(target, WeightRenaming(key, target))
            pattern = key
        else:
            transform = transforms.setdefault(target, deepcopy(converter_by_pattern[pattern]))
        transform.add_tensor(target, key, pattern, tensor)
    converted = {}
    for target, transform in transforms.items():
        for name, value in transform.convert(target, model=model, config=model.config).items():
            converted[name] = value[0] if isinstance(value, list) else value
    return converted


def pack_rows(rows: torch.Tensor, rows_per_example: torch.Tensor, max_rows: int, fill: float) -> torch.Tensor:
    """
    Pack `rows` (`(num_rows, ...)`), whose first `rows_per_example[0]` rows belong to the first record, the next
    `rows_per_example[1]` to the second and so on, into `(num_records, max_rows, ...)`. The slots after the rows of a
    record hold `fill`.
    """
    if int(rows_per_example.sum()) != rows.shape[0]:
        raise ValueError(
            f"rows_per_example adds up to {int(rows_per_example.sum())} rows, but the batch has {rows.shape[0]}"
        )
    if rows_per_example.numel() and int(rows_per_example.max()) > max_rows:
        raise ValueError(f"A record has more than {max_rows} rows")
    records = torch.arange(len(rows_per_example), device=rows.device)
    record = torch.repeat_interleave(records, rows_per_example.to(rows.device))
    starts = rows_per_example.to(rows.device).cumsum(0) - rows_per_example.to(rows.device)
    position = torch.arange(rows.shape[0], device=rows.device) - starts[record]
    packed = rows.new_full((len(rows_per_example), max_rows, *rows.shape[1:]), fill)
    packed[record, position] = rows
    return packed


def unpack_rows(packed: torch.Tensor, rows_per_example: torch.Tensor) -> torch.Tensor:
    """The rows packed by [`pack_rows`], in record order then row order."""
    slots = torch.arange(packed.shape[1], device=packed.device)
    return packed[slots[None, :] < rows_per_example.to(packed.device)[:, None]]


# Batch entries that describe the rows of a batch for the metrics and are not passed to the model
METADATA_KEYS = frozenset({"is_choice", "rows_per_example", "hard_labels"})


class DecisionTrainer(_BaseTrainer):
    """
    Trainer for models that return a supervised decision loss.

    Models return a mean `loss` and candidate `logits`. Labels may be class indices or target distributions. Coverage
    outputs are optional. Optimization, distributed wrapping, checkpointing, and callbacks use
    [`~transformers.Trainer`].

    Args:
        model ([`~transformers.PreTrainedModel`] or `str`):
            Model to train, or identifier of a saved decision checkpoint.
        args ([`experimental.decision.DecisionConfig`], *optional*):
            Training configuration.
        data_collator (`Callable`, *optional*):
            Collator producing the model's forward inputs and `labels`. It may add `is_choice` (which rows count for
            choice accuracy), `rows_per_example` (the number of consecutive rows of each record, when a record has
            several rows) and `hard_labels` (the gold label of each row, as an option index, next to target
            distributions in `labels`); none of them is passed to the model.
        train_dataset (`Dataset` or `IterableDataset`, *optional*):
            Dataset of model inputs or raw requests prepared by the collator.
        eval_dataset (`Dataset` or `IterableDataset` or `dict`, *optional*):
            Evaluation dataset or named evaluation datasets.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Encoder tokenizer. Loaded from the checkpoint when omitted.
        compute_metrics (`Callable`, *optional*):
            Receives predicted probabilities and labels. Nullable decisions include null in the first column. Padding
            added during evaluation has value `-100`. With `rows_per_example`, each record has the same number of row
            slots within an evaluation batch, in order; the slots it does not use are `-100` in every cell. With
            `hard_labels`, the labels are a tuple `(labels, hard_labels)`, in the same row layout.
        callbacks (`list[TrainerCallback]`, *optional*):
            Callbacks passed to [`~transformers.Trainer`].
        optimizers (`tuple`, *optional*, defaults to `(None, None)`):
            Optimizer and scheduler overrides.
        optimizer_cls_and_kwargs (`tuple`, *optional*):
            Optimizer class and keyword arguments.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Prediction preprocessing passed to [`~transformers.Trainer`].
        peft_config (`PeftConfig`, *optional*):
            PEFT configuration used to wrap the model. Configure any trainable head through `modules_to_save`.
    """

    loss_is_scaled_for_ga = True
    _tag_names = ["trl", "decision"]
    _name = "Decision"

    def __init__(
        self,
        model: PreTrainedModel | str,
        args: DecisionConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
        optimizer_cls_and_kwargs: tuple | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: Any | None = None,
    ):
        if args is None:
            args = DecisionConfig(output_dir="Decision")
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        set_seed(args.seed)
        model_id = (
            model
            if isinstance(model, str)
            else model.config._name_or_path or model.config.get_text_config()._name_or_path
        )
        if isinstance(model, str):
            model = AutoModelForDecision.from_pretrained(model, **(args.model_init_kwargs or {}))
        if model.is_distributed_loading_by_transformers and "ep_dispatch_experts" in model.ep_plan.values():
            raise ValueError(
                "Token-dispatch expert parallelism is not supported by Trainer; use masked expert parallelism"
            )
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        if args.n_gpu > 1:
            raise ValueError(
                "Use accelerate launch or torchrun for multi-GPU decision training instead of DataParallel"
            )
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("Install `trl[peft]` to use peft_config")
            model = get_peft_model(model, peft_config)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.model_accepts_loss_kwargs = False
        parallelism_config = self.accelerator.parallelism_config
        if parallelism_config is not None and (parallelism_config.cp_enabled or parallelism_config.sp_enabled):
            raise ValueError("DecisionTrainer does not support context or sequence parallelism")

    def create_optimizer(self, model=None):
        """
        Set up the optimizer. With `head_learning_rate`, the parameters outside the backbone (`base_model`) train at
        that rate and the backbone, including any adapters inside it, at `learning_rate`; zero-dimensional parameters
        (scales, gates) are never decayed. Otherwise this is [`~transformers.Trainer.create_optimizer`].
        """
        if self.args.head_learning_rate is None or self.optimizer is not None:
            return super().create_optimizer(model)
        opt_model = self.model if model is None else model
        unwrapped = self.accelerator.unwrap_model(opt_model)
        if is_peft_available() and isinstance(unwrapped, PeftModel):
            unwrapped = unwrapped.get_base_model()
        backbone = {id(parameter) for parameter in unwrapped.base_model.parameters()}
        if not backbone or len(backbone) == len(list(unwrapped.parameters())):
            raise ValueError("head_learning_rate needs a model whose base_model is part of it, next to a head")
        decay = set(self.get_decay_parameter_names(opt_model))
        groups = {(in_backbone, decayed): [] for in_backbone in (True, False) for decayed in (True, False)}
        for name, parameter in opt_model.named_parameters():
            if parameter.requires_grad:
                decayed = name in decay and parameter.dim() > 0
                groups[id(parameter) in backbone, decayed].append(parameter)
        param_groups = [
            {
                "params": parameters,
                "weight_decay": self.args.weight_decay if decayed else 0.0,
                "lr": self.args.learning_rate if in_backbone else self.args.head_learning_rate,
            }
            for (in_backbone, decayed), parameters in groups.items()
            if parameters
        ]
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        try:
            from transformers.trainer_optimizer import is_optimizer_factory
        except ImportError:  # Transformers versions without optimizer factories
            is_optimizer_factory = lambda optimizer_cls: False  # noqa: E731
        if is_optimizer_factory(optimizer_cls) or {"params", "model", "optimizer_dict"} & set(optimizer_kwargs):
            raise ValueError(f"head_learning_rate is not supported with the {self.args.optim} optimizer")
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def _get_num_items_in_batch(self, batch_samples, device):
        count = torch.tensor(sum(batch["labels"].shape[0] for batch in batch_samples), device=device)
        return self.accelerator.gather(count).sum()

    @staticmethod
    def _get_decision_probabilities(outputs, inputs):
        logits = outputs.logits.float()
        if logits.ndim == 1:
            logits = logits[None]
        mask = inputs.get("option_mask", torch.ones_like(logits, dtype=torch.bool))
        probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        allow_null = inputs.get("allow_null")
        if allow_null is not None:
            coverage_logits = outputs.get("coverage_logits")
            if coverage_logits is None:
                if allow_null.any():
                    raise ValueError("Nullable decisions require a model that returns coverage_logits")
                null_probability = torch.zeros_like(probabilities[:, :1])
            else:
                coverage_logits = coverage_logits.float()
                coverage = torch.where(allow_null, coverage_logits.sigmoid(), 1.0)
                null_probability = torch.where(allow_null, (-coverage_logits).sigmoid(), 0.0)[:, None]
                probabilities = coverage[:, None] * probabilities
            probabilities = torch.cat([null_probability, probabilities], -1)
            mask = F.pad(mask, (1, 0), value=True)
        return probabilities, mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        model_inputs = {key: value for key, value in inputs.items() if key not in METADATA_KEYS}
        # Sharded encoders must execute the same number of option chunks on every rank.
        if self.accelerator.num_processes > 1 and inputs.get("option_chunk_size") is not None:
            count = inputs["option_input_ids"].new_tensor([inputs["option_input_ids"].shape[0]])
            padding = self.accelerator.gather(count).max().item() - count.item()
            for key in ("option_input_ids", "option_attention_mask"):
                if key in model_inputs:
                    value = model_inputs[key]
                    model_inputs[key] = torch.cat([value, value[:1].expand(padding, *value.shape[1:])])
        loss, outputs = super().compute_loss(model, model_inputs, return_outputs=True)
        loss = loss.float()
        if num_items_in_batch is not None:
            loss = loss * inputs["labels"].shape[0] / num_items_in_batch * self.accelerator.num_processes
        with torch.no_grad():
            probabilities, mask = self._get_decision_probabilities(outputs, inputs)
            labels = inputs["labels"]
            targets = F.one_hot(labels, probabilities.shape[-1]).float() if labels.ndim == 1 else labels.float()
            has_null = "allow_null" in inputs
            candidates = targets[:, 1:] if has_null else targets
            option_mask = mask[:, 1:] if has_null else mask
            logits = outputs.logits.float()
            if logits.ndim == 1:
                logits = logits[None]
            log_q = logits.masked_fill(~option_mask, -torch.inf).log_softmax(-1)
            candidate_loss = -(candidates * log_q.masked_fill((~option_mask) | (candidates == 0), 0.0)).sum(-1)
            predicted = probabilities.argmax(-1)
            target = targets.argmax(-1)
            is_choice = inputs.get("is_choice", torch.ones_like(target, dtype=torch.bool))
            hard_choice = is_choice & (targets.max(-1).values == 1)
            present = hard_choice & (target != 0) if has_null else hard_choice
            ones = torch.ones_like(candidate_loss)
            metrics = {
                "candidate_loss": (candidate_loss, ones),
                "choice_accuracy": ((predicted == target) & hard_choice, hard_choice),
                "candidate_accuracy": ((log_q.argmax(-1) + int(has_null) == target) & present, present),
                "brier": ((probabilities - targets).square().sum(-1), ones),
                "entropy": (-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1), ones),
            }
            if has_null:
                predicted_null = (predicted == 0) & hard_choice
                target_null = (target == 0) & hard_choice
                true_null = predicted_null & target_null
                metrics.update(
                    {
                        "null_precision": (true_null, predicted_null),
                        "null_recall": (true_null, target_null),
                        "null_rate": ((predicted == 0) & inputs["allow_null"], inputs["allow_null"]),
                        "null_brier": ((probabilities[:, 0] - targets[:, 0]).square(), inputs["allow_null"]),
                    }
                )
                if outputs.get("coverage_logits") is not None:
                    logits = outputs.coverage_logits.float()
                    coverage = candidates.sum(-1)
                    coverage_loss = coverage * F.softplus(-logits) + (1 - coverage) * F.softplus(logits)
                    metrics["coverage_loss"] = (coverage_loss * inputs["allow_null"], ones)
            if outputs.get("aux_loss") is not None:
                metrics["aux_loss"] = (outputs.aux_loss.detach().expand_as(ones), ones)
            values = torch.stack(
                [
                    torch.stack([numerator.float(), denominator.float()], -1)
                    for numerator, denominator in metrics.values()
                ],
                1,
            )
            rows_per_example = inputs.get("rows_per_example")
            if rows_per_example is not None:
                # One entry per record, summed over its rows: padded slots are zero in numerator and denominator
                max_rows = int(rows_per_example.max()) if rows_per_example.numel() else 0
                values = pack_rows(values, rows_per_example, max_rows, 0.0).sum(1)
            values = self._gather_for_metrics(values)
            for key, totals in zip(metrics, values.sum(0).tolist(), strict=True):
                self._metrics[mode][key].append(totals)
        return (loss, outputs) if return_outputs else loss

    def _gather_for_metrics(self, values):
        replicas = self.get_tp_size()
        if replicas == 1:
            if self.model.training:
                return self.accelerator.gather(values)
            return self.accelerator.gather_for_metrics(values)

        def remove_replicas(tensor):
            # Keep one copy per data-parallel rank before removing the final batch's padding.
            tensor = tensor.reshape(self.accelerator.num_processes, -1, *tensor.shape[1:])[::replicas].flatten(0, 1)
            if (
                not self.model.training
                and self.accelerator.gradient_state.end_of_dataloader
                and self.accelerator.gradient_state.remainder > 0
            ):
                tensor = tensor[: self.accelerator.gradient_state.remainder]
            return tensor

        return recursively_apply(remove_replicas, self.accelerator.gather(values))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.get_tp_size() > 1:
            self.gather_function = self._gather_for_metrics
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        probabilities, mask = self._get_decision_probabilities(outputs, inputs)
        labels = inputs["labels"]
        if labels.ndim == 2:
            labels = labels.masked_fill(~mask, -100)
        probabilities = probabilities.masked_fill(~mask, -100).detach()
        hard_labels = inputs.get("hard_labels")
        rows_per_example = inputs.get("rows_per_example")
        if rows_per_example is not None:
            probabilities, labels, hard_labels = self._pack_predictions(
                probabilities, labels, rows_per_example, hard_labels
            )
        return loss.detach(), probabilities, labels if hard_labels is None else (labels, hard_labels)

    def _pack_predictions(self, probabilities, labels, rows_per_example, hard_labels=None):
        """
        Pack the rows of the batch by record, `(num_records, max_rows, num_options)`, with `-100` in padded slots and
        cells, so that the evaluation gathers whole records across ranks and drops the padded records of the last
        batch. `max_rows` and `num_options` are the largest of all ranks. `hard_labels` (one per row, or `None`) is
        packed the same way, as `(num_records, max_rows)`.
        """
        if "inputs" in (self.args.include_for_metrics or []):
            raise ValueError("include_for_metrics=['inputs'] is not supported with rows_per_example")
        max_rows = int(rows_per_example.max()) if rows_per_example.numel() else 0
        sizes = torch.tensor([max_rows, probabilities.shape[-1]], device=probabilities.device)
        if self.accelerator.num_processes > 1:
            sizes = self.accelerator.gather(sizes[None]).amax(0)
        max_rows, num_options = sizes.tolist()
        probabilities = F.pad(probabilities, (0, num_options - probabilities.shape[-1]), value=-100)
        if labels.ndim == 2:
            labels = F.pad(labels, (0, num_options - labels.shape[-1]), value=-100)
        probabilities = pack_rows(probabilities, rows_per_example, max_rows, -100)
        labels = pack_rows(labels, rows_per_example, max_rows, -100)
        if hard_labels is not None:
            hard_labels = pack_rows(hard_labels, rows_per_example, max_rows, -100)
        # After gathering, the records are flattened back to rows (the padded slots stay, at `-100`), so that batches
        # with different numbers of options concatenate
        gather = self._gather_for_metrics if self.get_tp_size() > 1 else self.accelerator.gather_for_metrics
        self.gather_function = partial(self._gather_record_rows, gather)
        return probabilities, labels, hard_labels

    @staticmethod
    def _gather_record_rows(gather, tensors):
        # The losses are one value per record; the predictions and labels (a tensor or a tuple) are record-shaped
        return recursively_apply(lambda tensor: tensor.flatten(0, 1) if tensor.ndim > 1 else tensor, gather(tensors))

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(numerator for numerator, _ in values) / max(sum(denominator for _, denominator in values), 1)
            for key, values in self._metrics[mode].items()
        }
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _decision_checkpoint_files(self, model, checkpoint: str) -> list[str] | None:
        """
        The safetensors files of `checkpoint` that [`_load_decision_checkpoint`] loads into `model`, or `None` for
        checkpoints left to [`~transformers.Trainer`]: adapters of a PEFT model, DeepSpeed, FSDP, SageMaker, or `.bin`
        weights.
        """
        if (
            self.is_deepspeed_enabled
            or self.is_fsdp_enabled
            or is_sagemaker_mp_enabled()
            or not isinstance(model, PreTrainedModel)
        ):
            return None
        index_file = os.path.join(checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.isfile(os.path.join(checkpoint, SAFE_WEIGHTS_NAME)):
            return [os.path.join(checkpoint, SAFE_WEIGHTS_NAME)]
        if os.path.isfile(index_file):
            with open(index_file) as file:
                return sorted({os.path.join(checkpoint, name) for name in json.load(file)["weight_map"].values()})
        return None

    def _load_decision_checkpoint(self, model, checkpoint: str, files: list[str]) -> None:
        """
        Load the full safetensors checkpoint `checkpoint` (its `files`) into `model` through the model's own weight
        conversions, then check that every weight was restored. [`~transformers.Trainer`] loads the saved state dict as
        it is, without strict matching, so the weights of a model saved in another layout (such as the head of Clef)
        silently keep their current values. As there, tied weights (saved once) and a model's `_keys_to_ignore_on_save`
        (when they are exactly the missing keys, which are then tied again) are not missing.
        """
        state_dict = {}
        for file in files:
            state_dict.update(safetensors.torch.load_file(file, device="cpu"))
        result = model.load_state_dict(_to_model_layout(model, state_dict), strict=False)
        # Tied weights are saved once, and loading their source restores them
        tied = set(getattr(model, "all_tied_weights_keys", None) or {})
        missing = [key for key in result.missing_keys if key not in tied]
        # As `Trainer._issue_warnings_after_load`: the keys a model leaves out when saving are tied again
        ignored_on_save = set(getattr(model, "_keys_to_ignore_on_save", None) or ())
        if missing and ignored_on_save and set(result.missing_keys) == ignored_on_save:
            model.tie_weights()
            missing = []
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"The checkpoint {checkpoint} does not match the model: missing keys {missing}, unexpected keys "
                f"{result.unexpected_keys}"
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = self.model if model is None else model
        files = self._decision_checkpoint_files(model, resume_from_checkpoint)
        if files is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        logger.info(f"Loading model from {resume_from_checkpoint}.")
        self._load_decision_checkpoint(model, resume_from_checkpoint, files)

    def _load_best_model(self):
        checkpoint = self.state.best_model_checkpoint
        files = self._decision_checkpoint_files(self.model, checkpoint)
        if files is None:
            return super()._load_best_model()
        logger.info(f"Loading best model from {checkpoint} (score: {self.state.best_metric}).")
        self._load_decision_checkpoint(self.model, checkpoint, files)

    def _save_checkpoint(self, model, trial):
        model_name = (
            self.args.hub_model_id.split("/")[-1] if self.args.hub_model_id else Path(self.args.output_dir).name
        )
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
