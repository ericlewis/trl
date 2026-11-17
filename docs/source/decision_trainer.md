# DecisionTrainer

`DecisionTrainer` trains decision models using their supervised loss. Transformers owns the architecture, checkpoint format, and objective; TRL handles optimization, distributed training, and metrics. The model returns a mean `loss` and candidate `logits`. Model-specific token layouts and batching live in example collators. The library provides `DataCollatorForDecision` for shared request preparation and target normalization.

## Usage

Pass a configured model and a collator matching its forward inputs:

```python
from trl.experimental.decision import DecisionConfig, DecisionTrainer

trainer = DecisionTrainer(
    model=model,
    args=DecisionConfig(output_dir="decision-output"),
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=collator,
)
trainer.train()
trainer.save_model()
trainer.end()
```

The trainer preserves which parameters are trainable. Freeze the backbone or prepare adapters before constructing it. Alternatively, pass `peft_config` to wrap the whole model through PEFT; configure `modules_to_save` for any head that should remain trainable. Resume with the same model and adapter setup. The Jeva example saves a native base checkpoint before wrapping it with PEFT, so the adapter can be reloaded with its decision head.

## Jeva example

The [BANKING77 example](https://github.com/huggingface/trl/tree/main/examples/decisions) provides Jeva's model construction and a 1,000-step LoRA configuration with a stratified validation split:

```bash
python examples/decisions/jeva_banking77.py --config examples/decisions/jeva_banking77.yaml
```

> [!TIP]
> This example currently requires [Transformers #50614](https://github.com/huggingface/transformers/pull/50614). CLM and JEV use their own Transformers PRs linked below.

Jeva's `--mode` selects head, LoRA, or full training. Its token limits, truncation policy, and option chunk size are example arguments, rather than trainer settings.

Requests use Transformers' `state` / `questions` schema, with one question per training row. The example stores requests as JSON strings to preserve candidate names and order in Arrow datasets. Targets are separate:

| Field | Meaning |
|---|---|
| `target_index` | Zero-based option index, or `None` for a nullable choice. |
| `target_probs` | Null probability, then one probability per option. |
| `p_true` | Probability of true for a `noul` question. |
| `score` | Numeric score, interpolated between rubric levels. |

A row with several questions rendered into one prompt (a template that reads `questions`, such as Clef's) goes through
`prepare_request`, which returns the inputs, the questions in template order and one target per question. Targets come
from `targets` (`{question_name: {"target_index" | "target_probs" | "p_true" | "score": ...}}`), or from `gold` /
`answers`, where each entry is a bare label or `{"label", "probabilities", "noul"}`. `target_from_gold` turns the latter
into `target_probs` in the question's option order. A template that renders one prompt per question is rejected.

Choice questions use a `criteria` mapping and may set `allow_null=True`. Score questions use a `criteria` list and
may supply numeric `values`. The [Jeva example collator](https://github.com/huggingface/trl/blob/main/examples/decisions/jeva_banking77.py)
uses Transformers' template to prepare separate state and option spans. It subclasses `DataCollatorForDecision`,
which delegates request rendering to `model.decision_config.prepare_inputs`. Preprocess once with
`dataset.map(collator.tokenize, remove_columns=dataset.column_names)` so batches only need padding.


## CLM example

The [typed-decisions example](https://github.com/huggingface/trl/tree/main/examples/decisions) trains
CLM projection heads on a frozen SmolLM2-135M encoder. It requires [Transformers #50649](https://github.com/huggingface/transformers/pull/50649).

```bash
python examples/decisions/clm_typed_decisions.py --config examples/decisions/clm_typed_decisions.yaml
```

The [CLM example collator](https://github.com/huggingface/trl/blob/main/examples/decisions/clm_typed_decisions.py) reuses the request schema above and packs each batch into one prompt with a pool of distinct
candidates. Soft targets supervise a bidirectional contrastive loss. Validation holds out source-row IDs before
expanding their questions. `eval_accuracy` uses each question's own candidates; contrastive negatives are local
to each device's batch. Context parallelism is unsupported for this example.

## Clef example

Clef answers every question of a record in one prompt: the model reads the state once and scores the options of all
questions jointly. It requires [Transformers #50058](https://github.com/huggingface/transformers/pull/50058). The
[Clef example](https://github.com/huggingface/trl/blob/main/examples/decisions/clef_typed_decisions.py) trains it on
`LocalLLaMA/typed-decisions`, with KLM-1.6 as the backbone by default:

```bash
python examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
```

The configuration trains the head at `head_learning_rate` (`3e-4`, the backbone at `1e-4`), evaluates with
`compute_calibration_metrics` and keeps the checkpoint with the best `eval_selection_score`.

`mode` selects head-only, LoRA, or full training. LoRA puts adapters on the backbone and trains the head in full; the output is the merged native checkpoint, with the adapter alone in `adapter/`. A Clef checkpoint is loaded as it is; any other model becomes the
backbone of a new head. A causal LM reads its last token with Clef's ChatML prompt; a bidirectional encoder (BERT, ModernBERT, or a reranker, unwrapped to its base model) reads its first token with the plain prompt between `[CLS]` and `[SEP]`. `global_pooling` (`last` or `first`) and `prompt_style` (`chatml` or `plain`) override that choice, and `head_type: cosine` (with `cosine_width` and `cosine_initial_scale`) selects the small cosine head; the script prints the resolved setup at start-up. Each row holds `state`, `questions` and the targets of all of its questions, either `targets`
(per question `target_index`, `target_probs`, `p_true` or `score`) or `gold` / `answers` (per question a label or
probabilities), through `DataCollatorForDecision.prepare_request`. The example collator `DataCollatorForClef` returns
one row of `labels` and `option_mask` per question, with the questions of all records of a batch one after the other.
Nullable questions raise: Clef has no coverage output. A prompt longer than `max_length`, an invalid request, a nullable question or unusable targets raise, or with
`truncation: skip` the record is dropped and counted in the build report the script prints (rows, questions, skipped by reason, truncated). With `permute_fields`, each training batch re-renders its prompts with the questions in a new
random order, each keeping its target; evaluation never does. Activation checkpointing (`gradient_checkpointing: true`) covers the backbone and the head. The collator returns `rows_per_example`, so the
example also runs on several processes:

```bash
torchrun --nproc_per_node=2 \
    examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
```

`label_smoothing`, `brier_weight` and `ordinal_weight` set the terms of Clef's native loss, and `rlcd_weight` adds its
RLCD policy term, adapted from [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0): noisy samples of the option
logits are rewarded by a strictly proper scoring rule, and a REINFORCE term with a per-question mean baseline joins the
supervised loss. `rlcd_sigma`, `rlcd_samples` and `rlcd_normalize_advantage` tune it; each setting left unset keeps the
value of the model's config. The term applies in training only, so `eval_loss` stays the deterministic supervised
loss. Its defaults (sigma 0.4, 32 samples, advantages not divided by their standard deviation; `0.25` is the suggested
weight) are a revised variant of Laya's objective; `rlcd_normalize_advantage: true` with `rlcd_samples: 4` reproduces
Laya's original. These loss terms are a community recipe, not an official Cloudflare objective.

## JEV

JEV uses [Transformers #50586](https://github.com/huggingface/transformers/pull/50586). Use the
[JEV training script](https://github.com/huggingface/trl/blob/main/examples/decisions/jev.py) with the model's tokenizer and `decision_config`, and set both `per_device_train_batch_size` and
`per_device_eval_batch_size` to `1`. Disable gradient checkpointing for the current JEV implementation. Each row contains one question; use accumulation for more decisions per update.
The script prepares packed option targets and uses Transformers' `default_data_collator` for JEV's native KL and ordinal RPS loss. Predictions are padded per-question
probabilities, like the other collators. Nullable decisions are unsupported.

## Labels and metrics

Collators may return integer class labels or soft target distributions. Pass `option_mask` when padding candidate sets. With `allow_null`, target distributions contain null first and the model may return `coverage_logits`. Nullable decisions require a coverage output. `is_choice` can exclude binary and score rows from choice accuracy.

The trainer logs candidate loss, choice and candidate accuracy, Brier score, and entropy. `loss` and `eval_loss` are the native training objective; the reported candidate cross-entropy can differ from it. Null and coverage metrics are logged when present; auxiliary loss is logged only when returned by the model. Metrics aggregate across ranks and accumulation steps, and evaluation removes padded examples. `compute_metrics` receives predicted probabilities and the original labels, with `-100` marking padded candidate columns.

A collator whose records each produce several rows (such as one per question) returns `rows_per_example`, a
`LongTensor` with the number of consecutive rows of each record, adding up to the number of rows. The trainer removes
it before calling the model, and packs every per-row tensor it gathers across processes by record, so that each rank
sends the same shape and the padding of the last evaluation batch is removed in records. `compute_metrics` then
receives, per record, `max_rows` question slots in order (`max_rows` is the largest number of rows of a record in that
evaluation batch, over all processes), flattened to `(num_slots, num_options)`: the slots a record does not use are
`-100` in every cell, and dropping them gives the rows in record order, then row order. Without `rows_per_example`,
predictions keep one row per example.

`compute_calibration_metrics` can be passed as `compute_metrics` as it is. It reports, per question, `accuracy`, `kl`
(from the target distribution), `brier`, `ece`, `mean_confidence`, `soft_nll`, `confidence_gap` (mean confidence minus
accuracy) and `selection_score` (`accuracy - kl_weight * kl - brier_weight * brier`, with both weights `0.25` by
default), a single number to select checkpoints with `metric_for_best_model: eval_selection_score`.

## Head learning rate

Set `head_learning_rate` to train the parameters outside the model's backbone (its `base_model`), such as a decision
head, at their own rate while the backbone keeps `learning_rate`. Adapters inside the backbone train at
`learning_rate`. Zero-dimensional parameters, such as logit scales and gates, are never decayed.

## Distributed training

DDP, DeepSpeed ZeRO, and FSDP2 use the usual Trainer configuration. FSDP2 currently requires Accelerate main,
included in the example script dependencies. Context and sequence parallelism are unsupported: the current model
readouts need complete sequences. Tensor and masked expert parallelism use Transformers' `distributed_config`
when loading a complete checkpoint. Token-dispatch expert parallelism is not supported by Trainer.

The trainer normalizes the native mean loss by decision count across ranks and accumulation steps. It does not
add MoE router regularization; auxiliary losses must come from the model itself. CLM's contrastive negatives remain
local to each microbatch.

## Inference

Use the model's Transformers inference API. For the LoRA example, load the saved native `base_model` and its adapter with PEFT, then call `merge_and_unload()` before creating the decision pipeline. Saved Jeva checkpoints work with the decision pipeline:

```python
from transformers import AutoModelForDecision, AutoTokenizer, pipeline

model = AutoModelForDecision.from_pretrained("decision-output")
tokenizer = AutoTokenizer.from_pretrained("decision-output")
decide = pipeline("decision", model=model, tokenizer=tokenizer)
answers = decide(request)
```

## DecisionConfig

[[autodoc]] experimental.decision.DecisionConfig

## DecisionTrainer

[[autodoc]] experimental.decision.DecisionTrainer

## DataCollatorForDecision

[[autodoc]] experimental.decision.DataCollatorForDecision
    - prepare_example
    - prepare_request

## compute_calibration_metrics

[[autodoc]] experimental.decision.compute_calibration_metrics
