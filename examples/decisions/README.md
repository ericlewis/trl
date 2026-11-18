# Decision models

These examples train Transformers' native decision models with TRL's experimental `DecisionTrainer`. Request
preparation uses the shared `DataCollatorForDecision` base and `DecisionConfig.prepare_inputs`; Jeva, CLM and Clef
define their model-specific collators inside their training scripts.

| Example | Trains | Needs |
|---|---|---|
| [`jeva_banking77.py`](jeva_banking77.py) | A Jeva head and encoder LoRA on BANKING77 | [Transformers #50614](https://github.com/huggingface/transformers/pull/50614) |
| [`clm_typed_decisions.py`](clm_typed_decisions.py) | CLM projection heads on a frozen encoder | [Transformers #50649](https://github.com/huggingface/transformers/pull/50649) |
| [`jev.py`](jev.py) | JEV on prepared decision requests | [Transformers #50586](https://github.com/huggingface/transformers/pull/50586) |
| [`clef_typed_decisions.py`](clef_typed_decisions.py) | A Clef head with its backbone on typed decisions | A Transformers build with Clef training support, see [below](#clef-typed-decisions) |

## Jeva: BANKING77

Requires [Transformers #50614](https://github.com/huggingface/transformers/pull/50614).
The supplied config trains a Jeva head and encoder LoRA on BANKING77 with a stratified validation holdout.

```bash
python examples/decisions/jeva_banking77.py --config examples/decisions/jeva_banking77.yaml
```

`--mode` selects head, LoRA, or full training. LoRA saves a native `base_model` beside the adapter for reloads.

## CLM: typed decisions

Requires [Transformers #50649](https://github.com/huggingface/transformers/pull/50649).
Trains projection heads on a frozen SmolLM2-135M encoder using `LocalLLaMA/typed-decisions`.

```bash
python examples/decisions/clm_typed_decisions.py --config examples/decisions/clm_typed_decisions.yaml
```

The collator deduplicates candidates within each batch and builds soft targets for native bidirectional InfoNCE.
Validation holds out source-row IDs before expanding their questions. Negatives remain local to each microbatch;
gradient accumulation does not enlarge the pool.

## JEV: prepared decision requests

Requires [Transformers #50586](https://github.com/huggingface/transformers/pull/50586).
Each dataset row contains one question in `request` (JSON), or `state` / `questions`, plus one of `target_index`,
`target_probs`, `p_true`, or `score`. `target_probs` contains a zero null entry followed by option probabilities.
The script prepares native model fields and uses Transformers' `default_data_collator`.

```bash
python examples/decisions/jev.py --model_name_or_path autotrust/JEV-9B \
    --dataset_name your/decision-requests --output_dir output/jev \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_checkpointing false
```

Use batch size one and accumulation for more decisions per update. The model computes KL and ordinal RPS;
nullable decisions and activation checkpointing are unsupported by the current JEV model.

## Clef: typed decisions

Trains a Clef decision model: a head on a backbone (KLM-1.6 by default) that answers every question of a record
from one prompt. Each row needs a `state`, its `questions`, and the target of every question (`gold`, `answers` or
`targets`). The default dataset is `LocalLLaMA/typed-decisions`.

> Needs a Transformers build with Clef training support (Clef's native loss, batched forward, head choice and RLCD)
> on top of [Transformers #50058](https://github.com/huggingface/transformers/pull/50058), which adds Clef without
> them.

### Run it

```bash
# one GPU
python examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml

# several GPUs
torchrun --nproc_per_node=2 \
    examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
```

Or open [`clef_decision_training.ipynb`](clef_decision_training.ipynb). From one settings cell it builds its own
virtual environment (a known torch for the machine's GPU, then pinned builds), trains with `accelerate launch` on
one GPU or several, continues with an RLCD phase, shows the metrics of both runs and asks the trained model
questions. Its `QUICK` setting runs every stage on a small sample first.

When it starts, the script prints the setup it resolved and a build report for each split: rows, questions, records
skipped (by reason) and records whose state was cut. The output folder is a complete model that
`pipeline("decision")` loads as it is.

### What the supplied configuration does

| Setting | Value | What it means |
|---|---|---|
| `head_type` | `cosine` | The small cosine head: two projections score each option by its cosine with its question. `joint` selects Clef's original head, which is also what the script builds when no `head_type` is given. |
| `mode` | `full` | Trains the backbone and the head. `head` freezes the backbone. `lora` puts adapters on the backbone and trains the head in full; it saves the merged model, with the adapter alone in `adapter/`. |
| `learning_rate`, `head_learning_rate` | `1e-4`, `3e-4` | The head is new, so it trains faster than the backbone. `head_learning_rate` is a setting of the script, as in the Jeva example, not of `DecisionConfig`: the script gives the trainer its parameter groups, the backbone (with any LoRA adapters) at `learning_rate` and the head at `head_learning_rate`, with zero-dimensional parameters never decayed. Unset, the trainer's default optimizer is used. |
| `dtype`, `bf16` | `float32`, `true` | float32 weights under bf16 autocast, which KLM's fused CUDA kernels need. |
| `truncation` | `state` | See [Long prompts](#long-prompts). |
| `permute_fields` | `true` | Training batches reorder the questions of each record. Evaluation never does. |
| `eval_steps`, `save_strategy` | `12`, `best` | Evaluates three times per epoch of the default dataset with `compute_calibration_metrics`, saves a checkpoint whenever `eval_selection_score` improves, and restores the best one at the end. |

### Choosing the model

`model_name_or_path` is one of two things:

- **A Clef checkpoint.** It is loaded as it is and keeps the head it was saved with, so `head_type` and the head
  sizes must match it or be left out.
- **Any other model.** It becomes the backbone of a new head. A causal LM (KLM-1.6, Qwen) reads its last token with
  Clef's ChatML prompt; an encoder (BERT, ModernBERT, UltraBERT, a reranker) reads its first token with a plain
  prompt. `global_pooling` (`last` or `first`) and `prompt_style` (`chatml` or `plain`) override that choice.

Backbones are loaded strictly:

- Backbone weights are never newly initialised. A checkpoint's own code runs only with `trust_remote_code: true` (see
  [Backbones that need their own code](#backbones-that-need-their-own-code)).
- A package that keeps its backbone under one prefix (a reranker with `backbone.<name>` weights next to its task
  head) has the backbone read from that prefix and its other tensors dropped. The start-up line says which
  (`backbone weights: prefix "backbone."` or `direct`). Missing weights, or several candidate prefixes, raise.
- A remote-code `config.json` that lists `attention_windows` or `rope_theta` per layer must keep one value per kind
  of layer; any other list raises rather than change the architecture.

`attn_implementation` (for example a flash-attention kernel) is applied to the backbone only, for a new head and for
a checkpoint alike. Every `model_init_kwargs` entry reaches the `from_pretrained` of the backbone of a new head, which
the model is built around as it was loaded. For a Clef checkpoint, an entry that sets a value of its backbone's saved
config must match it: the saved value is used, and changing it raises.

### Backbones that need their own code

A model that exists only as code in its repository (an `auto_map` in its `config.json` for a model type that
Transformers does not have) needs `trust_remote_code: true`, in the config file or on the command line
(`--trust_remote_code true`). It is off by default: such a model is refused with a message that names it, before any of
its code is imported, and nothing ever prompts. Models of Transformers, reranker packages included, are loaded with
Transformers' own classes with or without the setting. `trust_remote_code` in `model_init_kwargs` must agree with it.

With the setting, the backbone is the package's `AutoModel` when its `auto_map` has one, else its
`AutoModelForCausalLM`, used itself as the backbone. A package with a causal LM and no masked LM is causal (`last`
pooling, ChatML prompt). Its weights load strictly, without the prefix repair above. The width of its hidden states is
read from the model, and when it returns no hidden states the submodule that gives them is found and shown on the
start-up line (`backbone_hidden_module=...`). A backbone that does not support activation checkpointing stops at
start-up when `activation_checkpointing` (or `gradient_checkpointing`) is on.

The saved model, with the package's code next to it when the backbone came from a local folder, loads again only with
the flag: `AutoModelForDecision.from_pretrained(output_dir, trust_remote_code=True)` or
`pipeline("decision", model=output_dir, trust_remote_code=True)`, and so does a second run of the example from it (the
RLCD phase) with `trust_remote_code: true`.

A context the package enforces can be raised where its config allows it, through `model_init_kwargs`, for example
`model_init_kwargs: {max_seq_len: 2048, max_position_embeddings: 2048}`. The values are saved in the model's
`text_config`, so the model loaded again has the same context. When `max_length` is above the backbone's
`max_position_embeddings` and it has no learned position table, the start-up prints a warning rather than an error,
since rotary models run past that value.

### Long prompts

`truncation` decides what happens to a record whose prompt is longer than `max_length`:

| `truncation` | A record that is too long |
|---|---|
| `state` | Has the end of its state cut, so the prompt is exactly `max_length` tokens. Questions and options are never cut. A record too long even without its state is skipped. |
| `skip` | Is dropped. |
| `error` | Stops the run. |

`state` and `skip` also skip invalid records. If no record is left, the run stops and says why. A prompt whose
questions were reordered by `permute_fields` is cut the same way.

### Sampling

- `max_train_samples` draws a random sample of the training split with `seed`.
- The evaluation holdout (`validation_fraction`) and `max_eval_samples` are drawn with `eval_sample_seed` (0 by
  default), so runs with different seeds evaluate on the same records.
- A sample is never the first rows of a split, which are often not representative.
- Samples are drawn before tokenization, so skipped records can leave fewer than asked for. The build report shows
  how many.

### RLCD

`rlcd_weight` adds Laya's RLCD ("Reinforcement Learning for Calibrated Decisions"; a community recipe, not an
official Cloudflare objective): a policy term on noisy samples of the option logits, added to the supervised loss.
It is a second phase after cross-entropy training. It applies in training only, so `eval_loss` is always the
supervised loss.

**Two runs (recommended).** Train with cross-entropy, then start the example again from the saved checkpoint:

```yaml
model_name_or_path: output/clef-typed-decisions      # the cross-entropy run's output
output_dir: output/clef-typed-decisions-rlcd
rlcd_weight: 0.25
rlcd_sigmas: [0.4, 0.3, 0.2, 0.1]                    # noise of each epoch; the last value is held
```

The second run starts from the checkpoint's weights with a fresh optimizer, so do not pass
`resume_from_checkpoint`. Its first evaluation reproduces the first run's final `eval_loss`.

**One run.** Cross-entropy until `rlcd_start_epoch`, then RLCD:

```yaml
num_train_epochs: 6
rlcd_weight: 0.25
rlcd_start_epoch: 2
rlcd_sigmas: [0.4, 0.3, 0.2, 0.1]
```

- `rlcd_weight` must be positive, and `rlcd_start_epoch` must fall before the end of the run.
- `rlcd_sigma` (one fixed noise scale), `rlcd_samples` and `rlcd_normalize_advantage` tune the term; `rlcd_sigma`
  cannot be combined with `rlcd_sigmas`.
- The schedule follows the epoch alone, so resuming and several processes agree. The saved config keeps the weight
  and noise in effect at the last step (weight 0 if training stopped before `rlcd_start_epoch`).
- On a new head, `rlcd_weight` without `rlcd_start_epoch` or `rlcd_sigmas` runs RLCD from the first step and logs a
  warning.

### Timing training steps

[`benchmark_clef.py`](benchmark_clef.py) times this training path. It builds the model, data and trainer with the
example's own `build_trainer`, trains `--steps` steps without evaluating or saving, evaluates once, and prints a
JSON summary (also written to `--output`).

```bash
# a synthetic recipe: a tiny random backbone and 256 generated records
python examples/decisions/benchmark_clef.py --steps 120

# the example's configuration, parsed exactly as the example parses it
python examples/decisions/benchmark_clef.py --config examples/decisions/clef_typed_decisions.yaml --steps 320 \
    --output output/clef-benchmark.json
```

With `--config`, `dataloader_drop_last` is forced on so that every step holds the same number of records. Every
timing boundary synchronises the CUDA device.

| Field | Meaning |
|---|---|
| `window_records_per_second` | Records per second over steps 21 to 320 (`window_steps` of them). |
| `steady_step_p50`, `steady_step_p95`, `steady_records_per_second` | Step time, from `on_step_begin` to `on_step_end`, after the first `--warmup_steps_excluded` (20) steps. |
| `steady_data_p50`, `steady_records_per_second_with_data` | The time between steps: logging and collation, which includes re-rendering reordered prompts. |
| `first_step_seconds` | From `on_train_begin` to the end of step 1. |
| `build_seconds`, `startup_seconds` | `build_trainer` alone; process start to just before `trainer.train()`. |
| `trainer_wall_seconds`, `eval_seconds` | `trainer.train()`, which here never evaluates or saves; one evaluation after training. |
| `peak_allocated_gib`, `peak_reserved_gib` | CUDA memory during training (not loading, nor the final evaluation). |
| `head_parameters` | The decision head alone. The `lm_head` table of the joint head's lexical prior is `recipe.lexical_prior_parameters`. |
| `steps` | Per step: `seconds` and `data_seconds`. |
| `recipe` | The resolved settings, the backbone and its dtypes, the attention implementation the backbone uses, activation checkpointing, head type and parameter counts. |

`startup_seconds` and `trainer_wall_seconds` are particular to this script; the summary's `notes` say what they
cover. Under `torchrun` only the main process writes the summary.

### Limits

- Nullable questions are not supported: Clef has no coverage output.
- `activation_checkpointing: true` covers the backbone and the head.
