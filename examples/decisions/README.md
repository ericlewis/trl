# Decision models

These examples use Transformers' native decision models and losses with TRL's `DecisionTrainer`.
Request preparation uses the shared `DataCollatorForDecision` base and `DecisionConfig.prepare_inputs`.
Jeva, CLM and Clef define their model-specific collators inside their training scripts.

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

## Clef: typed decisions

Requires [Transformers #50058](https://github.com/huggingface/transformers/pull/50058).
Trains a Clef head, with its KLM-1.6 backbone, on `LocalLLaMA/typed-decisions`. One prompt carries all questions of a
record, so each row needs `state`, `questions` and the targets of every question (`targets`, `gold` or `answers`).

```bash
python examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
```

The configuration trains the head at `head_learning_rate` (`3e-4`, the backbone at `1e-4`), evaluates with
`compute_calibration_metrics` and keeps the checkpoint with the best `eval_selection_score`.

`mode` selects `head`, `lora` or `full` training (LoRA puts adapters on the backbone and trains the head in full; the output is the merged native checkpoint, with the adapter alone in `adapter/`); a Clef checkpoint is loaded as it is and any other causal LM or encoder (BERT, ModernBERT, UltraBERT, a reranker)
becomes a new head's backbone. Backbone weights are never newly initialised and remote code is never run: when a
checkpoint stores its backbone under a prefix (the UltraBERT reranker package keeps `backbone.<name>` next to its
`pair_head`), the backbone is read from its safetensors under the one prefix that holds every weight with its shape,
the other tensors are dropped, and the start-up line says `backbone weights: prefix "backbone."` (or `direct`) with
what was dropped; missing weights or several candidate prefixes raise instead. `truncation` decides what happens to a
prompt longer than `max_length`: `state` (the configuration's choice, as Clef itself does) cuts the end of the record's
state so the prompt is exactly `max_length` tokens, never touching its questions or options, and skips a record that
is too long even without its state; `skip` drops over-length records; `error` stops at the first one. `skip` and
`state` also skip invalid records. `permute_fields` shuffles the question order of training batches only (a permuted
prompt is cut the same way). The script prints a build report of rows, questions, skipped records by reason, and
records whose state was cut (`truncated`). `max_train_samples` limits the training split to a random sample drawn
with `seed`; the evaluation holdout and `max_eval_samples` are drawn with `eval_sample_seed` (0 by default), so runs
with different seeds evaluate on the same records. Never the first rows, which are often not representative. Samples
are drawn before tokenization, so skipped records can leave fewer than asked for; the build report shows how many. A
remote-code package whose `config.json` lists `attention_windows` or `rope_theta` per layer must keep one value per
kind of layer, since Transformers' UltraBERT config reads them that way; other lists raise rather than silently change
the architecture. The configuration keeps the weights in float32 and trains in bf16 autocast, since KLM's fused CUDA
kernels only run on float32 parameters; `attn_implementation` selects flash attention. Nullable questions are
unsupported; activation checkpointing (`gradient_checkpointing: true`) covers the backbone and the head. The collator returns `rows_per_example`, the
number of questions of each record, so `DecisionTrainer` gathers whole records and the example runs on several
processes:

```bash
torchrun --nproc_per_node=2 \
    examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
```

### RLCD

`rlcd_weight` adds Laya's RLCD ("Reinforcement Learning for Calibrated Decisions", a community recipe, not an official
Cloudflare objective): a policy term on noisy samples of the option logits, joined to the supervised loss. It is
normally run as a second phase after cross-entropy training, and applies in training only, so `eval_loss` is always the
supervised loss. Two ways to run it:

1. **Two runs (recommended).** Train with cross-entropy, then start the example again with `model_name_or_path` set to
   the saved Clef checkpoint and `rlcd_weight` set (and `rlcd_sigma`, `rlcd_samples`, `rlcd_normalize_advantage` if
   wanted). It starts from the checkpoint's weights with a fresh optimizer, so do not pass `resume_from_checkpoint`;
   its first evaluation reproduces the first run's final `eval_loss`. The paired RLCD experiments used a fresh
   optimizer for the RLCD phase.
2. **One run.** `rlcd_weight` with `rlcd_start_epoch` and/or `rlcd_sigmas`: cross-entropy until `rlcd_start_epoch`, then
   RLCD with `rlcd_weight`, and the noise scale annealed per epoch of the phase (`rlcd_sigmas: [0.4, 0.3, 0.2, 0.1]` is
   Laya's schedule; the last value is held). The weight must be positive; the model is built with weight 0 and the
   schedule follows the epoch alone, so resuming and several processes agree. The saved config keeps the weight and
   sigma in effect at the last step (weight 0 if training stopped before `rlcd_start_epoch`).

`benchmark_clef.py` times this training path: it builds the model, data and trainer with the example's own
`build_trainer`, trains `--steps` steps without evaluating or saving, evaluates once, and prints a JSON summary (also
written to `--output`; under torchrun by the main process only). Without `--config` it runs a synthetic recipe (a tiny
random Qwen2 backbone, 256 generated records; CUDA if available, else CPU); with `--config` it takes the example's YAML
as it is. Its fields match the baseline benchmark's `summary.json`; every timing boundary synchronises the CUDA device:

- `startup_seconds`: process start to just before `trainer.train()`; `build_seconds`: `build_trainer` alone;
  `trainer_wall_seconds`: `trainer.train()`; `eval_seconds`: one evaluation after training.
- `steps`: per step, `seconds` from `on_step_begin` to `on_step_end` (the trainer collates a step's batches before
  `on_step_begin`) and `data_seconds` from the previous step's end to `on_step_begin` (logging, saving or evaluation,
  then collation). `first_step_seconds` runs from `on_train_begin` to the end of step 1.
- Steps after the first `--warmup_steps_excluded` (20) are steady: `steady_step_p50` / `steady_step_p95` and
  `steady_records_per_second` use their `seconds`; `window_records_per_second` the same up to step 320 (steps 21 to
  320 by default, `window_steps` of them); `steady_data_p50` and `steady_records_per_second_with_data` show the cost
  of collation, which includes re-rendering permuted prompts.
- `head_parameters`: the decision head alone (the `lm_head` table of the lexical prior is
  `recipe.lexical_prior_parameters`). `peak_allocated_gib`, `peak_reserved_gib`: CUDA only, reset just before
  training and read at every step end, so they cover training but not loading nor the final evaluation.
- `recipe`: the resolved settings, backbone class and floating-point dtypes, attention implementation, activation
  checkpointing, head type, parameter counts and KLM's `_use_fla_kernels`.

```bash
python examples/decisions/benchmark_clef.py --steps 120
python examples/decisions/benchmark_clef.py --config examples/decisions/clef_typed_decisions.yaml --steps 320 \
    --output output/clef-benchmark.json
```

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
