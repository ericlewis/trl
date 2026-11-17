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

`mode` selects `head`, `lora` or `full` training (LoRA puts adapters on the backbone and trains the head in full; the output is the merged native checkpoint, with the adapter alone in `adapter/`); a Clef checkpoint is loaded as it is and any other causal LM becomes
a new head's backbone. `max_length` with `truncation: error|skip` handles over-length prompts, and `permute_fields`
shuffles the question order of training batches only. The script prints a build report of rows, questions and skipped
records. Nullable questions and activation checkpointing are unsupported. The collator returns `rows_per_example`, the
number of questions of each record, so `DecisionTrainer` gathers whole records and the example runs on several
processes:

```bash
torchrun --nproc_per_node=2 \
    examples/decisions/clef_typed_decisions.py --config examples/decisions/clef_typed_decisions.yaml
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
