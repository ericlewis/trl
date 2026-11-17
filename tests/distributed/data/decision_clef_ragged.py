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

"""
Evaluate, then train, a tiny Clef on 9 records with different numbers of questions, on CPU. Launched by
`tests/distributed/test_distributed.py` with one and with two processes: `python decision_clef_ragged.py OUTPUT_DIR`
under `torchrun`. Rank 0 writes the metrics of the evaluation before training to `OUTPUT_DIR/metrics.json`.
"""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from datasets import Dataset  # noqa: E402
from examples.decisions.clef_typed_decisions import DataCollatorForClef, clef_decision_config  # noqa: E402
from transformers import AutoTokenizer, ClefConfig, ClefForDecision, Qwen2Config, set_seed  # noqa: E402

from trl.experimental.decision import DecisionConfig, DecisionTrainer, compute_calibration_metrics  # noqa: E402


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None, "alpha": None, "beta": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}
# The questions of each record: one, two or three, in an irregular order
NAMES = [
    ("refund", "team", "urgency"),
    ("team",),
    ("urgency", "refund"),
    ("refund",),
    ("refund", "team", "urgency"),
    ("team", "urgency"),
    ("urgency",),
    ("refund", "team"),
    ("team", "refund", "urgency"),
]


def make_row(index, names):
    gold = {"refund": index % 2 == 0, "team": ("alpha", "zeta", "beta")[index % 3], "urgency": index % 3}
    return {
        "state": f"ticket {index}: {'refund' if index % 2 == 0 else 'thanks'} {gold['team']} {'!' * (index % 4)}",
        "questions": {name: QUESTIONS[name] for name in names},
        "gold": {name: gold[name] for name in names},
    }


def tiny_clef(tokenizer):
    set_seed(0)
    text_config = Qwen2Config(
        vocab_size=len(tokenizer) + 2,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config = ClefConfig(
        text_config=text_config,
        backbone_hidden_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_evidence_layers=1,
        num_attention_heads=2,
    )
    model = ClefForDecision(config)
    model.decision_config = clef_decision_config()
    return model


def compute_metrics(prediction):
    # Question slots that are -100 in every cell are padding of the record-shaped layout
    predictions = np.asarray(prediction.predictions)
    questions = int((predictions != -100).any(-1).sum())
    return {**compute_calibration_metrics(prediction), "questions": questions}


def main(output_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForClef(tokenizer, clef_decision_config())
    records = Dataset.from_list([collator.tokenize(make_row(index, names)) for index, names in enumerate(NAMES)])
    args = DecisionConfig(
        output_dir=output_dir,
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        max_steps=3,
        learning_rate=1e-3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    args.accelerator_config.dispatch_batches = False
    args.accelerator_config.split_batches = False
    trainer = DecisionTrainer(
        model=tiny_clef(tokenizer),
        args=args,
        train_dataset=records,
        eval_dataset=records,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate()
    trainer.train()
    if trainer.accelerator.is_main_process:
        metrics["num_processes"] = trainer.accelerator.num_processes
        metrics["train_steps"] = trainer.state.global_step
        Path(output_dir, "metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main(sys.argv[1])
