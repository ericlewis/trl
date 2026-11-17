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
Tiny Clef models, records and trainers shared by the decision test modules, so that neither imports the other. Import
it only when `HAS_CLEF_TRAINING` holds: it needs a Transformers build with Clef training support.
"""

from datasets import Dataset
from examples.decisions.clef_typed_decisions import DataCollatorForClef
from transformers import BertConfig, ClefConfig, ClefForDecision, ModernBertConfig, Qwen2Config, set_seed

from trl.experimental.decision import DecisionConfig, DecisionTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None, "alpha": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}


def make_row(index, names=("refund", "team", "urgency")):
    gold = {"refund": index % 2 == 0, "team": "alpha" if index % 3 == 0 else "zeta", "urgency": index % 3}
    return {
        "state": f"ticket {index}: {'refund' if index % 2 == 0 else 'thanks'} {'alpha' if index % 3 == 0 else 'zeta'}",
        "questions": {name: QUESTIONS[name] for name in names},
        "gold": {name: gold[name] for name in names},
    }


def tiny_clef(tokenizer, decision_config, backbone="qwen2", head_type="joint"):
    """
    A tiny Clef on a tiny causal (`qwen2`) or bidirectional (`bert`, `modernbert`) backbone. Encoders read the first
    token and keep the plain prompt of their decision config when `decision_config` is `None`.
    """
    set_seed(0)
    sizes = {"vocab_size": len(tokenizer) + 2, "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2}
    if backbone == "qwen2":
        text_config = Qwen2Config(**sizes, num_attention_heads=2, num_key_value_heads=1)
    elif backbone == "bert":
        text_config = BertConfig(**sizes, num_attention_heads=2)
    else:
        text_config = ModernBertConfig(**sizes, num_attention_heads=2, pad_token_id=tokenizer.pad_token_id)
    encoder = {} if backbone == "qwen2" else {"global_pooling": "first", "prompt_style": "plain"}
    config = ClefConfig(
        text_config=text_config,
        backbone_hidden_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_evidence_layers=1,
        num_attention_heads=2,
        head_type=head_type,
        cosine_width=16,
        **encoder,
    )
    model = ClefForDecision(config)
    if decision_config is not None:
        model.decision_config = decision_config
    return model


def make_trainer(tmp_dir, tokenizer, decision_config, **kwargs):
    collator = DataCollatorForClef(tokenizer, decision_config)
    train = Dataset.from_list([collator.tokenize(make_row(index)) for index in range(16)])
    val = Dataset.from_list([collator.tokenize(make_row(index)) for index in range(16, 24)])
    args = dict(
        output_dir=tmp_dir,
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        max_steps=20,
        learning_rate=1e-3,
        eval_strategy="steps",
        eval_steps=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        report_to="none",
    )
    peft_config = kwargs.pop("peft_config", None)
    compute_metrics = kwargs.pop("compute_metrics", None)
    model = kwargs.pop("model", None) or tiny_clef(tokenizer, decision_config)
    trainer = DecisionTrainer(
        model=model,
        args=DecisionConfig(**{**args, **kwargs}),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        peft_config=peft_config,
    )
    return trainer, collator, val


def lora_clef(tokenizer, decision_config):
    """The tiny Clef with LoRA adapters on its backbone and the head trained in full, as in the example's lora mode."""
    from peft import LoraConfig, get_peft_model

    model = tiny_clef(tokenizer, decision_config)
    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    lora_config = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION")
    model.model = get_peft_model(model.model, lora_config)
    return model
