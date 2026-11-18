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
Tiny Clef models, backbones, model packages, records and trainers shared by the decision test modules, so that none
imports another. Import it only when `HAS_CLEF_TRAINING` holds: it needs a Transformers build with Clef training
support.
"""

import copy
import json
import tempfile
from pathlib import Path

import torch
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


def tiny_encoder(tokenizer, kind):
    """A tiny BERT (`bert`), BERT reranker (`bert-reranker`) or ModernBERT (`modernbert`) with the tokenizer's vocabulary."""
    from transformers import BertForSequenceClassification, BertModel, ModernBertModel

    set_seed(0)
    sizes = {"vocab_size": len(tokenizer) + 2, "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2}
    if kind == "modernbert":
        return ModernBertModel(ModernBertConfig(**sizes, num_attention_heads=2, pad_token_id=tokenizer.pad_token_id))
    config = BertConfig(**sizes, num_attention_heads=2)
    return BertForSequenceClassification(config) if kind == "bert-reranker" else BertModel(config)


def ultrabert_tokenizer(tokenizer):
    """The tokenizer made like UltraBERT's: no `[CLS]`; padding, beginning and end of sequence one token; a separator."""
    tokenizer = copy.deepcopy(tokenizer)
    end = "<|endoftext|>"
    tokenizer.add_special_tokens({"bos_token": end, "eos_token": end, "pad_token": end, "sep_token": "<|sep|>"})
    assert tokenizer.cls_token is None
    return tokenizer


def tiny_ultrabert(tokenizer):
    """
    A tiny UltraBERT: two local layers of radius 8 and a global one, each kind with its own RoPE base. The bases are
    not the config's defaults, so a config read without them would differ.
    """
    from transformers import UltrabertConfig, UltrabertModel

    set_seed(0)
    config = UltrabertConfig(
        vocab_size=len(tokenizer) + 2,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=8,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 1234.0},
            "full_attention": {"rope_type": "default", "rope_theta": 98765.0},
        },
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return UltrabertModel(config)


def prefixed(backbone, prefix, keys=None):
    """The tensors of `backbone` (or of its `keys`) under `prefix`, as a wrapper stores them."""
    state = backbone.state_dict()
    return {f"{prefix}{key}": state[key].clone().contiguous() for key in (state if keys is None else keys)}


PAIR_HEAD = {
    "pair_head.0.weight": torch.full((16, 32), 0.5),
    "pair_head.0.bias": torch.zeros(16),
    "pair_head.2.weight": torch.ones(1, 16),
}


def reranker_package(folder, backbone, tokenizer, tensors, **config_changes):
    """
    A folder laid out like the UltraBERT reranker package: its config as its remote code writes it (per-layer
    `attention_windows` and `rope_theta`, the `UltrabertForReranking` architecture, an `auto_map` to code files that
    are not there) with `config_changes`, `tensors` as its `model.safetensors`, and the tokenizer.
    """
    from safetensors.torch import save_file

    Path(folder).mkdir(parents=True)
    config = backbone.config.to_dict()
    layer_types, window = config.pop("layer_types"), config.pop("sliding_window")
    rope_parameters = config.pop("rope_parameters")
    config.update(
        architectures=["UltrabertForReranking"],
        auto_map={
            "AutoConfig": "configuration_ultrabert.UltrabertConfig",
            "AutoModelForSequenceClassification": "modeling_reranker.UltrabertForReranking",
        },
        attention_windows=[window if kind == "sliding_attention" else None for kind in layer_types],
        rope_theta=[rope_parameters[kind]["rope_theta"] for kind in layer_types],
    )
    config.update(config_changes)
    Path(folder, "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, f"{folder}/model.safetensors", metadata={"format": "pt"})
    tokenizer.save_pretrained(folder)
    return folder


def encoder_tokenizer(tokenizer):
    """The tokenizer with a `[CLS]` and a `[SEP]`, which the plain prompt of encoders starts and ends with."""
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})
    return tokenizer


# A package like `MinimaLabs/min-spark-1.7`: a causal language model and nothing else in its `auto_map` (no `AutoModel`),
# whose inner model is a plain `nn.Module`, whose `forward` takes no `**kwargs` nor `position_ids` and never returns
# hidden states (they are the output of `model.final_norm`), whose config has `dim` and no `hidden_size`, and whose
# context (`max_seq_len`) is enforced with a `ValueError` and can be raised by a config value at load
SPARK_PACKAGE = {
    # docstyle-ignore
    "configuration_tinyspark.py": """
        from transformers import PretrainedConfig


        class TinySparkConfig(PretrainedConfig):
            model_type = "tinyspark"

            def __init__(
                self, vocab_size=32, dim=24, n_layers=2, max_seq_len=256, max_position_embeddings=256, **kwargs
            ):
                self.vocab_size = vocab_size
                self.dim = dim
                self.n_layers = n_layers
                self.max_seq_len = max_seq_len
                self.max_position_embeddings = max_position_embeddings
                self.tie_word_embeddings = True
                super().__init__(**kwargs)
    """,
    # docstyle-ignore
    "core.py": """
        import torch
        from torch import nn


        class Core(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed = nn.Embedding(config.vocab_size, config.dim)
                self.layers = nn.ModuleList(nn.Linear(config.dim, config.dim) for _ in range(config.n_layers))
                self.final_norm = nn.LayerNorm(config.dim)

            def forward(self, tokens, return_hidden=False):
                x = self.embed(tokens)
                length = tokens.shape[1]
                causal = torch.ones(length, length, device=x.device).tril()
                causal = causal / causal.sum(-1, keepdim=True)  # the mean of the tokens so far
                for layer in self.layers:
                    x = x + torch.tanh(layer(causal @ x))
                x = self.final_norm(x)
                return x if return_hidden else nn.functional.linear(x, self.embed.weight)
    """,
    # docstyle-ignore
    "modeling_tinyspark.py": """
        from torch import nn
        from transformers import PreTrainedModel
        from transformers.generation import GenerationMixin
        from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

        from .configuration_tinyspark import TinySparkConfig
        from .core import Core


        class TinySparkPreTrainedModel(PreTrainedModel):
            config_class = TinySparkConfig
            base_model_prefix = "model"
            supports_gradient_checkpointing = False
            _no_split_modules = []

            def __init__(self, config):
                super().__init__(config)
                self.model = Core(config)
                self.post_init()

            def _init_weights(self, module):
                pass

            def get_input_embeddings(self):
                return self.model.embed

            def set_input_embeddings(self, value):
                self.model.embed = value

            def check(self, input_ids, attention_mask):
                if input_ids.shape[1] > self.config.max_seq_len:
                    raise ValueError(f"sequence of {input_ids.shape[1]} tokens exceeds max_seq_len")
                if attention_mask is not None and (~attention_mask.bool()[:, 0]).any():
                    raise ValueError("left-padded batches are not supported")


        class TinySparkForCausalLM(TinySparkPreTrainedModel, GenerationMixin):
            def get_output_embeddings(self):
                return self.model.embed

            def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None, return_dict=True):
                self.check(input_ids, attention_mask)
                logits = self.model(input_ids)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
                return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=None)


        class TinySparkModel(TinySparkPreTrainedModel):
            def forward(self, input_ids, attention_mask=None):
                self.check(input_ids, attention_mask)
                return BaseModelOutput(last_hidden_state=self.model(input_ids, return_hidden=True))
    """,
}
SPARK_AUTO_MAP = {
    "AutoConfig": "configuration_tinyspark.TinySparkConfig",
    "AutoModelForCausalLM": "modeling_tinyspark.TinySparkForCausalLM",
}
# What the package's own limit becomes: above the length of the prompts of `make_row` (about 280 tokens)
RAISED_CONTEXT = {"max_seq_len": 512, "max_position_embeddings": 512}


def spark_package(directory, tokenizer, with_base_model=False) -> str:
    """
    The code, `config.json`, random weights and tokenizer of the package in a new folder of `directory`, written
    without importing its code. Its name is new, and so is the name of its dynamic module.
    """
    import textwrap

    from safetensors.torch import save_file

    folder = tempfile.mkdtemp(prefix="tinyspark_", dir=directory)
    for name, code in SPARK_PACKAGE.items():
        Path(folder, name).write_text(textwrap.dedent(code).lstrip(), encoding="utf-8")
    auto_map = {**SPARK_AUTO_MAP, **({"AutoModel": "modeling_tinyspark.TinySparkModel"} if with_base_model else {})}
    vocab_size, dim = len(tokenizer), 24
    config = {"model_type": "tinyspark", "auto_map": auto_map, "vocab_size": vocab_size, "dim": dim, "n_layers": 2}
    config["architectures"] = ["TinySparkForCausalLM"]
    Path(folder, "config.json").write_text(json.dumps(config), encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors = {"model.embed.weight": torch.randn(vocab_size, dim, generator=generator) * 0.5}
    for index in range(2):
        tensors[f"model.layers.{index}.weight"] = torch.randn(dim, dim, generator=generator) * 0.5
        tensors[f"model.layers.{index}.bias"] = torch.randn(dim, generator=generator) * 0.5
    tensors["model.final_norm.weight"] = torch.ones(dim)
    tensors["model.final_norm.bias"] = torch.zeros(dim)
    save_file(tensors, f"{folder}/model.safetensors", metadata={"format": "pt"})
    tokenizer.save_pretrained(folder)
    return folder


def package_modules(folder) -> list[str]:
    """The imported modules of the package in `folder`: its dynamic module, under the name of the folder."""
    import sys

    return [module for module in sys.modules if Path(folder).name in module]
