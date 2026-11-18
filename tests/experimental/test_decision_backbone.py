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

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("transformers.decision")
from transformers.loss.loss_utils import LOSS_MAPPING  # noqa: E402


if "ClefForDecision" not in LOSS_MAPPING:
    pytest.skip(
        "Requires a Transformers build with Clef training support (native loss and batched forward) on top of #50058",
        allow_module_level=True,
    )

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForDecision,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
    ClefConfig,
    ClefForDecision,
    set_seed,
)

from trl.experimental.decision import (  # noqa: E402
    check_max_length,
    load_backbone,
    load_decision_backbone,
    position_limit,
    remote_code_setting,
)

from ..testing_utils import TrlTestCase  # noqa: E402
from .decision_clef_helpers import (  # noqa: E402
    MODEL_ID,
    PAIR_HEAD,
    RAISED_CONTEXT,
    encoder_tokenizer,
    package_modules,
    prefixed,
    reranker_package,
    spark_package,
    tiny_encoder,
    tiny_ultrabert,
    ultrabert_tokenizer,
)


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def assert_weights(backbone, expected):
    """`backbone` holds exactly the tensors of the state dict `expected`."""
    built = backbone.state_dict()
    assert set(built) == set(expected)
    for name, value in built.items():
        assert torch.equal(value, expected[name]), name


class TestStrictLoading(TrlTestCase):
    """Backbones of Transformers: no weight newly initialised, or from the one key prefix that holds all of them."""

    def test_unwrappable_reranker_raises(self, tokenizer, monkeypatch):
        tokenizer = encoder_tokenizer(tokenizer)
        tiny_encoder(tokenizer, "bert-reranker").save_pretrained(f"{self.tmp_dir}/reranker")
        config = AutoConfig.from_pretrained(f"{self.tmp_dir}/reranker")

        class Opaque(torch.nn.Module):
            base_model = property(lambda self: self)

        info = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set()}
        monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: (Opaque(), info))
        with pytest.raises(ValueError, match="cannot be unwrapped"):
            load_backbone(f"{self.tmp_dir}/reranker", config, {})

    def test_reranker_package_backbone_is_loaded_not_reinitialised(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        folder = reranker_package(
            f"{self.tmp_dir}/reranker", backbone, tokenizer, {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        )
        # As measured on the real package: `AutoModel` alone loads none of its backbone, with only a warning
        _, info = AutoModel.from_pretrained(folder, output_loading_info=True)
        assert set(info["missing_keys"]) == set(backbone.state_dict())
        loaded = load_decision_backbone(folder)
        assert type(loaded.model).__name__ == "UltrabertModel"
        assert_weights(loaded.model, backbone.state_dict())
        # The settings of the remote config are read: the windows and RoPE bases of the layers
        config = loaded.model.config
        assert config.layer_types == ["sliding_attention", "sliding_attention", "full_attention"]
        assert config.sliding_window == 8
        assert config.rope_parameters == backbone.config.rope_parameters
        bases = {kind: parameters["rope_theta"] for kind, parameters in config.rope_parameters.items()}
        assert bases == {"sliding_attention": 1234.0, "full_attention": 98765.0}
        assert (loaded.causal, loaded.hidden_size, loaded.hidden_module) == (False, 32, None)
        assert (
            str(loaded.weights)
            == f'prefix "backbone." (dropped 3 non-backbone tensors: {", ".join(sorted(PAIR_HEAD))})'
        )

    def test_sharded_reranker_package_backbone_is_loaded(self, tokenizer):
        from safetensors.torch import save_file

        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        folder = f"{self.tmp_dir}/reranker"
        reranker_package(folder, backbone, tokenizer, {**prefixed(backbone, "backbone."), **PAIR_HEAD})
        # The same tensors in two shards with an index, as large packages store them
        tensors = {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        names = sorted(tensors)
        shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
        Path(folder, "model.safetensors").unlink()
        for shard, keys in shards.items():
            save_file({key: tensors[key] for key in keys}, f"{folder}/{shard}", metadata={"format": "pt"})
        weight_map = {key: shard for shard, keys in shards.items() for key in keys}
        Path(folder, "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        loaded = load_decision_backbone(folder)
        assert_weights(loaded.model, backbone.state_dict())
        assert str(loaded.weights).startswith('prefix "backbone." (dropped 3 non-backbone tensors')

    @pytest.mark.parametrize(
        "changes, match",
        [
            ({}, None),
            ({"attention_windows": [8, 8, None], "rope_theta": [1234.0, 1234.0, 98765.0]}, None),
            ({"attention_windows": [8, 4, None]}, "attention_windows"),
            ({"rope_theta": [1234.0, 4321.0, 98765.0]}, "rope_theta"),
            ({"attention_windows": [8, None, None], "rope_theta": [1234.0, 98765.0, 5.0]}, "rope_theta"),
            # One base too many, which the native config drops
            ({"rope_theta": [1234.0, 1234.0, 98765.0, 98765.0]}, "rope_theta"),
        ],
    )
    def test_collapsed_per_layer_settings_raise(self, tokenizer, changes, match):
        # The native config keeps one window and one RoPE base per kind of layer: lists that vary within a kind would
        # load another architecture without a word
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        tensors = {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        folder = reranker_package(f"{self.tmp_dir}/reranker", backbone, tokenizer, tensors, **changes)
        if match is None:
            assert type(load_decision_backbone(folder).model).__name__ == "UltrabertModel"
            return
        with pytest.raises(ValueError, match=match) as error:
            load_decision_backbone(folder)
        assert "per layer" in str(error.value)

    def test_missing_backbone_weights_raise(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        keys = list(backbone.state_dict())
        kept, removed = keys[::2], keys[1::2]
        folder = reranker_package(
            f"{self.tmp_dir}/half", backbone, tokenizer, {**prefixed(backbone, "backbone.", kept), **PAIR_HEAD}
        )
        with pytest.raises(ValueError, match="missing") as error:
            load_decision_backbone(folder)
        message = str(error.value)
        assert removed[0] in message
        assert sum(key in message for key in removed) == min(10, len(removed))
        # Every key under the prefix but one of the wrong shape: refused too
        tensors = {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        tensors["backbone.norm.weight"] = torch.ones(16)
        folder = reranker_package(f"{self.tmp_dir}/shape", backbone, tokenizer, tensors)
        with pytest.raises(ValueError, match="norm.weight"):
            load_decision_backbone(folder)

    def test_ambiguous_prefix_raises(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        keys = list(backbone.state_dict())
        # Split across two prefixes: no single one holds every key
        split = {**prefixed(backbone, "backbone.", keys[::2]), **prefixed(backbone, "encoder.", keys[1::2])}
        folder = reranker_package(f"{self.tmp_dir}/split", backbone, tokenizer, split)
        with pytest.raises(ValueError, match="missing"):
            load_decision_backbone(folder)
        # Two whole copies under two prefixes: which one is the backbone is not guessed
        teacher = {f"teacher.{key}": torch.zeros_like(value) for key, value in backbone.state_dict().items()}
        copies = {**prefixed(backbone, "backbone."), **teacher}
        folder = reranker_package(f"{self.tmp_dir}/copies", backbone, tokenizer, copies)
        with pytest.raises(ValueError, match="ambiguous") as error:
            load_decision_backbone(folder)
        assert '"backbone."' in str(error.value) and '"teacher."' in str(error.value)

    def test_native_ultrabert_checkpoint_loads_directly(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        backbone.save_pretrained(f"{self.tmp_dir}/native")
        tokenizer.save_pretrained(f"{self.tmp_dir}/native")
        loaded = load_decision_backbone(f"{self.tmp_dir}/native")
        assert type(loaded.model).__name__ == "UltrabertModel"
        assert_weights(loaded.model, backbone.state_dict())
        assert str(loaded.weights) == "direct (dropped 0 non-backbone tensors)"

    def test_code_of_a_native_model_type_is_never_executed(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        folder = reranker_package(
            f"{self.tmp_dir}/reranker", backbone, tokenizer, {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        )
        sentinel = Path(self.tmp_dir, "imported")
        for module in ("configuration_ultrabert", "modeling_reranker", "modeling_ultrabert"):
            Path(folder, f"{module}.py").write_text(
                f"from pathlib import Path\nPath({str(sentinel)!r}).write_text({module!r})\n"
                f"raise RuntimeError('remote code {module} was imported')\n",
                encoding="utf-8",
            )
        # Its model type is one of Transformers': trust_remote_code still loads it with Transformers' own classes
        for trust_remote_code in (False, True):
            loaded = load_decision_backbone(folder, trust_remote_code)
            assert type(loaded.model).__name__ == "UltrabertModel"
            assert str(loaded.weights).startswith('prefix "backbone."')
        assert not sentinel.exists()
        assert not any("modeling_reranker" in name or "configuration_ultrabert." in name for name in sys.modules)

    @pytest.mark.parametrize("kind", ["qwen2", "bert", "bert-reranker", "modernbert"])
    def test_embeddings_the_lexical_prior_starts_from(self, tokenizer, kind):
        # A causal LM: its base model, with its output embeddings; an encoder (a reranker unwrapped to its base
        # model): with its input embeddings
        folder = f"{self.tmp_dir}/{kind}"
        if kind == "qwen2":
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
            expected = model.get_output_embeddings().weight
        else:
            model = tiny_encoder(encoder_tokenizer(tokenizer), kind)
            expected = model.base_model.get_input_embeddings().weight
        model.save_pretrained(folder)
        loaded = load_decision_backbone(folder, model_kwargs={"dtype": "float32"})
        assert type(loaded.model) is type(model.base_model)
        assert_weights(loaded.model, model.base_model.state_dict())
        assert torch.equal(loaded.embeddings, expected)
        assert loaded.causal == (kind == "qwen2")
        assert (loaded.hidden_size, loaded.hidden_module) == (model.config.hidden_size, None)
        # A reranker's task head is not kept
        dropped = ["classifier.bias", "classifier.weight"] if kind == "bert-reranker" else []
        assert (loaded.weights.source, loaded.weights.dropped) == ("direct", dropped)

    def test_position_limit(self, tokenizer):
        tokenizer = encoder_tokenizer(tokenizer)
        sizes = {"vocab_size": len(tokenizer) + 2, "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2}
        set_seed(0)
        backbone = BertModel(BertConfig(**sizes, num_attention_heads=2, max_position_embeddings=128))
        assert position_limit(backbone) == 128
        rotary = AutoModel.from_pretrained(MODEL_ID)
        assert position_limit(rotary) is None  # rotary positions
        # Beyond the table: refused; beyond a rotary model's max_position_embeddings: a warning
        assert check_max_length(backbone, backbone.config, 128) is None
        with pytest.raises(ValueError, match="at most 128 positions; set max_length to 128 or less"):
            check_max_length(backbone, backbone.config, 129)
        limit = rotary.config.max_position_embeddings
        assert check_max_length(rotary, rotary.config, limit) is None
        warning = check_max_length(rotary, rotary.config, limit + 1)
        assert warning.startswith(f"Warning: max_length is {limit + 1} tokens, above the max_position_embeddings")


class TestRemoteCode(TrlTestCase):
    """A backbone that exists only as code in its repository, which runs only with `trust_remote_code`."""

    def test_without_the_flag_nothing_of_the_package_is_imported(self, tokenizer, monkeypatch):
        folder = spark_package(self.tmp_dir, tokenizer)

        def from_pretrained(*args, **kwargs):
            raise AssertionError("the config was loaded")

        # Refused from its config.json alone: once a package of the same model type is registered in this process, its
        # dynamic module is no longer imported, so the modules alone cannot tell
        monkeypatch.setattr(AutoConfig, "from_pretrained", from_pretrained)
        with pytest.raises(ValueError, match="trust_remote_code: true") as error:
            load_decision_backbone(folder, model_kwargs=RAISED_CONTEXT)
        assert folder in str(error.value) and "modeling_tinyspark.TinySparkForCausalLM" in str(error.value)
        assert not package_modules(folder)
        with pytest.raises(ValueError, match="argument of its own"):
            load_decision_backbone(folder, model_kwargs={"trust_remote_code": True})
        assert not package_modules(folder)

    def test_flag_in_model_init_kwargs_must_agree(self, tokenizer):
        flag = SimpleNamespace(trust_remote_code=True)
        with pytest.raises(ValueError, match="model_init_kwargs sets trust_remote_code"):
            remote_code_setting(flag, {**RAISED_CONTEXT, "trust_remote_code": False})
        # Only a boolean (or 0 and 1): the string "false" is not read as true
        for value in ("false", "true", 2):
            with pytest.raises(ValueError, match=f"trust_remote_code to {value!r}; it must be true or false"):
                remote_code_setting(flag, {**RAISED_CONTEXT, "trust_remote_code": value})
        # The same value, in both places, is accepted: it is taken out of the kwargs
        assert remote_code_setting(flag, {**RAISED_CONTEXT, "trust_remote_code": 1}) == (True, RAISED_CONTEXT)
        assert remote_code_setting(SimpleNamespace(), {}) == (False, {})
        folder = spark_package(self.tmp_dir, tokenizer)
        trust_remote_code, kwargs = remote_code_setting(flag, {"dtype": "float32", "trust_remote_code": True})
        model = load_decision_backbone(folder, trust_remote_code, kwargs).model
        # A class of the package's code (of this folder, or of the same model type registered by an earlier load)
        assert type(model).__name__ == "TinySparkForCausalLM"
        assert type(model).__module__.startswith("transformers_modules.")

    def test_causal_lm_of_its_own_code_is_the_backbone(self, tokenizer):
        folder = spark_package(self.tmp_dir, tokenizer)
        loaded = load_decision_backbone(folder, True, {"dtype": "float32", **RAISED_CONTEXT})
        # The causal language model itself, its width read from the model and its hidden states from an inner layer
        assert type(loaded.model).__name__ == "TinySparkForCausalLM"
        assert loaded.causal and loaded.embeddings is None
        assert (loaded.hidden_size, loaded.hidden_module) == (24, "model.final_norm")
        assert str(loaded.weights) == "direct, AutoModelForCausalLM of its own code (dropped 0 non-backbone tensors)"
        # The settings of model_kwargs reach its config
        assert (loaded.model.config.max_seq_len, loaded.model.config.max_position_embeddings) == (512, 512)

    def test_package_with_a_base_model_is_loaded_as_that(self, tokenizer):
        folder = spark_package(self.tmp_dir, tokenizer, with_base_model=True)
        loaded = load_decision_backbone(folder, True, {"dtype": "float32", **RAISED_CONTEXT})
        assert type(loaded.model).__name__ == "TinySparkModel"
        # Its output has `last_hidden_state`: no submodule is needed
        assert (loaded.hidden_size, loaded.hidden_module) == (24, None)
        assert str(loaded.weights).startswith("direct, AutoModel of its own code")
        # A model built around it is rebuilt with the same class
        config = ClefConfig(
            text_config=loaded.model.config, backbone_hidden_size=loaded.hidden_size, head_type="cosine"
        )
        ClefForDecision(config, model=loaded.model).save_pretrained(f"{self.tmp_dir}/clef")
        reloaded = AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/clef", trust_remote_code=True)
        assert type(reloaded.model).__name__ == "TinySparkModel"

    def test_missing_weights_raise_without_a_repair(self, tokenizer):
        from safetensors.torch import load_file, save_file

        folder = spark_package(self.tmp_dir, tokenizer)
        tensors = load_file(f"{folder}/model.safetensors")
        save_file({f"backbone.{key}": value for key, value in tensors.items()}, f"{folder}/model.safetensors")
        with pytest.raises(ValueError, match="not looked for under another key prefix"):
            load_decision_backbone(folder, True, {"dtype": "float32"})
