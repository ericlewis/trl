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
Loading the backbone of a new decision head: strictly (no weight newly initialised), from the one key prefix of a
checkpoint that holds all of it, or with the checkpoint's own code when it has no model of Transformers and the caller
allows it. [`load_decision_backbone`] does all of it in one call; the other functions are its parts.
"""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES
from transformers.utils import cached_file


def _listed(keys: list[str]) -> str:
    return ", ".join(keys[:10]) + (", ..." if len(keys) > 10 else "")


def is_causal(config) -> bool:
    """
    Whether a backbone attends causally: a model type with a causal language modeling class that is not a masked
    language model (BERT and RoBERTa have both and are encoders), or a config marked as a decoder. A model of its
    checkpoint's own code is causal when its `auto_map` has a causal language modeling class and no masked one.
    """
    from transformers import ClefConfig

    model_type = config.model_type
    is_decoder_type = (
        model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and model_type not in MODEL_FOR_MASKED_LM_MAPPING_NAMES
    )
    code = ClefConfig.backbone_code(config)
    if code is not None:
        is_decoder_type = "AutoModelForCausalLM" in code and "AutoModelForMaskedLM" not in code
    return is_decoder_type or bool(getattr(config, "is_decoder", False))


@dataclass
class BackboneWeights:
    """Where the weights of a backbone came from: `direct` or `prefix "<p>"`, and the checkpoint tensors left out."""

    source: str
    dropped: list[str]

    def __str__(self):
        names = _listed(self.dropped)
        return f"{self.source} (dropped {len(self.dropped)} non-backbone tensors{': ' + names if names else ''})"


def remote_code_setting(model_args, model_kwargs) -> tuple[bool, dict]:
    """
    The `trust_remote_code` setting of `model_args` and `model_kwargs` without it: an entry of `model_kwargs` (the
    `model_init_kwargs`) is taken out, and must agree with the setting.
    """
    trust_remote_code = bool(getattr(model_args, "trust_remote_code", False))
    model_kwargs = dict(model_kwargs)
    if "trust_remote_code" in model_kwargs:
        given = model_kwargs.pop("trust_remote_code")
        if not isinstance(given, bool) and given not in (0, 1):  # bool("false") would be True
            raise ValueError(f"model_init_kwargs sets trust_remote_code to {given!r}; it must be true or false")
        if bool(given) != trust_remote_code:
            raise ValueError(
                f"model_init_kwargs sets trust_remote_code to {given!r}, but trust_remote_code is {trust_remote_code}; "
                "set it once, as trust_remote_code"
            )
    return trust_remote_code, model_kwargs


def raw_config(name, revision=None) -> dict:
    """The `config.json` of `name` as it is written (empty when there is none): reading it imports nothing."""
    path = cached_file(name, "config.json", revision=revision, _raise_exceptions_for_missing_entries=False)
    return {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_code(name, revision=None) -> dict | None:
    """
    The `auto_map` of a model that only loads with its checkpoint's own code (a model type that Transformers does not
    have), or of the backbone of such a Clef model; `None` otherwise. Reads `config.json` only: imports nothing.
    """
    from transformers import ClefConfig

    raw = raw_config(name, revision=revision)
    if raw.get("model_type") == ClefConfig.model_type:
        raw = raw.get("text_config") or {}
    return ClefConfig.backbone_code(raw)


def needs_remote_code(name, trust_remote_code, revision=None) -> bool:
    """
    Whether `name` is loaded with its checkpoint's own code: only when it cannot be loaded without it, and then only
    when `trust_remote_code` is set. Otherwise it raises, before any of that code is imported.
    """
    code = checkpoint_code(name, revision=revision)
    if code is None:
        return False
    if not trust_remote_code:
        classes = ", ".join(f"{key}: {value}" for key, value in sorted(code.items()))
        raise ValueError(
            f"{name} can only be loaded with its own code ({classes}), which is run only when asked to. Read that "
            "code; if you trust it, set trust_remote_code: true."
        )
    return True


def loading_problems(info) -> list[str]:
    """The keys that `from_pretrained` initialised anew: missing from the checkpoint, or of another shape."""
    mismatched = [key[0] if isinstance(key, tuple) else key for key in info.get("mismatched_keys", ())]
    return sorted({*info.get("missing_keys", ()), *mismatched})


def checkpoint_shapes(name, revision=None) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The file and shape of every tensor of the safetensors weights of `name`, read from their headers only."""
    files = []
    single = cached_file(name, "model.safetensors", revision=revision, _raise_exceptions_for_missing_entries=False)
    if single is not None:
        files = [single]
    else:
        index = cached_file(
            name, "model.safetensors.index.json", revision=revision, _raise_exceptions_for_missing_entries=False
        )
        if index is not None:
            shards = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
            files = [cached_file(name, shard, revision=revision) for shard in shards]
    shapes = {}
    for file in files:
        with safe_open(file, framework="pt") as weights:
            for key in weights.keys():
                shapes[key] = (file, tuple(weights.get_slice(key).get_shape()))
    return shapes


def repair_backbone(name, backbone, missing, revision=None) -> BackboneWeights:
    """
    Load the weights of `backbone` from the safetensors of `name` under the single key prefix that holds every one of
    them with its shape (`backbone.`, `model.`, ...), strictly. Raises a `ValueError` naming the missing keys when no
    prefix holds them all, or when several do.
    """
    expected = {key: tuple(value.shape) for key, value in backbone.state_dict().items()}
    shapes = checkpoint_shapes(name, revision)
    reference = next(iter(expected))
    # Any prefix that holds every key holds the first: the candidates are the prefixes of that one
    candidates = sorted(
        key[: -len(reference)]
        for key in shapes
        if key.endswith("." + reference) and key[: -len(reference)] not in ("", ".")
    )
    # The weights each candidate lacks, or holds with another shape
    absent = {
        prefix: [key for key, shape in expected.items() if shapes.get(prefix + key, (None, None))[1] != shape]
        for prefix in candidates
    }
    resolved = [prefix for prefix in candidates if not absent[prefix]]
    if len(resolved) > 1:
        prefixes = ", ".join(f'"{prefix}"' for prefix in resolved)
        raise ValueError(
            f"The weights of {name} are ambiguous: the prefixes {prefixes} each hold a whole "
            f"{type(backbone).__name__}. Save the backbone on its own and train from that."
        )
    if not resolved:
        where = "under the names of its class"
        if candidates:  # the keys that the closest prefix lacks
            closest = min(candidates, key=lambda prefix: len(absent[prefix]))
            missing, where = absent[closest], f'under the closest prefix "{closest}"'
        raise ValueError(
            f"{len(missing)} of the {len(expected)} weights of the {type(backbone).__name__} backbone are missing "
            f"from {name} {where} (or have another shape), and no single key prefix of its safetensors holds them "
            f"all: {_listed(missing)}. Refusing to train from newly initialised weights."
        )
    (prefix,) = resolved
    state = {}
    for key in expected:
        file, _ = shapes[prefix + key]
        with safe_open(file, framework="pt") as weights:
            state[key] = weights.get_tensor(prefix + key)
    backbone.load_state_dict(state, strict=True)
    used = {prefix + key for key in expected}
    return BackboneWeights(f'prefix "{prefix}"', sorted(key for key in shapes if key not in used))


def check_per_layer_settings(name, config, revision=None):
    """
    Refuse a checkpoint whose `config.json` lists `attention_windows` or `rope_theta` per layer (as remote-code
    UltraBERT packages do) with values that differ within a kind of layer, or with a length other than the number of
    layers. Transformers' native config reads such lists as one window and one RoPE base per kind of layer (`null`
    windows are global layers), so it would build another architecture without a warning.
    """
    raw = raw_config(name, revision=revision)
    lists = {key: raw[key] for key in ("attention_windows", "rope_theta") if isinstance(raw.get(key), list)}
    if not lists:
        return
    kinds = getattr(config, "layer_types", None)
    if kinds is None:  # the kinds the windows give, or else one kind
        windows = lists.get("attention_windows")
        kinds = ["full_attention" if window is None else "sliding_attention" for window in windows or []]
        kinds = kinds or ["all"] * len(lists["rope_theta"])
    for key, values in lists.items():
        if len(values) != len(kinds):
            raise ValueError(
                f"{name} sets {key} per layer for {len(values)} layers, but has {len(kinds)}; refusing to guess "
                "which value each layer takes"
            )
        for kind in dict.fromkeys(kinds):
            found = list(dict.fromkeys(value for value, other in zip(values, kinds, strict=False) if other == kind))
            if len(found) > 1:
                raise ValueError(
                    f"{name} sets {key} per layer with different values within its {kind} layers ({found}): "
                    "Transformers reads one value per kind of layer, so it would build another architecture"
                )
            if key == "attention_windows" and (found[0] is None) != (kind == "full_attention"):
                raise ValueError(
                    f"{name} sets {key} per layer with {found[0]} for its {kind} layers, which does not match "
                    "the kind of these layers"
                )


def load_backbone(name, config, model_kwargs, remote_code=False):
    """
    The backbone of a new head, the table its `lm_head` starts from, and where its weights came from: the base model of
    a causal LM with its output embeddings, or a bidirectional encoder with its input embeddings. A checkpoint that
    only registers a sequence classification class (a reranker) is loaded as that and unwrapped to its `base_model`.

    No weight of the backbone is newly initialised: when the checkpoint does not provide one under the names of the
    loaded class, the backbone is read from its safetensors under the one key prefix that holds all of its weights (see
    [`repair_backbone`]), or a `ValueError` is raised. The checkpoint's own code runs only with `remote_code`, for a
    model that cannot be loaded without it (see [`load_remote_backbone`]).
    """
    if remote_code:
        return load_remote_backbone(name, config, model_kwargs)
    model_kwargs = {**model_kwargs, "trust_remote_code": False}
    check_per_layer_settings(name, config, revision=model_kwargs.get("revision"))
    architectures = config.architectures or []
    if is_causal(config):
        loaded, info = AutoModelForCausalLM.from_pretrained(name, output_loading_info=True, **model_kwargs)
    elif architectures and all(architecture.endswith("ForSequenceClassification") for architecture in architectures):
        loaded, info = AutoModelForSequenceClassification.from_pretrained(
            name, output_loading_info=True, **model_kwargs
        )
        if loaded.base_model is loaded or not hasattr(loaded.base_model, "get_input_embeddings"):
            raise ValueError(
                f"{name} registers only {architectures}, whose base model cannot be unwrapped into a backbone"
            )
    else:
        loaded, info = AutoModel.from_pretrained(name, output_loading_info=True, **model_kwargs)
    backbone = loaded.base_model
    # The keys of the loaded class that belong to its backbone, and the others (a task head, an `lm_head`)
    wrapper = "" if backbone is loaded else f"{loaded.base_model_prefix}."
    problems = loading_problems(info)
    missing = [key[len(wrapper) :] for key in problems if key.startswith(wrapper)]
    head = [key for key in problems if not key.startswith(wrapper)]
    if is_causal(config) and head:  # the output embeddings the lexical prior starts from
        raise ValueError(f"{name} is missing weights of its causal LM outside its backbone: {', '.join(head[:10])}")
    if missing:
        weights = repair_backbone(name, backbone, missing, revision=model_kwargs.get("revision"))
    else:
        dropped = sorted(info.get("unexpected_keys", ()))
        if wrapper and not is_causal(config):  # the task head of the loaded class is not kept
            dropped += sorted(key for key in loaded.state_dict() if not key.startswith(wrapper) and key not in head)
        weights = BackboneWeights("direct", dropped)
    embeddings = loaded.get_output_embeddings().weight if is_causal(config) else backbone.get_input_embeddings().weight
    return backbone, embeddings, weights


def load_remote_backbone(name, config, model_kwargs):
    """
    The backbone of a new head on a model of its checkpoint's own code, loaded with that code: the class of its
    `auto_map` that a saved Clef model rebuilds, `AutoModel` when there is one, else `AutoModelForCausalLM`, used
    itself as the backbone (not unwrapped). No weight is newly initialised: missing or mismatched weights raise, and no
    other key prefix is looked for. There is no table for an `lm_head` (`None`): the lexical prior of the joint head
    reads the backbone's own embeddings.
    """
    from transformers import ClefConfig

    code = ClefConfig.backbone_code(config)
    classes = {"AutoModel": AutoModel, "AutoModelForCausalLM": AutoModelForCausalLM}
    key = next((key for key in classes if isinstance(code.get(key), str)), None)
    if key is None:
        raise ValueError(
            f"{name} is defined by its own code, whose auto_map names neither AutoModel nor AutoModelForCausalLM "
            f"({sorted(code)}): a saved Clef model could not rebuild its backbone"
        )
    auto_class = classes[key]
    backbone, info = auto_class.from_pretrained(name, output_loading_info=True, **model_kwargs, trust_remote_code=True)
    problems = loading_problems(info)
    if problems:
        raise ValueError(
            f"{len(problems)} weights of the {type(backbone).__name__} backbone are missing from {name} or have "
            f"another shape: {_listed(problems)}. Its class comes from the checkpoint's own code, so its weights are "
            "not looked for under another key prefix. Refusing to train from newly initialised weights."
        )
    weights = BackboneWeights(f"direct, {key} of its own code", sorted(info.get("unexpected_keys", ())))
    return backbone, None, weights


def hidden_states_of(backbone) -> tuple[int, str | None]:
    """
    The width of the final hidden states of a backbone of its checkpoint's own code, read from the model in one forward
    pass of a few tokens, and the submodule they are the output of (`backbone_hidden_module`) when the backbone returns
    neither `last_hidden_state` nor `hidden_states`, found with `ClefForDecision.find_backbone_hidden_module`.
    """
    from transformers import ClefForDecision

    weight = backbone.get_input_embeddings().weight
    input_ids = (torch.arange(1, 9, device=weight.device) % weight.shape[0])[None]
    parameters = inspect.signature(backbone.forward).parameters
    kwargs = {}
    if "output_hidden_states" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        kwargs["output_hidden_states"] = True
    training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            output = backbone(input_ids=input_ids, **kwargs)
    finally:
        backbone.train(training)
    hidden_states = getattr(output, "last_hidden_state", None)
    if hidden_states is None and getattr(output, "hidden_states", None) is not None:
        hidden_states = output.hidden_states[-1]
    if hidden_states is not None:
        return hidden_states.shape[-1], None
    module = ClefForDecision.find_backbone_hidden_module(backbone)
    # The module's output gives the logits through the output embeddings: it has their width
    return backbone.get_output_embeddings().weight.shape[-1], module


def position_limit(backbone) -> int | None:
    """
    The longest sequence a backbone with a learned table of absolute positions (BERT, RoBERTa) can read, or `None` when
    it has none (rotary or relative positions).
    """
    for name, module in backbone.named_modules():
        if name.split(".")[-1] == "position_embeddings" and isinstance(module, torch.nn.Embedding):
            # RoBERTa-style tables count positions from after the padding index
            offset = 0 if module.padding_idx is None else module.padding_idx + 1
            return module.num_embeddings - offset
    return None


def check_max_length(backbone, text_config, max_length) -> str | None:
    """
    Refuse a `max_length` beyond the positions a backbone with a learned table of absolute positions can read (see
    [`position_limit`]). Without such a table, return a warning when `max_length` is above the
    `max_position_embeddings` of its config `text_config` (rotary models run past it, but a model may enforce it), else
    `None`.
    """
    model_type = text_config.model_type
    limit = position_limit(backbone)
    if limit is not None and max_length > limit:
        raise ValueError(
            f"max_length is {max_length} tokens, but the {model_type} backbone "
            f"reads at most {limit} positions; set max_length to {limit} or less"
        )
    configured = getattr(text_config, "max_position_embeddings", None)
    if limit is None and isinstance(configured, int) and max_length > configured:
        return (
            f"Warning: max_length is {max_length} tokens, above the max_position_embeddings of the "
            f"{model_type} backbone ({configured}); the model may refuse or degrade on longer prompts. Where its "
            "architecture allows it, model_init_kwargs can raise its limit."
        )
    return None


@dataclass
class DecisionBackbone:
    """
    What [`load_decision_backbone`] loads.

    Args:
        model (`PreTrainedModel`):
            The backbone, to pass as `model=` to the decision model.
        embeddings (`torch.Tensor` or `None`):
            The table a joint head's `lm_head` starts from (`None` for a backbone of its checkpoint's own code).
        weights ([`BackboneWeights`]):
            Where the weights came from.
        causal (`bool`):
            Whether the backbone attends causally.
        hidden_size (`int`):
            The width of its final hidden states.
        hidden_module (`str` or `None`):
            The submodule whose output they are, when the backbone does not return them (else `None`).
    """

    model: Any
    embeddings: torch.Tensor | None
    weights: BackboneWeights
    causal: bool
    hidden_size: int
    hidden_module: str | None


def load_decision_backbone(name, trust_remote_code=False, model_kwargs=None, config=None) -> DecisionBackbone:
    """
    Load `name` as the backbone of a new decision head, with no weight newly initialised (see [`load_backbone`]).

    Its checkpoint's own code runs only when `trust_remote_code` is set and the model cannot be loaded without it (see
    [`needs_remote_code`]); otherwise such a model is refused before any of its code is imported. `model_kwargs` go to
    `from_pretrained` (`trust_remote_code` is the argument of its own); `config` is the model's config when it is
    already loaded.
    """
    model_kwargs = dict(model_kwargs or {})
    if "trust_remote_code" in model_kwargs:
        raise ValueError("Pass trust_remote_code as the argument of its own, not in model_kwargs")
    revision = model_kwargs.get("revision")
    remote_code = needs_remote_code(name, trust_remote_code, revision=revision)
    if config is None:
        config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=remote_code)
    backbone, embeddings, weights = load_backbone(name, config, model_kwargs, remote_code=remote_code)
    if remote_code:  # such a config may have no `hidden_size`, and its model no hidden states in its output
        hidden_size, hidden_module = hidden_states_of(backbone)
    else:
        hidden_size, hidden_module = backbone.config.hidden_size, None
    return DecisionBackbone(backbone, embeddings, weights, is_causal(backbone.config), hidden_size, hidden_module)
