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

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset


pytest.importorskip("transformers.decision")

from examples.decisions.clef_typed_decisions import (
    ClefModelArguments,
    ClefScriptArguments,
    DataCollatorForClef,
    build_model,
    clef_decision_config,
    describe,
    load_backbone,
    prepare_split,
)
from examples.decisions.clef_typed_decisions import main as clef_main
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForDecision,
    AutoTokenizer,
    BertConfig,
    ClefConfig,
    ClefForDecision,
    ModernBertConfig,
    Qwen2Config,
    pipeline,
    set_seed,
)

from trl.experimental.decision import DecisionConfig, DecisionTrainer, compute_calibration_metrics
from trl.experimental.decision.decision_trainer import pack_rows, unpack_rows

from ..testing_utils import TrlTestCase, require_peft


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None, "alpha": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}
INSTRUCTIONS = {question["instructions"]: name for name, question in QUESTIONS.items()}
LONG_ROW = {"state": "word " * 200, "questions": QUESTIONS, "gold": {"refund": True, "team": "zeta", "urgency": 0}}
# A long state of distinct words, so that a cut shows which end of the state was kept
VARIED_ROW = {**LONG_ROW, "state": " ".join(f"item{index} costs {index * 7}" for index in range(80))}


def make_row(index, names=("refund", "team", "urgency")):
    gold = {"refund": index % 2 == 0, "team": "alpha" if index % 3 == 0 else "zeta", "urgency": index % 3}
    return {
        "state": f"ticket {index}: {'refund' if index % 2 == 0 else 'thanks'} {'alpha' if index % 3 == 0 else 'zeta'}",
        "questions": {name: QUESTIONS[name] for name in names},
        "gold": {name: gold[name] for name in names},
    }


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def decision_config():
    return clef_decision_config()


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
    """A tiny UltraBERT: two local layers of radius 8 and a global one, each kind with its own RoPE base."""
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


def reranker_package(folder, backbone, tokenizer, tensors):
    """
    A folder laid out like the UltraBERT reranker package: its config as its remote code writes it (per-layer
    `attention_windows` and `rope_theta`, the `UltrabertForReranking` architecture, an `auto_map` to code files that
    are not there), `tensors` as its `model.safetensors`, and the tokenizer.
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
    Path(folder, "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, f"{folder}/model.safetensors", metadata={"format": "pt"})
    tokenizer.save_pretrained(folder)
    return folder


def encoder_tokenizer(tokenizer):
    """The tokenizer with a `[CLS]` and a `[SEP]`, which the plain prompt of encoders starts and ends with."""
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})
    return tokenizer


def tokenized(collator, rows):
    return [collator.tokenize(row) for row in rows]


def question_names(tokenizer, batch, row):
    """The names of the questions of a row of a batch, read from the instruction span of each question."""
    marks = batch["marks"][row]
    names, position = [], 0
    for count in batch["num_options"][row].tolist():
        if count == 0:
            break
        start, end = marks[position : position + 2].tolist()
        text = tokenizer.decode(batch["input_ids"][row, start:end])
        names.append(next(name for instruction, name in INSTRUCTIONS.items() if instruction in text))
        position += 2 * (1 + count)
    return names


def marked_spans(tokenizer, features):
    """The text of every instruction and option span of a tokenized record, read through its marks."""
    marks, input_ids = features["marks"], features["input_ids"]
    return [tokenizer.decode(input_ids[start:end]) for start, end in zip(marks[::2], marks[1::2], strict=False)]


def state_layout(tokenizer, collator, row):
    """
    The untruncated features of `row`, the tokens of its state (a string between sentinels, so tokenized on its own)
    and where they start, found independently of the collator's own search.
    """
    full = collator.tokenize(row)
    state_ids = tokenizer(row["state"], add_special_tokens=False)["input_ids"]
    input_ids = full["input_ids"]
    starts = [
        start
        for start in range(len(input_ids) - len(state_ids) + 1)
        if input_ids[start : start + len(state_ids)] == state_ids
    ]
    assert len(starts) == 1
    return full, state_ids, starts[0]


class TestDataCollatorForClef:
    def test_collator_shapes_and_alignment(self, tokenizer, decision_config):
        collator = DataCollatorForClef(tokenizer, decision_config)
        rows = [make_row(0), make_row(1, names=("team",)), make_row(2, names=("urgency", "refund"))]
        features = tokenized(collator, rows)
        assert features[0]["num_options"] == [2, 2, 3]  # template order: refund, team, urgency
        assert features[0]["is_choice"] == [False, True, False]
        batch = collator(features)

        total = 3 + 1 + 2
        assert batch["labels"].shape == (total, 3)
        assert batch["option_mask"].shape == (total, 3)
        assert batch["is_choice"].shape == (total,)
        assert batch["is_choice"].tolist() == [False, True, False, True, False, False]
        assert batch["rows_per_example"].tolist() == [3, 1, 2]
        assert batch["num_options"].tolist() == [[2, 2, 3], [2, 0, 0], [3, 2, 0]]
        assert batch["question_type_ids"].shape == (3, 3)
        assert batch["question_type_ids"][1, 1:].tolist() == [0, 0]
        # Every real question has a distribution over its options and nothing outside them
        assert torch.allclose(batch["labels"].sum(-1), torch.ones(total))
        assert (batch["labels"][~batch["option_mask"]] == 0).all()
        assert batch["option_mask"].sum(-1).tolist() == [2, 2, 3, 2, 3, 2]
        # Row 0: refund true, team alpha (options sorted: alpha, zeta), urgency level 0
        assert batch["labels"][:3].tolist() == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        # Row 1 asks only the team, with the answer zeta, the second option
        assert batch["labels"][3].tolist() == [0, 1, 0]
        # Marks hold two spans per question and per option, padded with -1
        expected_marks = [2 * sum(1 + count for count in row.tolist() if count) for row in batch["num_options"]]
        assert batch["marks"].shape == (3, max(expected_marks))
        for row, count in enumerate(expected_marks):
            assert (batch["marks"][row, :count] >= 0).all()
            assert (batch["marks"][row, count:] == -1).all()
        # Padding is on the right with the pad id, and the attention mask covers the real tokens
        lengths = [len(feature["input_ids"]) for feature in features]
        assert batch["input_ids"].shape == (3, max(lengths))
        assert batch["attention_mask"].sum(-1).tolist() == lengths
        assert (batch["input_ids"][batch["attention_mask"] == 0] == tokenizer.pad_token_id).all()
        # The model accepts the batch as it is, and returns one row of logits per question
        model = tiny_clef(tokenizer, decision_config).eval()
        inputs = {key: value for key, value in batch.items() if key not in {"is_choice", "rows_per_example"}}
        output = model(**inputs)
        assert output.logits.shape == batch["labels"].shape
        assert torch.isfinite(output.loss)

    def test_overlength_record_errors_or_is_skipped(self, tokenizer, decision_config):  # Review Focus 5
        with pytest.raises(ValueError, match="max_length"):
            DataCollatorForClef(tokenizer, decision_config, max_length=64).tokenize(LONG_ROW)
        skipped = DataCollatorForClef(tokenizer, decision_config, max_length=64, truncation="skip").tokenize(LONG_ROW)
        assert skipped["skip"] is True
        assert "max_length" in skipped["reason"]
        collator = DataCollatorForClef(tokenizer, decision_config, truncation="skip")
        assert collator.tokenize(make_row(0))["skip"] is False
        with pytest.raises(ValueError, match="Skipped"):
            collator([skipped])
        with pytest.raises(ValueError, match="truncation"):
            DataCollatorForClef(tokenizer, decision_config, truncation="truncate")

    def test_invalid_records_error_or_are_skipped(self, tokenizer, decision_config):
        nullable = {**QUESTIONS["team"], "allow_null": True}
        invalid = [
            ("invalid request", {"state": "s", "questions": {"team": {"type": "bogus"}}, "gold": {"team": "zeta"}}),
            ("invalid request", {"state": "s", "questions": ["team"], "gold": {"team": "zeta"}}),
            ("invalid request", {"state": "s", "gold": {"team": "zeta"}}),
            ("nullable question", {"state": "s", "questions": {"team": nullable}, "gold": {"team": "zeta"}}),
            ("unusable targets", {**make_row(0, names=("team",)), "gold": {"team": "omega"}}),
            ("unusable targets", {**make_row(0, names=("team",)), "gold": {}}),
        ]
        strict = DataCollatorForClef(tokenizer, decision_config)
        skipping = DataCollatorForClef(tokenizer, decision_config, truncation="skip")
        for reason, row in invalid:
            # A request as JSON too, as datasets store it
            for example in (row, {key: json.dumps(value) for key, value in row.items()}):
                with pytest.raises((ValueError, KeyError)):  # as before, when the record is not skipped
                    strict.tokenize(example)
                skipped = skipping.tokenize(example)
                assert skipped["skip"] is True and skipped["reason"] == reason
                assert skipped["input_ids"] == [] and skipped["labels"] == []
        assert skipping.tokenize(make_row(0))["skip"] is False

    def test_decision_config_takes_any_number_of_options(self, tokenizer, decision_config):
        # Clef reads no label token: a record can have more options than there are letters
        assert decision_config.labels == {"noul": None, "choice": None, "score": None}
        collator = DataCollatorForClef(tokenizer, decision_config)
        criteria = {f"option {index:02d}": None for index in range(29)}
        row = {
            "state": "s",
            "questions": {"answer": {"type": "choice", "criteria": criteria}},
            "gold": {"answer": "option 28"},
        }
        features = collator.tokenize(row)
        assert features["num_options"] == [29]
        assert features["labels"][0][28] == 1.0
        output = tiny_clef(tokenizer, decision_config)(
            **{k: v for k, v in collator([features]).items() if k not in {"is_choice", "rows_per_example"}}
        )
        assert output.logits.shape == (1, 29) and torch.isfinite(output.loss)

    def test_skip_reasons_are_grouped(self, tokenizer, decision_config):
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=64, truncation="skip")
        longer = {**LONG_ROW, "state": "word " * 300}
        reasons = [collator.tokenize(row)["reason"] for row in (LONG_ROW, longer)]
        assert reasons == ["longer than max_length (64 tokens)"] * 2

    def test_pad_falls_back_to_eos(self, tokenizer, decision_config):
        tokenizer.pad_token = None
        assert tokenizer.pad_token_id is None
        collator = DataCollatorForClef(tokenizer, decision_config)
        batch = collator([make_row(0), make_row(1, names=("team",))])
        padding = batch["input_ids"][batch["attention_mask"] == 0]
        assert len(padding) > 0
        assert (padding == tokenizer.eos_token_id).all()

    def test_permutation_keeps_each_question_with_its_target(self, tokenizer, decision_config):
        rows = [make_row(index) for index in range(6)]
        plain = DataCollatorForClef(tokenizer, decision_config)
        collator = DataCollatorForClef(tokenizer, decision_config, permute_fields=True, seed=1)
        features = tokenized(collator, rows)
        assert all("request" in feature for feature in features)
        # The target of each question, by name, from the unpermuted batch
        reference = plain(tokenized(plain, rows))
        assert all("request" not in feature for feature in tokenized(plain, rows))
        expected = {
            (row, name): reference["labels"][row * 3 + index].tolist()
            for row in range(6)
            for index, name in enumerate(question_names(tokenizer, reference, row))
        }
        expected_mask = {
            (row, name): reference["option_mask"][row * 3 + index].tolist()
            for row in range(6)
            for index, name in enumerate(question_names(tokenizer, reference, row))
        }
        orders = set()
        for _ in range(8):
            batch = collator(features)
            assert batch["labels"].shape == reference["labels"].shape
            for row in range(6):
                names = question_names(tokenizer, batch, row)
                orders.add(tuple(names))
                assert sorted(names) == ["refund", "team", "urgency"]
                for index, name in enumerate(names):
                    assert batch["labels"][row * 3 + index].tolist() == expected[row, name]
                    assert batch["option_mask"][row * 3 + index].tolist() == expected_mask[row, name]
        assert len(orders) > 1
        # Same seed, same permutations
        first = DataCollatorForClef(tokenizer, decision_config, permute_fields=True, seed=1)(features)
        second = DataCollatorForClef(tokenizer, decision_config, permute_fields=True, seed=1)(features)
        assert torch.equal(first["input_ids"], second["input_ids"])

    def test_eval_collator_does_not_permute(self, tokenizer, decision_config):
        rows = [make_row(index) for index in range(4)]
        permuting = DataCollatorForClef(tokenizer, decision_config, permute_fields=True)
        evaluating = DataCollatorForClef(tokenizer, decision_config)
        reference = evaluating(tokenized(evaluating, rows))
        for _ in range(3):
            batch = evaluating(tokenized(permuting, rows))
            assert all(torch.equal(batch[key], reference[key]) for key in reference)
            # Features tokenized by the eval collator carry no request, so even a permuting collator leaves them
            assert all(torch.equal(permuting(tokenized(evaluating, rows))[key], reference[key]) for key in reference)

    def test_failed_rerender_raises(self, tokenizer, decision_config):
        collator = DataCollatorForClef(tokenizer, decision_config, permute_fields=True)
        feature = collator.tokenize(make_row(0))
        broken = {**feature, "request": json.dumps({"state": "s", "questions": {"refund": {"type": "bogus"}}})}
        with pytest.raises(ValueError):
            collator([broken])
        mismatched = {**feature, "labels": feature["labels"][:2]}
        with pytest.raises(ValueError, match="questions"):
            collator([mismatched])

    def test_nullable_question_raises(self, tokenizer, decision_config):
        collator = DataCollatorForClef(tokenizer, decision_config)
        nullable = {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None}, "allow_null": True}
        row = {"state": "s", "questions": {"team": nullable}, "gold": {"team": "zeta"}}
        with pytest.raises(ValueError, match="nullable"):
            collator.tokenize(row)

    def test_request_json_and_targets(self, tokenizer, decision_config):
        collator = DataCollatorForClef(tokenizer, decision_config)
        row = make_row(0)
        as_request = {
            "request": json.dumps({"state": row["state"], "questions": row["questions"]}),
            "targets": json.dumps({"refund": {"p_true": 0.75}, "team": {"target_index": 1}, "urgency": {"score": 1}}),
        }
        features = collator.tokenize(as_request)
        assert features["labels"][0] == pytest.approx([0.75, 0.25])
        assert features["labels"][1] == [0.0, 1.0]
        assert features["labels"][2] == [0.0, 1.0, 0.0]

    def test_state_truncation_leaves_fitting_records(self, tokenizer, decision_config):
        strict = DataCollatorForClef(tokenizer, decision_config)
        cutting = DataCollatorForClef(tokenizer, decision_config, truncation="state")
        rows = [make_row(0), make_row(1, names=("team",)), {**make_row(2), "state": {"ticket": 2, "body": "hi"}}]
        for row in rows:
            features = cutting.tokenize(row)
            assert features == strict.tokenize(row)
            assert features["truncated"] is False and features["skip"] is False

    @pytest.mark.parametrize("prompt_style", ["chatml", "plain"])
    def test_state_truncation_cuts_the_end_of_the_state(self, tokenizer, prompt_style):
        if prompt_style == "plain":  # the encoder prompt, between [CLS] and [SEP]
            tokenizer = encoder_tokenizer(tokenizer)
        decision_config = ClefConfig.prompt_decision_config(prompt_style)
        unlimited = DataCollatorForClef(tokenizer, decision_config, max_length=100_000)
        full, state_ids, start = state_layout(tokenizer, unlimited, VARIED_ROW)
        length, end = len(full["input_ids"]), start + len(state_ids)
        schema_only = len(unlimited.tokenize({**VARIED_ROW, "state": ""})["input_ids"])
        assert schema_only == length - len(state_ids)
        if prompt_style == "plain":
            assert full["input_ids"][0] == tokenizer.cls_token_id and full["input_ids"][-1] == tokenizer.sep_token_id
        for max_length in (schema_only, schema_only + 1, schema_only + 17, length - 1):
            collator = DataCollatorForClef(tokenizer, decision_config, max_length=max_length, truncation="state")
            cut = collator.tokenize(VARIED_ROW)
            assert cut["skip"] is False and cut["truncated"] is True
            input_ids, kept = cut["input_ids"], max_length - schema_only
            assert len(input_ids) == max_length
            # The prompt before the state and the schema after it are untouched
            assert input_ids[:start] == full["input_ids"][:start]
            assert input_ids[start + kept :] == full["input_ids"][end:]
            # The state keeps its beginning: a strict prefix of its tokens
            assert input_ids[start : start + kept] == state_ids[:kept] and kept < len(state_ids)
            # Every instruction and option span reads the same text, through marks moved by the cut
            assert cut["marks"] == [mark - (length - max_length) for mark in full["marks"]]
            assert marked_spans(tokenizer, cut) == marked_spans(tokenizer, full)
            for key in ("labels", "num_options", "question_type_ids", "is_choice"):
                assert cut[key] == full[key], key
        # A record that fits is left as it is, and not counted as truncated
        fitting = DataCollatorForClef(tokenizer, decision_config, max_length=length, truncation="state")
        assert fitting.tokenize(VARIED_ROW) == full

    @pytest.mark.parametrize(
        "state",
        [
            # The state ends with the tokens before it, or begins with those after it: several splits fit
            VARIED_ROW["state"] + "\nSTATE:\n",
            "\n\nSCHEMA FIELDS:\n" + VARIED_ROW["state"],
        ],
    )
    def test_state_truncation_finds_an_ambiguous_state(self, tokenizer, decision_config, state, monkeypatch):
        row = {**VARIED_ROW, "state": state}
        unlimited = DataCollatorForClef(tokenizer, decision_config, max_length=100_000)
        full, state_ids, start = state_layout(tokenizer, unlimited, row)
        schema_only = len(full["input_ids"]) - len(state_ids)
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=schema_only + 9, truncation="state")
        rendered = []
        render = collator._render
        monkeypatch.setattr(collator, "_render", lambda request: rendered.append(request["state"]) or render(request))
        cut = collator.tokenize(row)
        # The prompt, the empty state, and the two states that tell where the state starts
        assert rendered == [state, "", "a", "b"]
        assert (
            cut["input_ids"] == full["input_ids"][:start] + state_ids[:9] + full["input_ids"][start + len(state_ids) :]
        )
        assert marked_spans(tokenizer, cut) == marked_spans(tokenizer, full)

    def test_state_truncation_skips_records_whose_schema_is_too_long(self, tokenizer, decision_config):
        unlimited = DataCollatorForClef(tokenizer, decision_config)
        schema_only = len(unlimited.tokenize({**VARIED_ROW, "state": ""})["input_ids"])
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=schema_only - 1, truncation="state")
        reason = f"schema longer than max_length ({schema_only - 1} tokens)"
        for row in (VARIED_ROW, LONG_ROW, {**VARIED_ROW, "state": ""}):
            skipped = collator.tokenize(row)
            assert skipped["skip"] is True and skipped["truncated"] is False
            assert skipped["reason"] == reason
            assert skipped["input_ids"] == [] and skipped["labels"] == []
        # Invalid records are skipped as with truncation="skip"
        invalid = collator.tokenize({"state": "s", "questions": {"team": {"type": "bogus"}}, "gold": {"team": "zeta"}})
        assert invalid["skip"] is True and invalid["reason"] == "invalid request"

    def test_state_truncation_needs_the_state_between_sentinels(self, tokenizer):
        decision_config = clef_decision_config()
        template = decision_config.chat_template
        joined = 'STATE:\\n<|split|>" ~ r(state) ~ "<|split|>\\n\\nSCHEMA FIELDS:\\n"'
        assert joined in template
        # The state now runs into the text around it, so the tokens next to it depend on it
        decision_config.chat_template = template.replace(joined, 'STATE:" ~ r(state) ~ "ing SCHEMA FIELDS:\\n"')
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=300, truncation="state")
        with pytest.raises(ValueError, match="sentinels"):
            collator.tokenize({**LONG_ROW, "state": "runn" * 200})

    def test_truncated_batch_trains(self, tokenizer, decision_config):
        unlimited = DataCollatorForClef(tokenizer, decision_config, max_length=100_000)
        rows = [VARIED_ROW, make_row(0), LONG_ROW]
        max_length = len(unlimited.tokenize({**VARIED_ROW, "state": ""})["input_ids"]) + 30
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=max_length, truncation="state")
        features = tokenized(collator, rows)
        assert [feature["truncated"] for feature in features] == [True, False, True]
        batch = collator(features)
        reference = unlimited(tokenized(unlimited, rows))
        assert batch["input_ids"].shape[1] == max_length
        assert batch["attention_mask"].sum(-1).tolist() == [len(feature["input_ids"]) for feature in features]
        # The targets of every question are those of the untruncated records, in the same order
        for key in ("labels", "option_mask", "is_choice", "rows_per_example", "num_options"):
            assert torch.equal(batch[key], reference[key]), key
        for row in range(len(rows)):
            assert question_names(tokenizer, batch, row) == question_names(tokenizer, reference, row)
        model = tiny_clef(tokenizer, decision_config)
        inputs = {key: value for key, value in batch.items() if key not in {"is_choice", "rows_per_example"}}
        output = model(**inputs)
        assert output.logits.shape == batch["labels"].shape
        assert torch.isfinite(output.loss)
        output.loss.backward()

    def test_permuted_prompts_are_cut_to_max_length(self, tokenizer, decision_config):
        rows = [VARIED_ROW, {**VARIED_ROW, "state": VARIED_ROW["state"][::-1]}, make_row(0)]
        unlimited = DataCollatorForClef(tokenizer, decision_config, max_length=100_000)
        max_length = len(unlimited.tokenize(VARIED_ROW)["input_ids"]) - 40
        collator = DataCollatorForClef(
            tokenizer, decision_config, max_length=max_length, truncation="state", permute_fields=True, seed=1
        )
        features = tokenized(collator, rows)
        assert [feature["truncated"] for feature in features] == [True, True, False]
        reference = unlimited(tokenized(unlimited, rows))
        expected = {
            (row, name): reference["labels"][row * 3 + index].tolist()
            for row in range(len(rows))
            for index, name in enumerate(question_names(tokenizer, reference, row))
        }
        orders = set()
        for _ in range(8):
            batch = collator(features)
            lengths = batch["attention_mask"].sum(-1).tolist()
            assert lengths[:2] == [max_length, max_length] and lengths[2] <= max_length
            for row in range(len(rows)):
                names = question_names(tokenizer, batch, row)
                orders.add(tuple(names))
                for index, name in enumerate(names):
                    assert batch["labels"][row * 3 + index].tolist() == expected[row, name]
        assert len(orders) > 1
        # Without the state mode, an over-length permutation raises rather than being cut
        strict = DataCollatorForClef(tokenizer, decision_config, max_length=max_length, permute_fields=True)
        with pytest.raises(ValueError, match="max_length"):
            strict([{**features[0], "truncated": False}])


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


def max_logit_difference(first, second, batch):
    inputs = {key: value for key, value in batch.items() if key not in {"is_choice", "labels", "rows_per_example"}}
    with torch.no_grad():
        return (first.eval()(**inputs).logits - second.eval()(**inputs).logits).abs().max().item()


class TestClefTraining(TrlTestCase):
    @pytest.mark.parametrize(
        "backbone, head_type",
        [
            ("qwen2", "joint"),
            ("bert", "joint"),
            ("bert", "cosine"),
            ("modernbert", "joint"),
            ("modernbert", "cosine"),
        ],
    )
    def test_train_eval_save_reload(self, tokenizer, decision_config, backbone, head_type):
        model = None
        if backbone != "qwen2":  # an encoder, on its plain prompt between [CLS] and [SEP]
            tokenizer = encoder_tokenizer(tokenizer)
            model = tiny_clef(tokenizer, None, backbone, head_type)
            decision_config = model.decision_config
            assert decision_config.chat_template != clef_decision_config().chat_template
        elif head_type == "cosine":
            model = tiny_clef(tokenizer, decision_config, head_type=head_type)
        trainer, collator, val = make_trainer(self.tmp_dir, tokenizer, decision_config, model=model)
        if backbone != "qwen2":
            assert val[0]["input_ids"][0] == tokenizer.cls_token_id
            assert val[0]["input_ids"][-1] == tokenizer.sep_token_id
        first = trainer.evaluate()["eval_loss"]
        trainer.train()
        trainer.save_model()
        assert trainer.evaluate()["eval_loss"] < first
        again = AutoModelForDecision.from_pretrained(self.tmp_dir).eval()
        assert isinstance(again, ClefForDecision)
        assert (again.config.head_type, type(again.model)) == (head_type, type(trainer.model.model))
        assert again.decision_config == decision_config
        batch = collator([val[i] for i in range(4)])
        assert max_logit_difference(again, trainer.model, batch) == 0.0

    def test_rlcd_eval_is_deterministic(self, tokenizer, decision_config):
        model = tiny_clef(tokenizer, decision_config)
        model.config.rlcd_weight, model.config.rlcd_samples = 0.25, 8
        trainer, _, _ = make_trainer(self.tmp_dir, tokenizer, decision_config, model=model)
        first = trainer.evaluate()["eval_loss"]
        assert trainer.evaluate()["eval_loss"] == first
        # The random policy term is a training term only: evaluation gives the supervised loss
        model.config.rlcd_weight = 0.0
        assert trainer.evaluate()["eval_loss"] == first

    def test_build_report_counts_skipped_records_by_reason(self, tokenizer, decision_config, capsys):
        invalid_request = {"state": "s", "questions": {"team": {"type": "bogus"}}, "gold": {"team": "zeta"}}
        unusable_gold = {**make_row(0, names=("team",)), "gold": {"team": "omega"}}
        longer = {**LONG_ROW, "state": "word " * 300}
        rows = [make_row(index) for index in range(3)] + [LONG_ROW, longer, invalid_request, unusable_gold]
        dataset = Dataset.from_list([{key: json.dumps(value) for key, value in row.items()} for row in rows])
        args = DecisionConfig(output_dir=self.tmp_dir, use_cpu=True, report_to="none")
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=400, truncation="skip")
        built, report = prepare_split(dataset, collator, args, "train")
        assert len(built) == 3
        assert report["rows"] == 7 and report["questions"] == 9
        assert report["skipped"] == {
            "longer than max_length (400 tokens)": 2,
            "invalid request": 1,
            "unusable targets": 1,
        }
        assert "Build report (train)" in capsys.readouterr().out
        # With truncation="error", an invalid record stops the build
        strict = DataCollatorForClef(tokenizer, decision_config, max_length=400)
        for row in (invalid_request, unusable_gold):
            records = Dataset.from_list(
                [{key: json.dumps(value) for key, value in r.items()} for r in (make_row(0), row)]
            )
            with pytest.raises(ValueError):
                prepare_split(records, strict, args, "train")

    def test_build_report_counts_truncated_records(self, tokenizer, decision_config):
        unlimited = DataCollatorForClef(tokenizer, decision_config, max_length=100_000)
        max_length = len(unlimited.tokenize({**VARIED_ROW, "state": ""})["input_ids"]) + 10
        criteria = {f"option {index:02d}": f"The description of option {index}" for index in range(40)}
        wide = {
            "state": "s",
            "questions": {"answer": {"type": "choice", "criteria": criteria}},
            "gold": {"answer": "option 03"},
        }
        assert len(unlimited.tokenize({**wide, "state": ""})["input_ids"]) > max_length
        invalid_request = {"state": "s", "questions": {"team": {"type": "bogus"}}, "gold": {"team": "zeta"}}
        rows = [make_row(index) for index in range(3)] + [VARIED_ROW, LONG_ROW, wide, invalid_request]
        dataset = Dataset.from_list([{key: json.dumps(value) for key, value in row.items()} for row in rows])
        args = DecisionConfig(output_dir=self.tmp_dir, use_cpu=True, report_to="none")
        collator = DataCollatorForClef(tokenizer, decision_config, max_length=max_length, truncation="state")
        built, report = prepare_split(dataset, collator, args, "train")
        assert len(built) == 5 and "truncated" not in built.column_names
        assert max(len(input_ids) for input_ids in built["input_ids"]) == max_length
        assert report == {
            "rows": 7,
            "questions": 15,
            "skipped": {f"schema longer than max_length ({max_length} tokens)": 1, "invalid request": 1},
            "truncated": 2,
        }
        # Without the state mode nothing is cut
        skipping = DataCollatorForClef(tokenizer, decision_config, max_length=max_length, truncation="skip")
        assert prepare_split(dataset, skipping, args, "train")[1]["truncated"] == 0

    def test_pipeline_on_saved_checkpoint(self, tokenizer, decision_config):
        model = tiny_clef(tokenizer, decision_config)
        model.save_pretrained(self.tmp_dir)
        tokenizer.save_pretrained(self.tmp_dir)
        decide = pipeline("decision", model=self.tmp_dir)
        answers = decide({"state": "ticket 3: refund alpha", "questions": QUESTIONS})
        assert set(answers) == {"refund", "team", "urgency"}
        assert {name: answer["type"] for name, answer in answers.items()} == {
            "refund": "noul",
            "team": "choice",
            "urgency": "score",
        }

    def run_main(
        self,
        tokenizer,
        mode,
        training_kwargs=None,
        backbone=None,
        script_kwargs=None,
        checkpoint=None,
        **model_kwargs,
    ):
        """
        Run the script's `main` on a tiny local backbone (the causal LM of `MODEL_ID` by default), or on the folder
        `checkpoint` as it is, and dataset; returns the trainer, the model `build_model` builds with the same seed, and
        the backbone.
        """
        if checkpoint is None:
            backbone = backbone if backbone is not None else AutoModelForCausalLM.from_pretrained(MODEL_ID)
            backbone.save_pretrained(f"{self.tmp_dir}/backbone")
            tokenizer.save_pretrained(f"{self.tmp_dir}/backbone")
            checkpoint = f"{self.tmp_dir}/backbone"
        rows = [make_row(index) for index in range(12)] + [LONG_ROW]
        with open(f"{self.tmp_dir}/train.jsonl", "w") as file:
            for row in rows:
                file.write(json.dumps({key: json.dumps(value) for key, value in row.items()}) + "\n")
        script_args = ClefScriptArguments(
            dataset_name=self.tmp_dir,
            dataset_config=None,
            validation_fraction=0.25,
            mode=mode,
            head_hidden_size=16,
            head_intermediate_size=32,
            head_num_layers=1,
            head_num_evidence_layers=1,
            head_num_attention_heads=2,
            max_length=400,
            truncation="skip",
            permute_fields=True,
            **(script_kwargs or {}),
        )
        training_args = dict(
            output_dir=f"{self.tmp_dir}/out",
            use_cpu=True,
            bf16=False,
            gradient_checkpointing=False,
            max_steps=5,
            learning_rate=1e-3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
        )
        training_args = DecisionConfig(**{**training_args, **(training_kwargs or {})})
        model_args = ClefModelArguments(model_name_or_path=checkpoint, dtype="float32", **model_kwargs)
        trainer = clef_main(script_args, training_args, model_args)
        set_seed(training_args.seed)
        initial = build_model(script_args, model_args, {"dtype": "float32", "revision": None})
        return trainer, initial, backbone

    def check_saved_model(self, trainer, tokenizer, **expected):
        """The saved output reloads with the expected config and exactly the logits of the trained model."""
        model = trainer.model
        reloaded = AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/out")
        assert isinstance(reloaded, ClefForDecision)
        for key, value in expected.items():
            assert getattr(reloaded.config, key) == value, key
        assert reloaded.decision_config == model.decision_config
        collator = DataCollatorForClef(tokenizer, model.decision_config)
        batch = collator([make_row(index) for index in range(3)])
        assert max_logit_difference(reloaded, model, batch) == 0.0
        return reloaded

    @pytest.mark.parametrize("kind", ["bert", "bert-reranker", "modernbert"])
    def test_main_on_an_encoder_checkpoint(self, tokenizer, kind, capsys):
        # BERT has a causal LM class but is an encoder; ModernBERT has none; a reranker registers only its classifier
        tokenizer = encoder_tokenizer(tokenizer)
        backbone = tiny_encoder(tokenizer, kind)
        trainer, initial, _ = self.run_main(tokenizer, "full", backbone=backbone)
        assert trainer.state.global_step == 5
        model_type = "modernbert" if kind == "modernbert" else "bert"
        assert (
            f"Clef setup: bidirectional backbone ({model_type}), global_pooling=first, prompt_style=plain, "
            "head_type=joint" in capsys.readouterr().out
        )
        assert type(trainer.model.model).__name__ == ("ModernBertModel" if kind == "modernbert" else "BertModel")
        assert hasattr(trainer.model, "lm_head") == (kind != "modernbert")
        reloaded = self.check_saved_model(
            trainer, tokenizer, global_pooling="first", prompt_style="plain", head_type="joint"
        )
        # The plain prompt, between [CLS] and [SEP]
        ((inputs, _),) = reloaded.decision_config.prepare_inputs(make_row(0), tokenizer)
        assert (
            inputs["input_ids"][0, 0] == tokenizer.cls_token_id
            and inputs["input_ids"][0, -1] == tokenizer.sep_token_id
        )
        # The new model starts from the encoder's weights (the reranker's unwrapped `base_model`)
        expected = backbone.base_model.state_dict()
        built = initial.model.state_dict()
        assert set(built) == set(expected)
        for name, value in built.items():
            assert torch.equal(value, expected[name]), name

    def test_unwrappable_reranker_raises(self, tokenizer, monkeypatch):
        from examples.decisions import clef_typed_decisions

        tokenizer = encoder_tokenizer(tokenizer)
        tiny_encoder(tokenizer, "bert-reranker").save_pretrained(f"{self.tmp_dir}/reranker")
        config = AutoConfig.from_pretrained(f"{self.tmp_dir}/reranker")

        class Opaque(torch.nn.Module):
            base_model = property(lambda self: self)

        info = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set()}
        monkeypatch.setattr(
            clef_typed_decisions.AutoModelForSequenceClassification,
            "from_pretrained",
            lambda *a, **k: (Opaque(), info),
        )
        with pytest.raises(ValueError, match="cannot be unwrapped"):
            load_backbone(f"{self.tmp_dir}/reranker", config, {})

    @pytest.mark.parametrize("backbone_kind", ["qwen2", "modernbert"])
    def test_main_with_the_cosine_head(self, tokenizer, backbone_kind, capsys):
        backbone = None
        if backbone_kind != "qwen2":
            tokenizer = encoder_tokenizer(tokenizer)
            backbone = tiny_encoder(tokenizer, backbone_kind)
        script_kwargs = {"head_type": "cosine", "cosine_width": 8, "cosine_initial_scale": 5.0}
        trainer, _, _ = self.run_main(tokenizer, "full", backbone=backbone, script_kwargs=script_kwargs)
        assert trainer.state.global_step == 5
        assert "head_type=cosine" in capsys.readouterr().out
        assert not hasattr(trainer.model, "lm_head")
        assert trainer.model.cosine_head.question_projection.out_features == 8
        self.check_saved_model(trainer, tokenizer, head_type="cosine", cosine_width=8, cosine_initial_scale=5.0)

    def test_head_mode_trains_the_cosine_head_only(self, tokenizer):
        set_seed(0)
        trainer, initial, backbone = self.run_main(tokenizer, "head", script_kwargs={"head_type": "cosine"})
        model = trainer.model
        trainable = {name.split(".")[0] for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable == {"cosine_head"}
        initial_parameters = dict(initial.named_parameters())
        assert any(
            not torch.equal(parameter, initial_parameters[name])
            for name, parameter in model.named_parameters()
            if name.startswith("cosine_head.")
        )
        loaded = backbone.base_model.state_dict()
        for name, value in model.model.state_dict().items():
            torch.testing.assert_close(value.float(), loaded[name].float())
        self.check_saved_model(trainer, tokenizer, head_type="cosine")

    @require_peft
    def test_lora_mode_with_the_cosine_head(self, tokenizer):
        lora = {"use_peft": True, "lora_r": 4, "lora_target_modules": ["q_proj", "v_proj"]}
        trainer, initial, backbone = self.run_main(tokenizer, "lora", script_kwargs={"head_type": "cosine"}, **lora)
        model = trainer.model
        assert type(model.model).__name__ == "Qwen2Model"  # merged
        # The merged backbone differs from the loaded one only where adapters sit
        loaded = backbone.base_model.state_dict()
        for name, value in model.model.state_dict().items():
            same = torch.equal(value.float(), loaded[name].float())
            assert same == (not name.endswith(("q_proj.weight", "v_proj.weight"))), name
        initial_parameters = dict(initial.named_parameters())
        assert any(
            not torch.equal(parameter, initial_parameters[name])
            for name, parameter in model.named_parameters()
            if name.startswith("cosine_head.")
        )
        self.check_saved_model(trainer, tokenizer, head_type="cosine")
        assert (Path(self.tmp_dir) / "out" / "adapter" / "adapter_model.safetensors").exists()

    @pytest.mark.parametrize(
        "backbone_kind, settings",
        [
            ("qwen2", {"global_pooling": "first", "prompt_style": "plain"}),
            ("modernbert", {"global_pooling": "last", "prompt_style": "chatml"}),
            ("modernbert", {"global_pooling": "last"}),
        ],
    )
    def test_explicit_pooling_and_prompt_style_win(self, tokenizer, backbone_kind, settings, capsys):
        backbone = None
        if backbone_kind != "qwen2":
            tokenizer = encoder_tokenizer(tokenizer)
            backbone = tiny_encoder(tokenizer, backbone_kind)
        trainer, _, _ = self.run_main(tokenizer, "full", backbone=backbone, script_kwargs=settings)
        defaults = (
            {"global_pooling": "last", "prompt_style": "chatml"}
            if backbone_kind == "qwen2"
            else {"global_pooling": "first", "prompt_style": "plain"}
        )
        expected = {**defaults, **settings}
        out = capsys.readouterr().out
        assert f"global_pooling={expected['global_pooling']}, prompt_style={expected['prompt_style']}" in out
        self.check_saved_model(trainer, tokenizer, **expected)

    def test_main_passes_rlcd_settings(self, tokenizer):
        script_kwargs = {"rlcd_weight": 0.25, "rlcd_samples": 8}
        trainer, initial, _ = self.run_main(
            tokenizer, "full", training_kwargs={"logging_steps": 1}, script_kwargs=script_kwargs
        )
        assert trainer.state.global_step == 5
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        assert losses and all(np.isfinite(loss) for loss in losses)
        assert (initial.config.rlcd_weight, initial.config.rlcd_samples) == (0.25, 8)
        # The settings left unset keep the config's defaults
        self.check_saved_model(
            trainer, tokenizer, rlcd_weight=0.25, rlcd_samples=8, rlcd_sigma=0.4, rlcd_normalize_advantage=False
        )

    def test_checkpoint_head_settings_cannot_change(self, tokenizer, decision_config):
        tiny_clef(tokenizer, decision_config).save_pretrained(f"{self.tmp_dir}/clef")
        model_args = ClefModelArguments(model_name_or_path=f"{self.tmp_dir}/clef", dtype="float32")
        script_args = ClefScriptArguments(head_type="cosine")
        with pytest.raises(ValueError, match="cannot change"):
            build_model(script_args, model_args, {"dtype": "float32"})
        # The settings it already has are fine
        model = build_model(ClefScriptArguments(head_type="joint"), model_args, {"dtype": "float32"})
        assert model.config.head_type == "joint"

    def test_main_builds_plain_backbone_and_reports(self, tokenizer, decision_config, capsys):
        # A local causal LM becomes the backbone of a new head; the build report counts skipped records
        set_seed(0)
        trainer, _, backbone = self.run_main(tokenizer, "head")
        report = capsys.readouterr().out
        assert "Build report (train)" in report and "Build report (eval)" in report
        assert "longer than max_length" in report
        saved = AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/out")
        assert isinstance(saved, ClefForDecision)
        assert saved.decision_config.chat_template == decision_config.chat_template
        # Head-only: the backbone is the one that was loaded, and the head moved from its seeded initialization
        loaded = copy.deepcopy(backbone.base_model.state_dict())
        for name, value in saved.model.state_dict().items():
            torch.testing.assert_close(value.float(), loaded[name].float())
        set_seed(trainer.args.seed)
        initial = ClefForDecision(copy.deepcopy(saved.config))
        assert any(
            not torch.equal(parameter, dict(initial.scorer.named_parameters())[name])
            for name, parameter in saved.scorer.named_parameters()
        )

    @pytest.mark.parametrize("mode", ["full", pytest.param("lora", marks=require_peft)])
    def test_main_trains_with_activation_checkpointing(self, tokenizer, mode, capsys):
        lora = {"use_peft": True, "lora_r": 4, "lora_target_modules": ["q_proj", "v_proj"]} if mode == "lora" else {}
        trainer, initial, _ = self.run_main(tokenizer, mode, training_kwargs={"gradient_checkpointing": True}, **lora)
        assert trainer.state.global_step == 5
        model = trainer.model
        # The layers of the backbone and of the head were checkpointed
        layers = [module for module in model.modules() if hasattr(module, "gradient_checkpointing")]
        assert layers and all(module.gradient_checkpointing for module in layers)
        assert any(name.startswith("encoder.") for name, module in model.named_modules() if module in layers)
        assert all(np.isfinite(entry["loss"]) for entry in trainer.state.log_history if "loss" in entry)
        initial_parameters = dict(initial.named_parameters())
        assert any(
            not torch.equal(parameter, initial_parameters[name])
            for name, parameter in model.named_parameters()
            if name.startswith("scorer.")
        )

    @require_peft
    def test_main_lora_mode(self, tokenizer, decision_config, monkeypatch):
        from peft import PeftModel
        from peft.tuners.lora import LoraModel
        from safetensors.torch import load_file

        seen = {}
        merge = LoraModel.merge_and_unload

        def record(peft_model, *args, **kwargs):
            seen["trainable"] = {name for name, parameter in peft_model.named_parameters() if parameter.requires_grad}
            seen["adapters"] = {
                name: parameter.detach().clone()
                for name, parameter in peft_model.named_parameters()
                if "lora_" in name
            }
            return merge(peft_model, *args, **kwargs)

        monkeypatch.setattr(LoraModel, "merge_and_unload", record)
        trainer, initial, backbone = self.run_main(
            tokenizer, "lora", use_peft=True, lora_r=4, lora_target_modules=["q_proj", "v_proj"]
        )
        model = trainer.model
        # The adapters, and only they, were trainable in the backbone; they moved off their zero initialization
        assert seen["trainable"] and all("lora_" in name for name in seen["trainable"])
        assert all(".layers." in name and ("q_proj" in name or "v_proj" in name) for name in seen["trainable"])
        assert any(parameter.abs().sum() > 0 for name, parameter in seen["adapters"].items() if "lora_B" in name)
        # The head is trained in full: it is exactly the trainable part of the model, and it moved
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable and {name.split(".")[0] for name in trainable} == {"embeddings", "encoder", "scorer"}
        assert {
            name for name, _ in model.named_parameters() if name.split(".")[0] in {"embeddings", "encoder", "scorer"}
        } == trainable
        initial_parameters = dict(initial.named_parameters())
        for part in ("encoder", "scorer"):
            assert any(
                not torch.equal(parameter, initial_parameters[name])
                for name, parameter in model.named_parameters()
                if name.startswith(part)
            )
        # The merged backbone differs from the loaded one only where adapters sit
        loaded = backbone.base_model.state_dict()
        for name, value in model.model.state_dict().items():
            same = torch.equal(value.float(), loaded[name].float())
            assert same == (not name.endswith(("q_proj.weight", "v_proj.weight"))), name
        # The output is a native Clef checkpoint with the same logits; the adapter is saved on its own
        out = f"{self.tmp_dir}/out"
        reloaded = AutoModelForDecision.from_pretrained(out)
        assert isinstance(reloaded, ClefForDecision)
        collator = DataCollatorForClef(tokenizer, decision_config)
        batch = collator([make_row(index) for index in range(3)])
        assert max_logit_difference(reloaded, model, batch) == 0.0
        # The adapter, on the original backbone (`AutoModel`), rebuilds the backbone of the saved checkpoint
        assert all("lora_" in name for name in load_file(f"{out}/adapter/adapter_model.safetensors"))
        adapted = PeftModel.from_pretrained(
            AutoModel.from_pretrained(f"{self.tmp_dir}/backbone", dtype=torch.float32), f"{out}/adapter"
        )
        merged = adapted.merge_and_unload().base_model.state_dict()

        for name, value in reloaded.model.state_dict().items():
            torch.testing.assert_close(value.float(), merged[name].float())

    def build_from(self, folder, **script_kwargs):
        """The model the script builds from `folder`, with a tiny head."""
        script_args = ClefScriptArguments(
            head_hidden_size=16,
            head_intermediate_size=32,
            head_num_layers=1,
            head_num_evidence_layers=1,
            head_num_attention_heads=2,
            **script_kwargs,
        )
        model_args = ClefModelArguments(model_name_or_path=folder, dtype="float32")
        return build_model(script_args, model_args, {"dtype": "float32", "revision": None})

    @staticmethod
    def no_training(monkeypatch):
        """Fail the test if the script gets as far as building its trainer."""
        from examples.decisions import clef_typed_decisions

        def trainer(*args, **kwargs):
            raise AssertionError("training started")

        monkeypatch.setattr(clef_typed_decisions, "DecisionTrainer", trainer)

    def test_reranker_package_backbone_is_loaded_not_reinitialised(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        folder = reranker_package(
            f"{self.tmp_dir}/reranker", backbone, tokenizer, {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        )
        # As measured on the real package: `AutoModel` alone loads none of its backbone, with only a warning
        _, info = AutoModel.from_pretrained(folder, output_loading_info=True)
        assert set(info["missing_keys"]) == set(backbone.state_dict())
        model = self.build_from(folder)
        assert type(model.model).__name__ == "UltrabertModel"
        assert not hasattr(model, "lm_head")
        expected, built = backbone.state_dict(), model.model.state_dict()
        assert set(built) == set(expected)
        for name, value in built.items():
            assert torch.equal(value, expected[name]), name
        # The settings of the remote config are read: the windows and RoPE bases of the layers
        text_config = model.config.text_config
        assert text_config.layer_types == ["sliding_attention", "sliding_attention", "full_attention"]
        assert text_config.sliding_window == 8
        assert text_config.rope_parameters == backbone.config.rope_parameters
        assert (model.config.global_pooling, model.config.prompt_style) == ("first", "plain")
        line = describe(model)
        assert "prefix" in line and "backbone." in line
        assert 'backbone weights: prefix "backbone."' in line
        assert "dropped 3 non-backbone tensors" in line
        assert all(key in line for key in PAIR_HEAD)

    def test_missing_backbone_weights_raise(self, tokenizer, monkeypatch):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        keys = list(backbone.state_dict())
        kept, removed = keys[::2], keys[1::2]
        folder = reranker_package(
            f"{self.tmp_dir}/half", backbone, tokenizer, {**prefixed(backbone, "backbone.", kept), **PAIR_HEAD}
        )
        self.no_training(monkeypatch)
        with pytest.raises(ValueError, match="missing") as error:
            self.run_main(tokenizer, "full", checkpoint=folder)
        message = str(error.value)
        assert removed[0] in message
        assert sum(key in message for key in removed) == min(10, len(removed))
        assert not Path(self.tmp_dir, "out").exists()
        # Every key under the prefix but one of the wrong shape: refused too
        tensors = {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        tensors["backbone.norm.weight"] = torch.ones(16)
        folder = reranker_package(f"{self.tmp_dir}/shape", backbone, tokenizer, tensors)
        with pytest.raises(ValueError, match="norm.weight"):
            self.build_from(folder)

    def test_ambiguous_prefix_raises(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        keys = list(backbone.state_dict())
        # Split across two prefixes: no single one holds every key
        split = {**prefixed(backbone, "backbone.", keys[::2]), **prefixed(backbone, "encoder.", keys[1::2])}
        folder = reranker_package(f"{self.tmp_dir}/split", backbone, tokenizer, split)
        with pytest.raises(ValueError, match="missing"):
            self.build_from(folder)
        # Two whole copies under two prefixes: which one is the backbone is not guessed
        teacher = {f"teacher.{key}": torch.zeros_like(value) for key, value in backbone.state_dict().items()}
        copies = {**prefixed(backbone, "backbone."), **teacher}
        folder = reranker_package(f"{self.tmp_dir}/copies", backbone, tokenizer, copies)
        with pytest.raises(ValueError, match="ambiguous") as error:
            self.build_from(folder)
        assert '"backbone."' in str(error.value) and '"teacher."' in str(error.value)

    def test_native_ultrabert_checkpoint_loads_directly(self, tokenizer):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        backbone.save_pretrained(f"{self.tmp_dir}/native")
        tokenizer.save_pretrained(f"{self.tmp_dir}/native")
        model = self.build_from(f"{self.tmp_dir}/native")
        assert type(model.model).__name__ == "UltrabertModel"
        expected, built = backbone.state_dict(), model.model.state_dict()
        assert set(built) == set(expected)
        for name, value in built.items():
            assert torch.equal(value, expected[name]), name
        line = describe(model)
        assert "backbone weights: direct" in line and "dropped 0 non-backbone tensors" in line
        assert "prefix" not in line

    @pytest.mark.parametrize("head_type", ["joint", "cosine"])
    def test_main_trains_on_ultrabert_reranker_package(self, tokenizer, head_type, capsys):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        folder = reranker_package(
            f"{self.tmp_dir}/reranker", backbone, tokenizer, {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        )
        script_kwargs = {"head_type": head_type, "cosine_width": 8}
        trainer, initial, _ = self.run_main(tokenizer, "full", checkpoint=folder, script_kwargs=script_kwargs)
        assert trainer.state.global_step == 5
        out = capsys.readouterr().out
        assert (
            f"Clef setup: bidirectional backbone (ultrabert), global_pooling=first, prompt_style=plain, "
            f'head_type={head_type}, backbone weights: prefix "backbone."' in out
        )
        assert type(trainer.model.model).__name__ == "UltrabertModel"
        assert not hasattr(trainer.model, "lm_head")
        reloaded = self.check_saved_model(
            trainer, tokenizer, global_pooling="first", prompt_style="plain", head_type=head_type
        )
        assert reloaded.config.text_config.layer_types == backbone.config.layer_types
        # The plain prompt starts with the beginning of sequence token and ends with the separator
        ((inputs, _),) = reloaded.decision_config.prepare_inputs(make_row(0), tokenizer)
        assert inputs["input_ids"][0, 0] == tokenizer.bos_token_id
        assert inputs["input_ids"][0, -1] == tokenizer.sep_token_id
        # The model starts from the reranker's backbone
        expected = backbone.state_dict()
        for name, value in initial.model.state_dict().items():
            assert torch.equal(value, expected[name]), name

    def test_remote_code_is_never_executed(self, tokenizer, capsys):
        import sys

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
        trainer, _, _ = self.run_main(tokenizer, "full", checkpoint=folder)
        assert trainer.state.global_step == 5
        assert 'backbone weights: prefix "backbone."' in capsys.readouterr().out
        assert not sentinel.exists()
        assert not any("modeling_reranker" in name or "configuration_ultrabert." in name for name in sys.modules)
        # Asking for remote code is refused rather than ignored
        with pytest.raises(ValueError, match="remote code"):
            remote = {"model_init_kwargs": {"trust_remote_code": True}}
            self.run_main(tokenizer, "full", checkpoint=folder, training_kwargs=remote)
        assert not sentinel.exists()

    @pytest.mark.parametrize(
        "settings", [{"rlcd_weight": -0.25}, {"rlcd_sigma": 0.0}, {"rlcd_samples": 1}, {"rlcd_sigma": -0.4}]
    )
    def test_loaded_checkpoint_loss_settings_are_validated(self, tokenizer, decision_config, settings, monkeypatch):
        # The loss settings applied to a loaded Clef checkpoint are checked as a new config checks them
        tiny_clef(tokenizer, decision_config).save_pretrained(f"{self.tmp_dir}/clef")
        tokenizer.save_pretrained(f"{self.tmp_dir}/clef")
        self.no_training(monkeypatch)
        (key,) = settings
        with pytest.raises(ValueError, match=key):
            self.run_main(tokenizer, "full", checkpoint=f"{self.tmp_dir}/clef", script_kwargs=settings)
        assert not Path(self.tmp_dir, "out").exists()
        # Valid settings are applied
        model = self.build_from(f"{self.tmp_dir}/clef", rlcd_weight=0.25, rlcd_sigma=0.3, rlcd_samples=4)
        assert (model.config.rlcd_weight, model.config.rlcd_sigma, model.config.rlcd_samples) == (0.25, 0.3, 4)

    def test_clef_checkpoint_with_missing_weights_raises(self, tokenizer, decision_config):
        from safetensors.torch import load_file, save_file

        folder = f"{self.tmp_dir}/clef"
        tiny_clef(tokenizer, decision_config).save_pretrained(folder)
        tensors = load_file(f"{folder}/model.safetensors")
        removed = next(key for key in tensors if key.startswith("model.layers."))
        del tensors[removed]
        save_file(tensors, f"{folder}/model.safetensors", metadata={"format": "pt"})
        with pytest.raises(ValueError, match="missing"):
            self.build_from(folder)


def lora_clef(tokenizer, decision_config):
    """The tiny Clef with LoRA adapters on its backbone and the head trained in full, as in the example's lora mode."""
    from peft import LoraConfig, get_peft_model

    model = tiny_clef(tokenizer, decision_config)
    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    lora_config = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION")
    model.model = get_peft_model(model.model, lora_config)
    return model


def checkpoint_state_dict(checkpoint, model):
    """
    The weights of a checkpoint in the layout of `model`, read without the trainer: every key that
    `AutoModelForDecision.from_pretrained` loads (it converts the saved head layout), and the keys a PEFT backbone
    stores as they are in the file.
    """
    from safetensors.torch import load_file

    loaded = AutoModelForDecision.from_pretrained(checkpoint).state_dict()
    raw = load_file(f"{checkpoint}/model.safetensors")
    expected = {}
    for name in model.state_dict():
        if name in loaded:
            expected[name] = loaded[name]
        else:
            assert name in raw, f"{name} is not in the checkpoint"
            expected[name] = raw[name]
    return expected


def max_state_difference(model_or_state, expected, prefixes=None):
    state = model_or_state if isinstance(model_or_state, dict) else model_or_state.state_dict()
    differences = [
        (value.detach().cpu().float() - expected[name].float()).abs().max().item()
        for name, value in state.items()
        if prefixes is None or name.startswith(prefixes)
    ]
    assert differences
    return max(differences)


HEAD = ("embeddings.", "encoder.", "scorer.")


class TestClefCheckpointRestore(TrlTestCase):
    """The head of Clef is saved in the layout of the original checkpoints; loading a checkpoint must convert it back."""

    def ranked_trainer(self, tokenizer, decision_config, model, **kwargs):
        # The metric ranks earlier evaluations higher, so the best checkpoint is the first one, not the last
        evaluations = []

        def compute_metrics(prediction):
            evaluations.append(None)
            return {"rank": -len(evaluations)}

        args = dict(
            max_steps=6,
            eval_steps=2,
            save_strategy="steps",
            save_steps=2,
            metric_for_best_model="eval_rank",
            greater_is_better=True,
            compute_metrics=compute_metrics,
        )
        trainer, _, _ = make_trainer(self.tmp_dir, tokenizer, decision_config, model=model, **{**args, **kwargs})
        return trainer

    def check_best_model_is_restored(self, tokenizer, decision_config, model):
        from transformers.testing_utils import CaptureLogger
        from transformers.utils import logging

        trainer = self.ranked_trainer(tokenizer, decision_config, model, load_best_model_at_end=True)
        with CaptureLogger(logging.get_logger("transformers")) as captured:
            trainer.train()
        assert trainer.state.best_model_checkpoint == f"{self.tmp_dir}/checkpoint-2"
        assert "missing keys" not in captured.out and "unexpected keys" not in captured.out, captured.out
        best = checkpoint_state_dict(f"{self.tmp_dir}/checkpoint-2", trainer.model)
        last = checkpoint_state_dict(f"{self.tmp_dir}/checkpoint-6", trainer.model)
        # The head trained on after the best step, so restoring the best checkpoint changes it
        assert max_state_difference(trainer.model, last, HEAD) > 0
        assert max_state_difference(trainer.model, best, HEAD) == 0.0
        assert max_state_difference(trainer.model, best) == 0.0

    def check_resume_loads_checkpoint(self, tokenizer, decision_config, make_model):
        from transformers.testing_utils import CaptureLogger
        from transformers.utils import logging

        # Both models are built first: training adds the template's tokens to the tokenizer
        first, model = make_model(), make_model()
        self.ranked_trainer(tokenizer, decision_config, first, max_steps=4).train()
        checkpoint = f"{self.tmp_dir}/checkpoint-2"
        # A fresh model, perturbed so that nothing of it matches the checkpoint by chance
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        trainer = self.ranked_trainer(tokenizer, decision_config, model, max_steps=3, eval_strategy="no")
        expected = checkpoint_state_dict(checkpoint, trainer.model)
        assert max_state_difference(trainer.model, expected, HEAD) > 0
        # The weights right after the trainer loads the checkpoint, before it trains on
        loaded = {}
        load = trainer._load_from_checkpoint

        def load_and_record(*args, **kwargs):
            load(*args, **kwargs)
            loaded.update({name: value.detach().clone() for name, value in trainer.model.state_dict().items()})

        trainer._load_from_checkpoint = load_and_record
        with CaptureLogger(logging.get_logger("transformers")) as captured:
            trainer.train(resume_from_checkpoint=checkpoint)
        assert "missing keys" not in captured.out and "unexpected keys" not in captured.out, captured.out
        assert max_state_difference(loaded, expected) == 0.0

    def test_load_best_model_at_end_restores_head_and_backbone(self, tokenizer, decision_config):
        self.check_best_model_is_restored(tokenizer, decision_config, tiny_clef(tokenizer, decision_config))

    @require_peft
    def test_load_best_model_at_end_with_lora_backbone(self, tokenizer, decision_config):
        self.check_best_model_is_restored(tokenizer, decision_config, lora_clef(tokenizer, decision_config))

    def test_resume_from_checkpoint_loads_head_and_backbone(self, tokenizer, decision_config):
        self.check_resume_loads_checkpoint(tokenizer, decision_config, lambda: tiny_clef(tokenizer, decision_config))

    @require_peft
    def test_resume_from_checkpoint_with_lora_backbone(self, tokenizer, decision_config):
        self.check_resume_loads_checkpoint(tokenizer, decision_config, lambda: lora_clef(tokenizer, decision_config))


class TestRecordRows(TrlTestCase):
    """`rows_per_example` makes the trainer gather whole records; on one process it changes no metric."""

    def test_pack_unpack_round_trip(self):
        rows_per_example = torch.tensor([3, 1, 2, 1])
        rows = torch.arange(7 * 2 * 3, dtype=torch.float).reshape(7, 2, 3)
        packed = pack_rows(rows, rows_per_example, 4, -100.0)
        assert packed.shape == (4, 4, 2, 3)
        assert torch.equal(unpack_rows(packed, rows_per_example), rows)
        # Record 1 holds row 3, record 2 rows 4 and 5; every other slot is padding
        assert torch.equal(packed[1, 0], rows[3]) and torch.equal(packed[2, :2], rows[4:6])
        is_padding = torch.arange(4)[None, :] >= rows_per_example[:, None]
        assert (packed[is_padding] == -100).all()
        with pytest.raises(ValueError, match="adds up"):
            pack_rows(rows[:6], rows_per_example, 4, 0.0)
        with pytest.raises(ValueError, match="more than"):
            pack_rows(rows, rows_per_example, 2, 0.0)

    def test_padded_rows_contribute_nothing(self, tokenizer, decision_config):
        collator = DataCollatorForClef(tokenizer, decision_config)
        # Records with three, one and two questions, so batches hold different numbers of rows and options, and the
        # last batch is smaller
        names = [("refund", "team", "urgency"), ("team",), ("urgency", "refund"), ("refund",), ("team", "urgency")]
        val = Dataset.from_list([collator.tokenize(make_row(index, names[index % 5])) for index in range(11)])

        def without_rows(features):
            batch = collator(features)
            del batch["rows_per_example"]
            return batch

        # Both models are built first: a trainer adds the template's tokens to the tokenizer
        models = {"records": tiny_clef(tokenizer, decision_config), "rows": tiny_clef(tokenizer, decision_config)}
        # The reference evaluates all records in one batch: without `rows_per_example`, the trainer can only trim the
        # padding of a last, smaller batch in records, from its rows
        batch_sizes = {"records": 4, "rows": len(val)}
        results = {}
        for name, data_collator in (("records", collator), ("rows", without_rows)):
            trainer, _, _ = make_trainer(
                self.tmp_dir,
                tokenizer,
                decision_config,
                model=models[name],
                compute_metrics=compute_calibration_metrics,
                per_device_eval_batch_size=batch_sizes[name],
            )
            trainer.data_collator = data_collator
            assert (name == "records") == ("rows_per_example" in trainer.data_collator([val[0], val[1]]))
            metrics = trainer.evaluate(val)
            prediction = trainer.predict(val)
            results[name] = metrics, prediction
        (packed_metrics, packed), (row_metrics, rows) = results["records"], results["rows"]
        # Every metric of the trainer and of `compute_calibration_metrics` is the same
        # (but `eval_loss`, which Trainer averages over batches, weighting each by its number of records)
        timings = ("runtime", "per_second", "preparation_time")
        keys = {key for key in row_metrics if not key.endswith(timings) and key != "eval_loss"}
        assert keys == {key for key in packed_metrics if key in keys} and len(keys) > 10
        for key in keys:
            assert packed_metrics[key] == pytest.approx(row_metrics[key], abs=1e-6), key
        # The packed predictions are the rows, with padded question slots of -100 in between
        total = sum(len(row["num_options"]) for row in val)
        assert rows.predictions.shape[0] == total
        assert packed.predictions.shape[0] > total
        real = (packed.predictions != -100).any(-1)
        assert real.sum() == total
        np.testing.assert_allclose(packed.predictions[real], rows.predictions, atol=1e-6)
        np.testing.assert_array_equal(packed.label_ids[real], rows.label_ids)
        assert (packed.label_ids[~real] == -100).all()
