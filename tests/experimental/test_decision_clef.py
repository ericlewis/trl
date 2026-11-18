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
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset


pytest.importorskip("transformers.decision")
from transformers.loss.loss_utils import LOSS_MAPPING  # noqa: E402


# Transformers #50058 adds Clef; its training (native loss, batched forward, head choice, RLCD) comes on top of it
if "ClefForDecision" not in LOSS_MAPPING:
    pytest.skip(
        "Requires a Transformers build with Clef training support (native loss and batched forward) on top of #50058",
        allow_module_level=True,
    )

from examples.decisions.clef_typed_decisions import (
    ClefModelArguments,
    ClefScriptArguments,
    DataCollatorForClef,
    build_model,
    clef_decision_config,
    prepare_split,
    sample_rows,
    set_head_learning_rate,
)
from examples.decisions.clef_typed_decisions import main as clef_main
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForDecision,
    AutoTokenizer,
    BertConfig,
    ClefConfig,
    ClefForDecision,
    TrainerState,
    pipeline,
    set_seed,
)

from trl.experimental.decision import (
    DecisionConfig,
    DecisionTrainer,
    RlcdScheduleCallback,
    compute_calibration_metrics,
)
from trl.experimental.decision.decision_trainer import pack_rows, unpack_rows

from ..testing_utils import TrlTestCase, require_peft
from .decision_clef_helpers import (
    MODEL_ID,
    PAIR_HEAD,
    QUESTIONS,
    RAISED_CONTEXT,
    SPARK_PACKAGE,
    encoder_tokenizer,
    lora_clef,
    make_row,
    make_trainer,
    prefixed,
    reranker_package,
    spark_package,
    tiny_clef,
    tiny_encoder,
    tiny_ultrabert,
    ultrabert_tokenizer,
)


INSTRUCTIONS = {question["instructions"]: name for name, question in QUESTIONS.items()}
LONG_ROW = {"state": "word " * 200, "questions": QUESTIONS, "gold": {"refund": True, "team": "zeta", "urgency": 0}}
# A long state of distinct words, so that a cut shows which end of the state was kept
VARIED_ROW = {**LONG_ROW, "state": " ".join(f"item{index} costs {index * 7}" for index in range(80))}


# Gold answers whose probabilities tie, with a label that is not the first of the tied options
TIED_GOLD = {
    "refund": {"label": False, "probabilities": {"true": 0.5, "false": 0.5}},
    "team": {"label": "zeta", "probabilities": {"alpha": 0.5, "zeta": 0.5}},
    "urgency": {"label": 2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}},
}


def tied_row(index, names=("refund", "team", "urgency")):
    return {**make_row(index, names), "gold": {name: TIED_GOLD[name] for name in names}}


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def decision_config():
    return clef_decision_config()


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
            # A target that is not a mapping, and targets given twice
            (
                "unusable targets",
                {**{k: v for k, v in make_row(0, names=("team",)).items() if k != "gold"}, "targets": {"team": 1}},
            ),
            ("unusable targets", {**make_row(0, names=("team",)), "targets": {"team": {"target_index": 1}}}),
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

    def test_unchanged_order_is_not_rendered_again(self, tokenizer, decision_config, monkeypatch):
        import random

        rows = [
            make_row(0, names=("team",)),
            VARIED_ROW,
            make_row(1),
            {**VARIED_ROW, "questions": {"team": QUESTIONS["team"]}, "gold": {"team": "zeta"}},
        ]
        rows += [make_row(index) for index in range(2, 12)]
        max_length = (
            len(DataCollatorForClef(tokenizer, decision_config).tokenize({**VARIED_ROW, "state": ""})["input_ids"])
            + 30
        )
        collator = DataCollatorForClef(
            tokenizer, decision_config, max_length=max_length, truncation="state", permute_fields=True, seed=7
        )
        features = tokenized(collator, rows)
        rendered = []
        render = collator._render
        monkeypatch.setattr(
            collator, "_render", lambda request: rendered.append(list(request["questions"])) or render(request)
        )
        # The orders the collator's generator draws, consumed as before: one shuffle per record, in batch order
        reference = random.Random("7/0/None")
        orders = []
        for feature in features:
            order = list(range(len(feature["labels"])))
            reference.shuffle(order)
            orders.append(order)
        assert any(len(order) > 1 and order == sorted(order) for order in orders)
        assert any(order != sorted(order) for order in orders)
        expected_renders = []
        for feature, order in zip(features, orders, strict=False):
            result = collator._permute(feature)
            names = list(json.loads(feature["request"])["questions"])
            if order == sorted(order):
                # The cached record, untouched, and nothing rendered
                assert result is feature
                continue
            # Still permuted in the order the unchanged random stream gives, and rendered (then, for a cut record,
            # rendered again with an empty state)
            assert result["labels"] == [feature["labels"][i] for i in order]
            assert len(result["input_ids"]) <= max_length
            expected_renders += [[names[i] for i in order]] * (2 if feature["truncated"] else 1)
        assert rendered == expected_renders

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

    def test_json_string_state_renders_like_its_record(self, tokenizer, decision_config):
        # Datasets store the state as a JSON string (with spaces); the record renders as compact JSON
        state = {"agent": {"autonomy": "high", "tools": ["refund", "escalate"]}, "first_passage": "How do I pay?"}
        record = {**make_row(0), "state": state}
        stored = {**record, "state": json.dumps(state)}
        assert ", " in stored["state"]
        columns = {key: json.dumps(value) for key, value in record.items()}  # every column a JSON string
        as_request = {
            "request": json.dumps({"state": json.dumps(state), "questions": QUESTIONS}),
            "gold": record["gold"],
        }
        collator = DataCollatorForClef(tokenizer, decision_config)
        expected = collator.tokenize(record)
        assert '"first_passage":"How' in tokenizer.decode(expected["input_ids"])
        for row in (stored, columns, as_request):
            assert collator.tokenize(row) == expected

        # Permuted prompts re-render the decoded state
        def permuting(**kwargs):
            return DataCollatorForClef(tokenizer, decision_config, permute_fields=True, seed=3, **kwargs)

        def check_same_permutations(first, second, **kwargs):
            # Two collators with the same seed draw the same orders, so the batches agree token for token
            features = [permuting(**kwargs).tokenize(first), permuting(**kwargs).tokenize(second)]
            assert features[0] == features[1]
            collators = [permuting(**kwargs), permuting(**kwargs)]
            orders = set()
            for _ in range(6):
                batches = [collator([feature] * 3) for collator, feature in zip(collators, features, strict=True)]
                assert all(torch.equal(batches[0][key], batches[1][key]) for key in batches[0])
                orders.update(tuple(question_names(tokenizer, batches[0], row)) for row in range(3))
            assert len(orders) > 1

        check_same_permutations(record, stored)
        # A cut state is cut from the decoded state
        unlimited = DataCollatorForClef(tokenizer, decision_config, max_length=100_000)
        long_state = {"items": [f"item{index} costs {index * 7}" for index in range(80)]}
        long_record = {**record, "state": long_state}
        max_length = len(unlimited.tokenize(long_record)["input_ids"]) - 40
        cutting = DataCollatorForClef(tokenizer, decision_config, max_length=max_length, truncation="state")
        cut = cutting.tokenize(long_record)
        assert cut["truncated"] is True and len(cut["input_ids"]) == max_length
        assert cutting.tokenize({**long_record, "state": json.dumps(long_state)}) == cut
        check_same_permutations(
            long_record,
            {**long_record, "state": json.dumps(long_state)},
            max_length=max_length,
            truncation="state",
        )

    @pytest.mark.parametrize("state", ["ticket 0: refund alpha", '"a JSON string"', "42", "[not JSON", ""])
    def test_state_that_is_not_a_json_object_is_left_alone(self, tokenizer, decision_config, state):
        collator = DataCollatorForClef(tokenizer, decision_config)
        request = collator._request({**make_row(0), "state": state})
        assert request["state"] == state
        if state:
            assert state in tokenizer.decode(collator.tokenize({**make_row(0), "state": state})["input_ids"])

    def test_gold_label_is_carried_next_to_the_target(self, tokenizer, decision_config):
        # Options in template order: refund (true, false), team (alpha, zeta), urgency (0, 1, 2)
        collator = DataCollatorForClef(tokenizer, decision_config)
        tied = {
            **make_row(0),
            "gold": {
                "refund": {"label": False, "probabilities": {"true": 0.5, "false": 0.5}},
                "team": {"label": "zeta", "probabilities": {"alpha": 0.5, "zeta": 0.5}},
                "urgency": {"label": 2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}},
            },
        }
        features = collator.tokenize(tied)
        assert features["labels"] == [[0.5, 0.5], [0.5, 0.5], pytest.approx([0.2, 0.4, 0.4])]
        assert features["hard_labels"] == [1, 1, 2]
        # Without a label (or with one that names no option), the first most probable option
        unlabeled = {**tied, "gold": {**tied["gold"], "urgency": {"probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}}}
        assert collator.tokenize(unlabeled)["hard_labels"] == [1, 1, 1]
        unknown = {**tied, "gold": {**tied["gold"], "team": {"label": "omega", "probabilities": {"zeta": 1.0}}}}
        assert collator.tokenize(unknown)["hard_labels"] == [1, 1, 2]
        targets = {"refund": {"p_true": 0.5}, "team": {"target_index": 1}, "urgency": {"score": 1.5}}
        without_gold = {key: value for key, value in make_row(0).items() if key != "gold"}
        assert collator.tokenize({**without_gold, "targets": targets})["hard_labels"] == [0, 1, 1]
        # A bare label, as `answers`
        answers = {**make_row(0), "answers": {"refund": False, "team": "zeta", "urgency": 2}}
        del answers["gold"]
        assert collator.tokenize(answers)["hard_labels"] == [1, 1, 2]
        batch = collator([features, collator.tokenize(make_row(1, names=("team",)))])
        assert batch["hard_labels"].tolist() == [1, 1, 2, 1]
        # A permuted record keeps the gold label with its question
        permuting = DataCollatorForClef(tokenizer, decision_config, permute_fields=True, seed=1)
        record = permuting.tokenize(tied)
        expected = {"refund": 1, "team": 1, "urgency": 2}
        orders = set()
        for _ in range(8):
            batch = permuting([record])
            names = question_names(tokenizer, batch, 0)
            orders.add(tuple(names))
            assert batch["hard_labels"].tolist() == [expected[name] for name in names]
        assert len(orders) > 1

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
        script_settings = dict(
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
        )
        script_args = ClefScriptArguments(**{**script_settings, **(script_kwargs or {})})
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

    def test_main_says_when_no_record_fits(self, tokenizer):
        # Every record is longer than max_length: the run stops with the count and the reason, not in the sampler
        with pytest.raises(ValueError, match=r"No training record is left.*max_length"):
            self.run_main(tokenizer, "head", script_kwargs={"max_length": 8})

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

    @staticmethod
    def drive(callback, model, epochs):
        """The (weight, sigma) the model's config holds at the start of a step at each of `epochs`."""
        state = TrainerState()
        seen = []
        for epoch in epochs:
            state.epoch = epoch
            callback.on_step_begin(None, state, None, model=model)
            seen.append((model.config.rlcd_weight, model.config.rlcd_sigma))
        return seen

    def test_schedule_values_follow_the_epoch(self, tokenizer, decision_config):
        model = tiny_clef(tokenizer, decision_config)
        callback = RlcdScheduleCallback(0.25, start_epoch=1, sigmas=[0.4, 0.3, 0.2, 0.1])
        epochs = [0.0, 0.5, 0.99, 1.0, 1.5, 2.0, 3.7, 9.0]
        seen = self.drive(callback, model, epochs)
        assert [weight for weight, _ in seen] == [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25]
        # Before start_epoch the sigma is left alone (the config's 0.4); from there one per epoch, the last held
        assert [sigma for _, sigma in seen[3:]] == [0.4, 0.4, 0.3, 0.2, 0.1]
        assert seen[0][1] == ClefConfig().rlcd_sigma
        # No sigmas: the config's sigma is never touched; an epoch of None counts as 0
        model = tiny_clef(tokenizer, decision_config)
        model.config.rlcd_sigma = 0.37
        callback = RlcdScheduleCallback(0.5, start_epoch=0.5)
        assert self.drive(callback, model, [None, 0.4, 0.5, 2.0]) == [
            (0.0, 0.37),
            (0.0, 0.37),
            (0.5, 0.37),
            (0.5, 0.37),
        ]
        # Wrapped models (DistributedDataParallel) are unwrapped
        model = tiny_clef(tokenizer, decision_config)
        wrapper = type("Wrapper", (), {"module": model})()
        RlcdScheduleCallback(0.25).on_step_begin(None, TrainerState(epoch=0.0), None, model=wrapper)
        assert model.config.rlcd_weight == 0.25

    def test_schedule_is_stateless_across_resume(self, tokenizer, decision_config):
        sigmas = [0.4, 0.3, 0.2, 0.1]
        whole = tiny_clef(tokenizer, decision_config)
        self.drive(RlcdScheduleCallback(0.25, start_epoch=1, sigmas=sigmas), whole, [0.0, 0.5, 1.0, 1.5, 2.0, 2.3])
        resumed = tiny_clef(tokenizer, decision_config)
        self.drive(RlcdScheduleCallback(0.25, start_epoch=1, sigmas=sigmas), resumed, [2.3])
        assert (resumed.config.rlcd_weight, resumed.config.rlcd_sigma) == (0.25, 0.3)
        assert (resumed.config.rlcd_weight, resumed.config.rlcd_sigma) == (
            whole.config.rlcd_weight,
            whole.config.rlcd_sigma,
        )

    def test_final_values_left_on_config(self, tokenizer, decision_config):
        model = tiny_clef(tokenizer, decision_config)
        callback = RlcdScheduleCallback(0.25, start_epoch=1, sigmas=[0.4, 0.3, 0.2, 0.1])
        state = TrainerState()
        callback.on_train_begin(None, state, None, model=model)
        self.drive(callback, model, [0.5, 1.5, 3.5])
        assert model.config.rlcd_sigma == 0.2
        callback.on_train_end(None, state, None, model=model)
        # The values in effect at the last step: the weight of the phase, and the last sigma that applied
        assert (model.config.rlcd_weight, model.config.rlcd_sigma) == (0.25, 0.2)
        # Training that stopped before the phase started (early stopping) never trained with RLCD, so the config does
        # not claim it did
        model = tiny_clef(tokenizer, decision_config)
        callback = RlcdScheduleCallback(0.25, start_epoch=3, sigmas=[0.4])
        callback.on_train_begin(None, state, None, model=model)
        self.drive(callback, model, [0.0, 1.0])
        callback.on_train_end(None, state, None, model=model)
        assert (model.config.rlcd_weight, model.config.rlcd_sigma) == (0.0, ClefConfig().rlcd_sigma)

    def test_invalid_arguments(self):
        for kwargs in (
            {"weight": -0.1},
            {"weight": 0.25, "start_epoch": -1},
            {"weight": 0.25, "sigmas": []},
            {"weight": 0.25, "sigmas": [0.4, 0.0]},
            {"weight": 0.25, "sigmas": [-0.1]},
        ):
            with pytest.raises(ValueError):
                RlcdScheduleCallback(**kwargs)
        assert RlcdScheduleCallback(0.0).weight == 0.0

    def test_model_without_rlcd_raises(self):
        model = torch.nn.Linear(2, 2)
        model.config = type("Config", (), {})()
        callback = RlcdScheduleCallback(0.25)
        with pytest.raises(ValueError, match="rlcd_weight"):
            callback.on_train_begin(None, TrainerState(), None, model=model)

    def record_training_steps(self, monkeypatch):
        """Record (epoch, rlcd_weight, rlcd_sigma) of the config each training step starts with."""
        seen = []
        original = DecisionTrainer.training_step

        def training_step(trainer, model, *args, **kwargs):
            seen.append((trainer.state.epoch, model.config.rlcd_weight, model.config.rlcd_sigma))
            return original(trainer, model, *args, **kwargs)

        monkeypatch.setattr(DecisionTrainer, "training_step", training_step)
        return seen

    def test_one_run_ce_then_rlcd(self, tokenizer, monkeypatch):
        seen = self.record_training_steps(monkeypatch)
        script_kwargs = {"rlcd_weight": 0.25, "rlcd_samples": 8, "rlcd_start_epoch": 1, "rlcd_sigmas": [0.4, 0.1]}
        training_kwargs = {"max_steps": -1, "num_train_epochs": 2, "logging_steps": 1}
        trainer, initial, _ = self.run_main(
            tokenizer, "full", training_kwargs=training_kwargs, script_kwargs=script_kwargs
        )
        # The model is built without RLCD: the schedule turns it on
        assert initial.config.rlcd_weight == 0.0
        assert trainer.state.global_step == len(seen) and len(seen) > 2
        first_epoch = [entry for entry in seen if entry[0] < 1.0]
        second_epoch = [entry for entry in seen if entry[0] >= 1.0]
        assert first_epoch and second_epoch
        assert all(weight == 0.0 for _, weight, _ in first_epoch)
        assert all((weight, sigma) == (0.25, 0.4) for _, weight, sigma in second_epoch)
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        assert losses and all(np.isfinite(loss) for loss in losses)
        # The supervised evaluation loss does not depend on the term or on any randomness
        first = trainer.evaluate()["eval_loss"]
        assert trainer.evaluate()["eval_loss"] == first
        # The saved config records the recipe: the weight, and the last sigma that applied
        self.check_saved_model(trainer, tokenizer, rlcd_weight=0.25, rlcd_samples=8, rlcd_sigma=0.4)

    def test_two_runs_ce_then_rlcd(self, tokenizer, monkeypatch):
        # Run 1: plain cross-entropy
        first, _, _ = self.run_main(tokenizer, "full", training_kwargs={"eval_strategy": "no"})
        assert first.model.config.rlcd_weight == 0.0
        first_loss = first.evaluate()["eval_loss"]
        # The dataset folder is the run's temporary folder, so the checkpoint moves out of it
        with tempfile.TemporaryDirectory() as keep:
            checkpoint = f"{keep}/checkpoint"
            shutil.move(f"{self.tmp_dir}/out", checkpoint)
            self.second_run_from(tokenizer, monkeypatch, checkpoint, first_loss)

    def second_run_from(self, tokenizer, monkeypatch, checkpoint, first_loss):
        # Run 2: the example again from the saved checkpoint, RLCD on from the first step, no resume
        starting = []
        original = DecisionTrainer.train

        def train(trainer, *args, **kwargs):
            assert kwargs.get("resume_from_checkpoint") is None
            starting.append(trainer.evaluate()["eval_loss"])
            return original(trainer, *args, **kwargs)

        monkeypatch.setattr(DecisionTrainer, "train", train)
        second, initial, _ = self.run_main(
            tokenizer,
            "full",
            training_kwargs={"output_dir": f"{self.tmp_dir}/out2", "logging_steps": 1},
            script_kwargs={"rlcd_weight": 0.25, "rlcd_samples": 8},
            checkpoint=checkpoint,
        )
        # Strict loading of the checkpoint, and the weights it saved
        assert initial.config.rlcd_weight == 0.25
        assert starting == [first_loss]
        # A fresh run: it trained from step 0 with a fresh optimizer, not resumed
        assert second.state.global_step == 5 and second.args.resume_from_checkpoint is None
        losses = [entry["loss"] for entry in second.state.log_history if "loss" in entry]
        assert losses and all(np.isfinite(loss) for loss in losses)
        assert second.model.config.rlcd_weight == 0.25
        assert AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/out2").config.rlcd_weight == 0.25
        assert AutoModelForDecision.from_pretrained(checkpoint).config.rlcd_weight == 0.0

    @pytest.mark.parametrize("weight", [None, 0.0])
    def test_schedule_requires_positive_weight(self, tokenizer, weight):
        for key, value in (("rlcd_start_epoch", 1.0), ("rlcd_sigmas", [0.4, 0.1])):
            script_kwargs = {key: value, **({} if weight is None else {"rlcd_weight": weight})}
            with pytest.raises(ValueError, match="rlcd_weight"):
                self.run_main(tokenizer, "full", script_kwargs=script_kwargs)

    def test_sample_rows_draws_a_seeded_random_subset(self):
        dataset = Dataset.from_list([{"index": index} for index in range(100)])
        first, again, other = (sample_rows(dataset, 10, seed)["index"] for seed in (3407, 3407, 1))
        assert len(first) == len(set(first)) == 10
        assert first == again
        assert first != other and set(first) != set(other)
        # Not the first rows of the split
        assert set(first) != set(range(10))
        # Asking for as many rows as there are, or more, or for no limit, keeps the split as it is
        for count in (100, 500, None):
            assert sample_rows(dataset, count, 1)["index"] == list(range(100))
        assert len(sample_rows(dataset, 0, 1)) == 0

    def test_main_samples_the_limited_splits(self, tokenizer):
        from datasets import load_dataset

        script_kwargs = {"max_train_samples": 5, "max_eval_samples": 2, "eval_sample_seed": 5}
        trainer, _, _ = self.run_main(tokenizer, "head", {"seed": 11}, script_kwargs=script_kwargs)
        # The splits main trains and evaluates on are the seeded samples of its splits, tokenized: the holdout and
        # the evaluation sample are drawn with eval_sample_seed, the training sample with the run's seed
        records = load_dataset("json", data_files=f"{self.tmp_dir}/train.jsonl")["train"]
        split = records.train_test_split(test_size=0.25, seed=5)
        collator = DataCollatorForClef(tokenizer, trainer.model.decision_config, max_length=400, truncation="skip")
        for dataset, name, count, seed in (
            (trainer.train_dataset, "train", 5, 11),
            (trainer.eval_dataset, "test", 2, 5),
        ):
            sampled = sample_rows(split[name], count, seed)
            expected = [collator.tokenize(row) for row in sampled]
            assert dataset["input_ids"] == [row["input_ids"] for row in expected if not row["skip"]]
            assert sampled["state"] != split[name].select(range(count))["state"]

    def test_runs_with_different_seeds_evaluate_on_the_same_records(self, tokenizer):
        script_kwargs = {"max_train_samples": 5, "max_eval_samples": 2}
        import shutil

        first, _, _ = self.run_main(tokenizer, "head", {"seed": 11}, script_kwargs=script_kwargs)
        shutil.rmtree(f"{self.tmp_dir}/out")  # its metrics files would join the dataset of the next run
        second, _, _ = self.run_main(tokenizer, "head", {"seed": 12}, script_kwargs=script_kwargs)
        assert len(first.eval_dataset) > 0
        assert first.eval_dataset["input_ids"] == second.eval_dataset["input_ids"]
        assert first.train_dataset["input_ids"] != second.train_dataset["input_ids"]

    def test_checkpoint_head_settings_cannot_change(self, tokenizer, decision_config):
        tiny_clef(tokenizer, decision_config).save_pretrained(f"{self.tmp_dir}/clef")
        model_args = ClefModelArguments(model_name_or_path=f"{self.tmp_dir}/clef", dtype="float32")
        script_args = ClefScriptArguments(head_type="cosine")
        with pytest.raises(ValueError, match="cannot change"):
            build_model(script_args, model_args, {"dtype": "float32"})
        # The settings it already has are fine
        model = build_model(ClefScriptArguments(head_type="joint"), model_args, {"dtype": "float32"})
        assert model.config.head_type == "joint"
        # So are its head sizes: they are not ignored, and the ones it has are fine
        with pytest.raises(ValueError, match="hidden_size"):
            build_model(ClefScriptArguments(head_hidden_size=64), model_args, {"dtype": "float32"})
        assert build_model(ClefScriptArguments(head_hidden_size=16), model_args, {"dtype": "float32"})
        # The cosine head's settings do not apply to the joint head
        with pytest.raises(ValueError, match="cosine_width"):
            build_model(ClefScriptArguments(cosine_width=16), model_args, {"dtype": "float32"})

    def test_head_settings_that_would_be_ignored_raise(self, tokenizer):
        AutoModelForCausalLM.from_pretrained(MODEL_ID).save_pretrained(f"{self.tmp_dir}/backbone")
        for settings in ({"cosine_width": 8}, {"head_type": "joint", "cosine_initial_scale": 5.0}):
            with pytest.raises(ValueError, match="head_type: cosine"):
                self.build_from(f"{self.tmp_dir}/backbone", **settings)
        assert self.build_from(f"{self.tmp_dir}/backbone", head_type="cosine", cosine_width=8).config.cosine_width == 8
        # rlcd_sigmas sets every epoch's noise scale, so rlcd_sigma would be ignored
        with pytest.raises(ValueError, match="rlcd_sigma"):
            self.build_from(f"{self.tmp_dir}/backbone", rlcd_weight=0.25, rlcd_sigma=0.3, rlcd_sigmas=[0.4, 0.1])

    def test_rlcd_on_a_new_head_without_a_schedule_warns(self, tokenizer, decision_config):
        AutoModelForCausalLM.from_pretrained(MODEL_ID).save_pretrained(f"{self.tmp_dir}/backbone")
        with pytest.warns(UserWarning, match="after cross-entropy"):
            self.build_from(f"{self.tmp_dir}/backbone", rlcd_weight=0.25)
        # Not with a schedule, nor for a checkpoint (the second phase of the recipe)
        tiny_clef(tokenizer, decision_config).save_pretrained(f"{self.tmp_dir}/clef")
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*after cross-entropy.*")
            self.build_from(f"{self.tmp_dir}/backbone", rlcd_weight=0.25, rlcd_start_epoch=1)
            self.build_from(f"{self.tmp_dir}/clef", rlcd_weight=0.25)

    def test_rlcd_start_at_or_after_the_end_raises(self, tokenizer):
        # 5 steps of 2 of the 9 training records: training ends at epoch 2
        for start in (2, 3):
            with pytest.raises(ValueError, match="never run"):
                self.run_main(tokenizer, "full", script_kwargs={"rlcd_weight": 0.25, "rlcd_start_epoch": start})
        trainer, _, _ = self.run_main(tokenizer, "full", script_kwargs={"rlcd_weight": 0.25, "rlcd_start_epoch": 1})
        assert trainer.model.config.rlcd_weight == 0.25

    def test_max_length_beyond_the_backbone_positions_raises(self, tokenizer):
        from transformers import BertModel

        tokenizer = encoder_tokenizer(tokenizer)
        sizes = {"vocab_size": len(tokenizer) + 2, "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2}
        set_seed(0)
        backbone = BertModel(BertConfig(**sizes, num_attention_heads=2, max_position_embeddings=128))
        with pytest.raises(ValueError, match="at most 128 positions"):
            self.run_main(tokenizer, "head", backbone=backbone)  # max_length 400
        assert not Path(self.tmp_dir, "out").exists()

    def test_missing_template_follows_the_prompt_style(self, tokenizer):
        from transformers.decision import DecisionConfig as TransformersDecisionConfig

        tokenizer = encoder_tokenizer(tokenizer)
        model = tiny_clef(tokenizer, None, "bert")
        model.decision_config = TransformersDecisionConfig()
        assert model.decision_config.chat_template is None
        model.save_pretrained(f"{self.tmp_dir}/clef")
        built = self.build_from(f"{self.tmp_dir}/clef")
        assert built.config.prompt_style == "plain"
        assert built.decision_config.chat_template == clef_decision_config("plain").chat_template
        assert built.decision_config.chat_template != clef_decision_config().chat_template

    def test_default_configuration_keeps_its_best_checkpoint(self):
        from trl import TrlParser

        parser = TrlParser((ClefScriptArguments, DecisionConfig, ClefModelArguments))
        path = Path(__file__).resolve().parents[2] / "examples/decisions/clef_typed_decisions.yaml"
        script_args, args, _ = parser.parse_args_and_config(["--config", str(path)])
        # LocalLLaMA/typed-decisions has 1,200 rows
        train_rows = 1200 - math.ceil(1200 * script_args.validation_fraction)
        steps_per_epoch = math.ceil(train_rows / args.per_device_train_batch_size / args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * args.num_train_epochs
        # Every evaluation that improves on the metric is saved, so the best checkpoint is on disk to restore
        assert args.load_best_model_at_end and args.save_strategy == "best"
        assert args.metric_for_best_model == "eval_selection_score" and args.greater_is_better
        # Evaluations within each epoch, after the warm-up, and several of them
        assert args.eval_strategy == "steps" and args.eval_steps < steps_per_epoch
        assert args.warmup_steps < args.eval_steps and total_steps // args.eval_steps >= 6

    def test_default_configuration_trains_the_cosine_head(self):
        from trl import TrlParser

        parser = TrlParser((ClefScriptArguments, DecisionConfig, ClefModelArguments))
        path = Path(__file__).resolve().parents[2] / "examples/decisions/clef_typed_decisions.yaml"
        script_args, _, _ = parser.parse_args_and_config(["--config", str(path)])
        # The shipped configuration picks the small cosine head, with its default width and scale
        assert script_args.head_type == "cosine"
        assert script_args.cosine_width is None and script_args.cosine_initial_scale is None
        # Without a configuration, the script leaves the model's own default (the joint head) alone
        assert ClefScriptArguments().head_type is None

    @pytest.mark.parametrize("attention", ["eager", "sdpa"])
    def test_attention_implementation_reaches_the_model(self, tokenizer, decision_config, attention):
        # A new head on a backbone, and a Clef checkpoint, both get the attention that was asked for
        AutoModelForCausalLM.from_pretrained(MODEL_ID).save_pretrained(f"{self.tmp_dir}/backbone")
        tiny_clef(tokenizer, decision_config).save_pretrained(f"{self.tmp_dir}/clef")
        new_head = ClefScriptArguments(
            head_hidden_size=16,
            head_intermediate_size=32,
            head_num_layers=1,
            head_num_evidence_layers=1,
            head_num_attention_heads=2,
        )
        # A checkpoint keeps its own head sizes
        for folder, script_args in (("backbone", new_head), ("clef", ClefScriptArguments())):
            model_args = ClefModelArguments(model_name_or_path=f"{self.tmp_dir}/{folder}", dtype="float32")
            kwargs = {"dtype": "float32", "revision": None, "attn_implementation": attention}
            model = build_model(script_args, model_args, kwargs)
            assert model.model.config._attn_implementation == attention, folder
            # The backbone's only: the head keeps its default, whatever kernel the backbone uses
            assert model.config._attn_implementation == "sdpa", folder

    def test_attention_implementation_reaches_main(self, tokenizer):
        trainer, _, _ = self.run_main(tokenizer, "head", attn_implementation="eager")
        # Evaluation batches with its own, non-permuting collator
        assert trainer.data_collator.permute_fields is True
        assert trainer.eval_data_collator.permute_fields is False
        assert trainer.model.model.config._attn_implementation == "eager"
        assert trainer.model.config._attn_implementation == "sdpa"

    @pytest.mark.parametrize("key, value", [("low_cpu_mem_usage", True), ("device_map", "cpu"), ("cache_dir", "x")])
    def test_model_init_kwargs_reach_the_backbone_of_a_new_head(self, key, value, monkeypatch):
        AutoModelForCausalLM.from_pretrained(MODEL_ID).save_pretrained(f"{self.tmp_dir}/backbone")
        calls = []
        original = AutoModelForCausalLM.from_pretrained

        def from_pretrained(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
        model_args = ClefModelArguments(model_name_or_path=f"{self.tmp_dir}/backbone", dtype="float32")
        model = build_model(ClefScriptArguments(), model_args, {"dtype": "float32", "revision": None, key: value})
        assert len(calls) == 1 and calls[0][key] == value
        assert isinstance(model, ClefForDecision)
        # A setting of the backbone's config given as a model_init_kwargs entry is kept by the built model
        calls.clear()
        model = build_model(
            ClefScriptArguments(), model_args, {"dtype": "float32", "revision": None, "use_cache": False}
        )
        assert calls[0]["use_cache"] is False and model.model.config.use_cache is False

    @pytest.mark.parametrize("head_type", ["joint", "cosine"])
    @pytest.mark.parametrize("backbone_kind", ["causal", "encoder"])
    def test_new_head_is_built_around_the_loaded_backbone(self, tokenizer, backbone_kind, head_type, monkeypatch):
        from examples.decisions import clef_typed_decisions

        folder = f"{self.tmp_dir}/backbone"
        if backbone_kind == "causal":
            AutoModelForCausalLM.from_pretrained(MODEL_ID).save_pretrained(folder)
        else:
            tokenizer = encoder_tokenizer(tokenizer)
            tiny_encoder(tokenizer, "bert").save_pretrained(folder)
        loaded = []

        load_decision_backbone = clef_typed_decisions.load_decision_backbone

        def spy(*args, **kwargs):
            loaded.append(load_decision_backbone(*args, **kwargs))
            return loaded[-1]

        monkeypatch.setattr(clef_typed_decisions, "load_decision_backbone", spy)
        script_args = ClefScriptArguments(
            head_type=head_type,
            head_hidden_size=16,
            head_intermediate_size=32,
            head_num_layers=1,
            head_num_evidence_layers=1,
            head_num_attention_heads=2,
        )
        model_args = ClefModelArguments(model_name_or_path=folder, dtype="bfloat16")
        kwargs = {"dtype": "bfloat16", "revision": None, "attn_implementation": "eager"}
        model = build_model(script_args, model_args, kwargs)
        ((backbone,),) = [(entry.model,) for entry in loaded]
        embeddings = loaded[0].embeddings
        # The loaded backbone itself, with the weights, dtype and attention it was loaded with
        assert model.model is backbone
        assert model.model.config._attn_implementation == "eager" and model.config._attn_implementation == "sdpa"
        expected = AutoModel.from_pretrained(folder, dtype=torch.bfloat16).state_dict()
        for name, value in model.model.state_dict().items():
            assert torch.equal(value, expected[name]), name
        # The head is float32, but for the joint head's lm_head, which starts from the embeddings, in the backbone's
        # dtype
        assert {parameter.dtype for parameter in model.model.parameters()} == {torch.bfloat16}
        head = {name: p.dtype for name, p in model.named_parameters() if not name.startswith(("model.", "lm_head."))}
        assert head and set(head.values()) == {torch.float32}
        assert hasattr(model, "lm_head") == (head_type == "joint")
        if head_type == "joint":
            assert model.lm_head.weight.dtype == torch.bfloat16 and torch.equal(model.lm_head.weight, embeddings)
        # Saved and reloaded, it gives the same logits. The reloaded joint head is in the saved dtype (the cosine head
        # stays in float32), so up to bfloat16 rounding
        model.save_pretrained(f"{self.tmp_dir}/clef")
        reloaded = AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/clef")

        def layout(model):
            return [
                (name, p.shape, p.dtype if head_type == "cosine" else None) for name, p in model.named_parameters()
            ]

        assert layout(reloaded) == layout(model)
        collator = DataCollatorForClef(tokenizer, model.decision_config)
        batch = collator([collator.tokenize(make_row(index)) for index in range(2)])
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected_logits = model.eval()(**batch).logits.float()
            logits = reloaded.eval()(**batch).logits.float()
        torch.testing.assert_close(logits, expected_logits, atol=2e-2, rtol=2e-2)

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

    # The groups of lora mode are checked in TestHeadLearningRate: main merges the adapters away after training
    @pytest.mark.parametrize("mode", ["full", "head"])
    def test_main_trains_the_head_at_head_learning_rate(self, tokenizer, mode):
        trainer, _, _ = self.run_main(tokenizer, mode, script_kwargs={"head_learning_rate": 3e-3})
        assert trainer.state.global_step == 5
        # The optimizer the trainer built is the example's: the backbone at learning_rate (1e-3), the head at 3e-3
        backbone = {id(parameter) for parameter in trainer.model.base_model.parameters()}
        initial_lr = {
            id(parameter): group["initial_lr"]
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        trainable = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]
        assert len(initial_lr) == len(trainable)
        assert {initial_lr[id(parameter)] for parameter in trainable if id(parameter) not in backbone} == {3e-3}
        in_backbone = {initial_lr[id(parameter)] for parameter in trainable if id(parameter) in backbone}
        assert in_backbone == (set() if mode == "head" else {1e-3})

    def test_main_without_head_learning_rate_keeps_the_default_optimizer(self, tokenizer):
        trainer, _, _ = self.run_main(tokenizer, "full")
        assert trainer.optimizer_cls_and_kwargs is None
        assert {group["initial_lr"] for group in trainer.optimizer.param_groups} == {1e-3}

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

    def test_backbone_that_cannot_be_loaded_stops_at_start_up(self, tokenizer, monkeypatch):
        # The loader's refusals (tests/experimental/test_decision_backbone.py) stop the run before training
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        keys = list(backbone.state_dict())
        half = {**prefixed(backbone, "backbone.", keys[::2]), **PAIR_HEAD}
        whole = {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        folders = {
            "missing": reranker_package(f"{self.tmp_dir}/half", backbone, tokenizer, half),
            "attention_windows": reranker_package(
                f"{self.tmp_dir}/windows", backbone, tokenizer, whole, attention_windows=[8, 4, None]
            ),
        }
        self.no_training(monkeypatch)
        for match, folder in folders.items():
            with pytest.raises(ValueError, match=match):
                self.run_main(tokenizer, "full", checkpoint=folder)
        assert not Path(self.tmp_dir, "out").exists()

    @pytest.mark.parametrize("head_type", ["joint", "cosine"])
    def test_main_trains_on_ultrabert_reranker_package(self, tokenizer, head_type, capsys):
        tokenizer = ultrabert_tokenizer(tokenizer)
        backbone = tiny_ultrabert(tokenizer)
        folder = reranker_package(
            f"{self.tmp_dir}/reranker", backbone, tokenizer, {**prefixed(backbone, "backbone."), **PAIR_HEAD}
        )
        script_kwargs = {"head_type": head_type, **({"cosine_width": 8} if head_type == "cosine" else {})}
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

    def test_code_of_a_native_model_type_is_never_executed(self, tokenizer, capsys):
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
        # Its model type is one of Transformers': trust_remote_code still loads it with Transformers' own classes
        capsys.readouterr()
        shutil.rmtree(f"{self.tmp_dir}/out")  # its files would be read as rows of the dataset folder
        trainer, _, _ = self.run_main(tokenizer, "full", checkpoint=folder, trust_remote_code=True)
        assert trainer.state.global_step == 5
        assert 'backbone weights: prefix "backbone."' in capsys.readouterr().out
        assert not sentinel.exists()
        # The flag in model_init_kwargs must agree with the setting
        with pytest.raises(ValueError, match="model_init_kwargs sets trust_remote_code"):
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
        rows = [(tied_row if index % 2 else make_row)(index, names[index % 5]) for index in range(11)]
        val = Dataset.from_list([collator.tokenize(row) for row in rows])

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
        # The labels are the target distributions and the gold labels, packed alike
        (packed_targets, packed_gold), (row_targets, row_gold) = packed.label_ids, rows.label_ids
        np.testing.assert_array_equal(packed_targets[real], row_targets)
        np.testing.assert_array_equal(packed_gold[real], row_gold)
        assert (packed_targets[~real] == -100).all() and (packed_gold[~real] == -100).all()
        assert row_gold.tolist() == [label for row in val for label in row["hard_labels"]]
        # Accuracy is counted against the gold label, which differs from the first most probable option on ties
        assert (row_gold != np.where(row_targets == -100, -np.inf, row_targets).argmax(-1)).sum() >= 5
        correct = rows.predictions.argmax(-1) == row_gold
        assert packed_metrics["eval_accuracy"] == pytest.approx(correct.mean(), abs=1e-9)


class TestEvaluationBatches(TrlTestCase):
    """Evaluation batches are never permuted, and records are never dispatched or split across processes."""

    def test_eval_data_collator_never_permutes(self, tokenizer, decision_config):
        permuting = DataCollatorForClef(tokenizer, decision_config, permute_fields=True, seed=1)
        evaluating = DataCollatorForClef(tokenizer, decision_config)
        rows = [make_row(index) for index in range(8)]
        # Even records that carry their request (tokenized by the permuting collator) are evaluated as they are
        carried = Dataset.from_list(tokenized(permuting, rows))
        plain = Dataset.from_list(tokenized(evaluating, rows))
        assert "request" in carried.column_names
        trainer, _, _ = make_trainer(
            self.tmp_dir, tokenizer, decision_config, compute_metrics=compute_calibration_metrics
        )
        trainer.data_collator = permuting
        trainer.eval_data_collator = evaluating
        reference = trainer.predict(plain)
        for _ in range(3):
            np.testing.assert_array_equal(trainer.predict(carried).predictions, reference.predictions)
        # The training collator still permutes
        batches = [trainer.data_collator([carried[i] for i in range(8)])["input_ids"] for _ in range(4)]
        assert any(not torch.equal(batches[0], batch) for batch in batches[1:])
        assert trainer.get_train_dataloader().collate_fn is not evaluating

    def test_dispatched_or_split_batches_raise(self, tokenizer, decision_config):
        from accelerate.data_loader import BatchSamplerShard, DataLoaderDispatcher
        from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

        from trl.experimental.decision.decision_trainer import splits_batches

        data = list(range(8))
        assert splits_batches(DataLoaderDispatcher(data, batch_size=2), 2)
        assert not splits_batches(DataLoaderDispatcher(data, batch_size=2), 1)
        assert not splits_batches(DataLoader(data, batch_size=2), 2)
        sampler = BatchSamplerShard(BatchSampler(SequentialSampler(data), 4, False), 2, 0, split_batches=True)
        assert splits_batches(DataLoader(data, batch_sampler=sampler), 2)
        trainer, _, _ = make_trainer(self.tmp_dir, tokenizer, decision_config)
        trainer._splits_batches = True
        with pytest.raises(ValueError, match="dispatched or split"):
            trainer.train()
        # The example refuses the settings outright
        for key in ("dispatch_batches", "split_batches"):
            args = DecisionConfig(output_dir=self.tmp_dir, use_cpu=True, report_to="none")
            setattr(args.accelerator_config, key, True)
            with pytest.raises(ValueError, match=key):
                from examples.decisions.clef_typed_decisions import build_trainer

                build_trainer(ClefScriptArguments(), args, ClefModelArguments())


def optimizer_settings(optimizer):
    """The learning rate and weight decay of each parameter of an optimizer, by parameter id."""
    lr = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    wd = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
    return lr, wd


class TestHeadLearningRate(TrlTestCase):
    """The parameter groups the example gives the trainer with `head_learning_rate`."""

    @staticmethod
    def model_for(mode, tokenizer, decision_config, head_type="joint"):
        """The tiny Clef with the parameters each `mode` of the example trains."""
        if mode == "lora":
            return lora_clef(tokenizer, decision_config)
        model = tiny_clef(tokenizer, decision_config, head_type=head_type)
        if mode == "head":
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)
        return model

    def optimizer(self, tokenizer, decision_config, model, **kwargs):
        trainer, _, _ = make_trainer(self.tmp_dir, tokenizer, decision_config, model=model, **kwargs)
        set_head_learning_rate(trainer, 1e-3)
        return trainer, trainer.create_optimizer()

    @pytest.mark.parametrize("mode", ["full", "head", pytest.param("lora", marks=require_peft)])
    def test_rates_by_mode(self, tokenizer, decision_config, mode):
        model = self.model_for(mode, tokenizer, decision_config)
        _, optimizer = self.optimizer(tokenizer, decision_config, model, learning_rate=1e-4, weight_decay=0.05)
        lr, _ = optimizer_settings(optimizer)
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        # Every trainable parameter once, and only those
        assert len(lr) == len(trainable) == sum(len(group["params"]) for group in optimizer.param_groups)
        backbone = {id(p) for p in model.base_model.parameters()}
        head = [name for name, p in trainable.items() if id(p) not in backbone]
        assert head and all(lr[id(trainable[name])] == 1e-3 for name in head)
        in_backbone = [name for name, p in trainable.items() if id(p) in backbone]
        assert all(lr[id(trainable[name])] == 1e-4 for name in in_backbone)
        if mode == "full":
            assert in_backbone
        elif mode == "head":
            assert not in_backbone
        else:
            # The adapters inside the backbone are its only trainable parameters, at learning_rate
            assert in_backbone and all("lora_" in name for name in in_backbone)
            assert {name.split(".")[0] for name in head} == {"embeddings", "encoder", "scorer"}

    @pytest.mark.parametrize("head_type", ["joint", "cosine"])
    def test_zero_dimensional_parameters_are_not_decayed(self, tokenizer, decision_config, head_type):
        model = self.model_for("full", tokenizer, decision_config, head_type=head_type)
        trainer, optimizer = self.optimizer(tokenizer, decision_config, model, learning_rate=1e-4, weight_decay=0.05)
        _, wd = optimizer_settings(optimizer)
        scalars = [p for p in model.parameters() if p.requires_grad and p.dim() == 0]
        assert scalars and all(wd[id(p)] == 0.0 for p in scalars)
        decayed = set(trainer.get_decay_parameter_names(model))
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() > 0:
                assert wd[id(p)] == (0.05 if name in decayed else 0.0), name
        assert any(value == 0.05 for value in wd.values())
        # Backbone and head, each decayed and not
        assert len(optimizer.param_groups) == 4

    @pytest.mark.parametrize("kwarg", ["params", "model", "optimizer_dict"])
    def test_optimizers_that_set_their_own_groups_are_refused(self, tokenizer, decision_config, kwarg, monkeypatch):
        model = self.model_for("full", tokenizer, decision_config)
        trainer, _, _ = make_trainer(self.tmp_dir, tokenizer, decision_config, model=model)
        # As GaLore (params), LOMO (model) and the layer-wise optimizers (optimizer_dict) do
        monkeypatch.setattr(
            trainer, "get_optimizer_cls_and_kwargs", lambda args, model=None: (torch.optim.AdamW, {kwarg: None})
        )
        with pytest.raises(ValueError, match="head_learning_rate is not supported"):
            set_head_learning_rate(trainer, 1e-3)
        assert trainer.optimizer_cls_and_kwargs is None


class TestClefBenchmark(TrlTestCase):
    """The benchmark runs the example's own training path and reports the numbers of the reference benchmark."""

    def test_synthetic_benchmark_reports_all_numbers(self):
        from examples.decisions.benchmark_clef import run_benchmark

        r = run_benchmark(steps=8, warmup_steps_excluded=3, synthetic_records=48)
        for key in ("build_seconds", "first_step_seconds", "steady_records_per_second", "eval_seconds"):
            assert math.isfinite(r[key]) and r[key] > 0
        assert r["recipe"]["synthetic"] is True
        # The fields of the reference's summary.json, under the same names
        for key in (
            "startup_seconds",
            "trainer_wall_seconds",
            "steady_step_p50",
            "steady_step_p95",
            "window_records_per_second",
            "steady_data_p50",
            "steady_records_per_second_with_data",
        ):
            assert math.isfinite(r[key]) and r[key] > 0, key
        # Called as a function, startup counts from the call: it covers the build, before training
        assert r["recipe"]["startup_origin"] == "run_benchmark call"
        assert r["build_seconds"] <= r["startup_seconds"] < r["build_seconds"] + r["trainer_wall_seconds"] + 60
        assert r["global_steps"] == 8 and r["window_steps"] == 5
        assert r["recipe"]["warmup_steps_excluded"] == 3
        assert r["torch"] == torch.__version__
        if not torch.cuda.is_available():
            assert r["gpu"] is None
            assert r["peak_memory_bytes"] is None
            assert r["peak_allocated_gib"] is None and r["peak_reserved_gib"] is None
        # The steps are written out, so the percentiles and rates can be recomputed
        steps = r["steps"]
        assert [step["step"] for step in steps] == list(range(1, 9))
        assert all(step["seconds"] > 0 and step["data_seconds"] >= 0 for step in steps)
        assert r["first_step_seconds"] == pytest.approx(steps[0]["seconds"] + steps[0]["data_seconds"])
        steady = [step for step in steps if step["step"] > 3]
        seconds = [step["seconds"] for step in steady]
        assert r["steady_step_p50"] == pytest.approx(float(np.percentile(seconds, 50)))
        assert r["steady_step_p95"] == pytest.approx(float(np.percentile(seconds, 95)))
        assert r["steady_data_p50"] == pytest.approx(float(np.percentile([s["data_seconds"] for s in steady], 50)))
        records_per_step = r["recipe"]["records_per_step"]
        assert records_per_step == 8
        assert r["window_records_per_second"] == pytest.approx(records_per_step * len(seconds) / sum(seconds))
        with_data = [step["seconds"] + step["data_seconds"] for step in steady]
        assert r["steady_records_per_second_with_data"] == pytest.approx(
            records_per_step * len(with_data) / sum(with_data)
        )
        # Under 320 steps, the window is every steady step
        assert r["steady_records_per_second"] == pytest.approx(r["window_records_per_second"])
        recipe = r["recipe"]
        assert recipe["backbone"] == "Qwen2Model"
        assert recipe["parameter_dtypes"] == ["float32"]
        assert isinstance(recipe["attn_implementation"], str)
        assert recipe["activation_checkpointing"] is False
        assert recipe["head_type"] == "joint"
        assert recipe["use_fla_kernels"] is None
        assert recipe["script"]["truncation"] == "state" and recipe["script"]["permute_fields"] is True
        assert recipe["training"]["max_steps"] == 8
        assert recipe["build_report"]["train"]["rows"] + recipe["build_report"]["eval"]["rows"] == 48
        json.dumps(r)

    def test_parameter_counts_separate_head_backbone_and_lexical_prior(self, tokenizer, decision_config):
        from types import SimpleNamespace

        from examples.decisions.benchmark_clef import describe_recipe

        def counts(model):
            run = SimpleNamespace(trainer=SimpleNamespace(model=model), reports={})
            args = DecisionConfig(output_dir=self.tmp_dir, use_cpu=True, report_to="none")
            return describe_recipe(run, ClefScriptArguments(), args, ClefModelArguments(), synthetic=False)

        def size(module):
            return sum(parameter.numel() for parameter in module.parameters())

        joint = tiny_clef(tokenizer, decision_config)
        recipe = counts(joint)
        # The decision head alone: what is neither the backbone nor the lm_head table
        head = sum(size(getattr(joint, name)) for name in ("embeddings", "encoder", "scorer"))
        assert recipe["head_parameters"] == head
        assert recipe["lexical_prior_parameters"] == joint.lm_head.weight.numel()
        assert recipe["total_parameters"] == size(joint.model) + head + joint.lm_head.weight.numel()
        # The cosine head has no lm_head; a tied lm_head belongs to the backbone
        cosine = tiny_clef(tokenizer, decision_config, head_type="cosine")
        recipe = counts(cosine)
        assert recipe["lexical_prior_parameters"] is None
        assert recipe["head_parameters"] == size(cosine.cosine_head) == size(cosine) - size(cosine.model)
        joint.lm_head.weight = joint.model.get_input_embeddings().weight
        assert counts(joint)["lexical_prior_parameters"] == 0
        assert counts(joint)["head_parameters"] == head
        # Every floating-point dtype of the backbone
        joint.model.layers[0].to(torch.bfloat16)
        assert counts(joint)["parameter_dtypes"] == ["bfloat16", "float32"]

    def test_config_settings_reach_the_example(self):
        from examples.decisions.benchmark_clef import run_benchmark, write_synthetic_recipe

        config = write_synthetic_recipe(Path(self.tmp_dir), records=48)
        config |= {
            "head_type": "cosine",
            "cosine_width": 8,
            "gradient_checkpointing": True,
            "per_device_train_batch_size": 4,
            "max_train_samples": 24,
            "max_eval_samples": 2,
        }
        r = run_benchmark(config, steps=3, process_start=(0.0, "test"))
        recipe = r["recipe"]
        assert recipe["synthetic"] is False and recipe["startup_origin"] == "test"
        assert r["startup_seconds"] > 1e9  # counted from the given start, the epoch
        assert recipe["head_type"] == "cosine" and recipe["activation_checkpointing"] is True
        assert recipe["lexical_prior_parameters"] is None
        assert recipe["records_per_step"] == 4
        assert recipe["build_report"]["train"]["rows"] == 24 and recipe["build_report"]["eval"]["rows"] == 2
        assert r["global_steps"] == 3 and len(r["steps"]) == 3
        # With the default warm-up, too few steps for a steady state: no steady numbers rather than made-up ones
        assert r["window_steps"] == 0
        for key in (
            "steady_records_per_second",
            "window_records_per_second",
            "steady_step_p50",
            "steady_step_p95",
            "steady_data_p50",
            "steady_records_per_second_with_data",
        ):
            assert r[key] is None, key

    def test_settings_are_parsed_as_the_example_parses_them(self):
        from examples.decisions.benchmark_clef import parse_settings

        # YAML reads 1e-4 as a string; the example's parser (`--config`) turns it into the field's type
        settings = {
            "output_dir": self.tmp_dir,
            "learning_rate": "1e-4",
            "head_learning_rate": "3e-4",
            "max_length": 64,
        }
        script_args, training_args, model_args = parse_settings(settings)
        assert training_args.learning_rate == 1e-4 and isinstance(training_args.learning_rate, float)
        assert script_args.head_learning_rate == 3e-4
        assert script_args.max_length == 64
        path = Path(self.tmp_dir, "settings.yaml")
        path.write_text("learning_rate: 1e-4\nmax_length: 64\n", encoding="utf-8")
        from trl import TrlParser

        parser = TrlParser((ClefScriptArguments, DecisionConfig, ClefModelArguments))
        example = parser.parse_args_and_config(["--config", str(path), "--output_dir", self.tmp_dir])
        assert example[1].learning_rate == training_args.learning_rate

    def test_recipe_redacts_model_init_kwargs(self, tokenizer, decision_config):
        from types import SimpleNamespace

        from examples.decisions.benchmark_clef import describe_recipe

        model = tiny_clef(tokenizer, decision_config)
        run = SimpleNamespace(trainer=SimpleNamespace(model=model), reports={})
        kwargs = {"token": "hf_secret", "dtype": "bfloat16", "revision": "main", "attn_implementation": "eager"}
        args = DecisionConfig(output_dir=self.tmp_dir, use_cpu=True, report_to="none", model_init_kwargs=kwargs)
        recipe = describe_recipe(run, ClefScriptArguments(), args, ClefModelArguments(), synthetic=False)
        assert "hf_secret" not in json.dumps(recipe, default=str)
        assert recipe["training"]["model_init_kwargs"] == {**kwargs, "token": "<redacted>"}

    def test_benchmark_states_what_its_timings_cover(self):
        from examples.decisions.benchmark_clef import run_benchmark

        # 20 records hold out 1; 19 training records in batches of 8 leave a partial batch, which is dropped
        r = run_benchmark(steps=4, warmup_steps_excluded=0, synthetic_records=20)
        assert r["recipe"]["training"]["dataloader_drop_last"] is True
        assert r["global_steps"] == 4
        assert set(r["notes"]) == {"trainer_wall_seconds", "startup_seconds"}
        assert "not comparable" in r["notes"]["trainer_wall_seconds"]
        assert "not comparable" in r["notes"]["startup_seconds"]

    def test_cli_writes_the_summary(self, capsys, monkeypatch):
        from examples.decisions import benchmark_clef

        # The CLI's own run, with a small recipe
        run = benchmark_clef.run_benchmark
        monkeypatch.setattr(benchmark_clef, "run_benchmark", lambda *a, **k: run(*a, **k, synthetic_records=24))
        output = Path(self.tmp_dir, "summary.json")
        benchmark_clef.main(["--steps", "2", "--warmup_steps_excluded", "1", "--output", str(output)])
        written = json.loads(output.read_text())
        assert written["global_steps"] == 2 and written["window_steps"] == 1
        assert written["recipe"]["startup_origin"] in {"process creation", "benchmark module import"}
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == written


class TestClefRemoteBackbone(TrlTestCase):
    """A head on a backbone that exists only as code in its repository, which runs only with `trust_remote_code`."""

    def write_config(self, folder, **settings) -> str:
        """A config file of the example on `folder` and a tiny dataset, as a user would write it."""
        import yaml

        rows = [make_row(index) for index in range(12)]
        Path(self.tmp_dir, "data").mkdir(exist_ok=True)
        with open(f"{self.tmp_dir}/data/train.jsonl", "w") as file:
            for row in rows:
                file.write(json.dumps({key: json.dumps(value) for key, value in row.items()}) + "\n")
        config = {
            "model_name_or_path": folder,
            "dataset_name": f"{self.tmp_dir}/data",
            "dataset_config": None,
            "validation_fraction": 0.25,
            "dtype": "float32",
            "mode": "full",
            "head_type": "cosine",
            "max_length": 400,
            "truncation": "skip",
            "permute_fields": True,
            "output_dir": f"{self.tmp_dir}/out",
            "use_cpu": True,
            "bf16": False,
            "gradient_checkpointing": False,
            "max_steps": 2,
            "learning_rate": 1e-3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "eval_strategy": "no",
            "save_strategy": "no",
            "report_to": "none",
            "trust_remote_code": 1,
            "model_init_kwargs": RAISED_CONTEXT,
            **settings,
        }
        if config["head_type"] == "joint":
            config.update(
                head_hidden_size=16,
                head_intermediate_size=32,
                head_num_layers=1,
                head_num_evidence_layers=1,
                head_num_attention_heads=2,
            )
        config = {key: value for key, value in config.items() if value is not ...}
        path = Path(self.tmp_dir, f"config-{len(list(Path(self.tmp_dir).glob('config-*')))}.yaml")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(path)

    @staticmethod
    def run_config(path, *argv):
        from trl import TrlParser

        parser = TrlParser((ClefScriptArguments, DecisionConfig, ClefModelArguments))
        return clef_main(*parser.parse_args_and_config(["--config", path, *argv]))

    def test_without_the_flag_the_script_refuses_the_package(self, tokenizer):
        # The script's default, and the flag set to false: refused from the package's config.json, before training
        folder = spark_package(self.tmp_dir, tokenizer)
        for flag in ({"trust_remote_code": ...}, {"trust_remote_code": False}):
            with pytest.raises(ValueError, match="trust_remote_code: true") as error:
                self.run_config(self.write_config(folder, **flag))
            assert folder in str(error.value) and "modeling_tinyspark.TinySparkForCausalLM" in str(error.value)
        assert not Path(self.tmp_dir, "out").exists()

    def test_package_with_a_base_model_is_loaded_as_that(self, tokenizer, capsys):
        folder = spark_package(self.tmp_dir, tokenizer, with_base_model=True)
        trainer = self.run_config(self.write_config(folder))
        assert trainer.state.global_step == 2
        assert type(trainer.model.model).__name__ == "TinySparkModel"
        out = capsys.readouterr().out
        # Its output has `last_hidden_state`: no submodule is needed
        assert "backbone weights: direct, AutoModel of its own code" in out
        assert "backbone_hidden_module" not in out and trainer.model.config.backbone_hidden_module is None
        reloaded = AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/out", trust_remote_code=True)
        assert type(reloaded.model).__name__ == "TinySparkModel"

    @pytest.mark.parametrize("value", ["true", "1"])
    def test_flag_parses_from_the_command_line(self, tokenizer, value):
        from trl import TrlParser

        parser = TrlParser((ClefScriptArguments, DecisionConfig, ClefModelArguments))
        argv = ["--output_dir", self.tmp_dir, "--trust_remote_code", value]
        *_, model_args = parser.parse_args_and_config(argv)
        assert model_args.trust_remote_code is True
        assert ClefModelArguments().trust_remote_code is False

    @pytest.mark.parametrize("head_type", ["cosine", "joint"])
    def test_trains_saves_answers_and_runs_rlcd_after(self, tokenizer, head_type, capsys):
        folder = spark_package(self.tmp_dir, tokenizer)
        trainer = self.run_config(self.write_config(folder, head_type=head_type))
        assert trainer.state.global_step == 2
        out = capsys.readouterr().out
        assert (
            f"Clef setup: causal backbone (tinyspark), global_pooling=last, prompt_style=chatml, head_type={head_type}, "
            "backbone_hidden_module=model.final_norm, backbone weights: direct, AutoModelForCausalLM of its own code"
        ) in out
        model = trainer.model
        # The causal language model is the backbone itself, its width read from the model, and no lm_head is added
        assert type(model.model).__name__ == "TinySparkForCausalLM"
        assert model.config.backbone_hidden_size == 24 and not hasattr(model, "lm_head")
        # Saved with its code, and loaded again only with the flag
        output = f"{self.tmp_dir}/out"
        for name in SPARK_PACKAGE:
            assert Path(output, name).is_file(), name
        with pytest.raises(ValueError, match="trust_remote_code=True"):
            AutoModelForDecision.from_pretrained(output)
        reloaded = AutoModelForDecision.from_pretrained(output, trust_remote_code=True)
        collator = DataCollatorForClef(tokenizer, model.decision_config)
        assert max_logit_difference(reloaded, model, collator([make_row(index) for index in range(2)])) == 0.0
        decide = pipeline("decision", model=output, trust_remote_code=True, device="cpu")
        answers = decide({"state": "ticket 3: refund alpha", "questions": QUESTIONS})
        assert set(answers) == set(QUESTIONS)
        # RLCD after cross-entropy: a second run from the saved model, with a schedule of noise scales
        config = self.write_config(
            output, head_type=head_type, output_dir=f"{self.tmp_dir}/rlcd", rlcd_weight=0.25, rlcd_sigmas=[0.4, 0.3]
        )
        trainer = self.run_config(config)
        assert trainer.state.global_step == 2
        assert "backbone weights: Clef checkpoint" in capsys.readouterr().out
        assert trainer.model.config.rlcd_weight == 0.25
        assert AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/rlcd", trust_remote_code=True)
        # The benchmark builds the same run from the saved model
        if head_type == "cosine":
            import yaml
            from examples.decisions.benchmark_clef import run_benchmark

            summary = run_benchmark(yaml.safe_load(Path(config).read_text()), steps=2, warmup_steps_excluded=0)
            assert summary["global_steps"] == 2 and summary["recipe"]["backbone"] == "TinySparkForCausalLM"

    def test_raised_context_is_saved_and_loads_again(self, tokenizer, capsys):
        folder = spark_package(self.tmp_dir, tokenizer)
        # Every prompt is longer than the package's own limit: a warning at start-up, then the model refuses
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            self.run_config(self.write_config(folder, model_init_kwargs={}))
        assert "above the max_position_embeddings of the tinyspark backbone (256)" in capsys.readouterr().out
        # Raised through model_init_kwargs: it trains, and the saved model has the same context
        trainer = self.run_config(self.write_config(folder))
        assert trainer.state.global_step == 2
        assert "max_position_embeddings" not in capsys.readouterr().out
        saved = json.loads(Path(self.tmp_dir, "out", "config.json").read_text(encoding="utf-8"))["text_config"]
        assert (saved["max_seq_len"], saved["max_position_embeddings"]) == (512, 512)
        decide = pipeline("decision", model=f"{self.tmp_dir}/out", trust_remote_code=True, device="cpu")
        assert decide.model.config.text_config.max_seq_len == 512
        row = make_row(0)
        prompt = DataCollatorForClef(tokenizer, decide.model.decision_config).tokenize(row)
        assert len(prompt["input_ids"]) > 256
        assert set(decide({"state": row["state"], "questions": row["questions"]})) == set(row["questions"])

    def test_activation_checkpointing_stops_at_start_up(self, tokenizer, monkeypatch):
        from examples.decisions import clef_typed_decisions

        folder = spark_package(self.tmp_dir, tokenizer)

        def trainer(*args, **kwargs):
            raise AssertionError("the trainer was built")

        monkeypatch.setattr(clef_typed_decisions, "DecisionTrainer", trainer)
        for settings in (
            {"activation_checkpointing": True, "gradient_checkpointing": ...},
            {"gradient_checkpointing": True},
        ):
            with pytest.raises(ValueError, match="activation_checkpointing.*TinySparkForCausalLM"):
                self.run_config(self.write_config(folder, **settings))

    @require_peft
    def test_lora_on_a_backbone_read_from_an_inner_layer_stops_at_start_up(self, tokenizer, monkeypatch):
        from examples.decisions import clef_typed_decisions

        folder = spark_package(self.tmp_dir, tokenizer)

        def trainer(*args, **kwargs):
            raise AssertionError("the trainer was built")

        monkeypatch.setattr(clef_typed_decisions, "DecisionTrainer", trainer)
        lora = {"mode": "lora", "use_peft": True, "lora_r": 4, "lora_target_modules": ["layers.0", "layers.1"]}
        with pytest.raises(ValueError, match=r"mode: lora cannot put adapters.*model\.final_norm.*mode: full"):
            self.run_config(self.write_config(folder, **lora))
        assert not Path(self.tmp_dir, "out").exists()

    def test_clef_checkpoint_with_its_own_classes_is_refused(self, tokenizer):
        import sys

        folder = spark_package(self.tmp_dir, tokenizer)
        self.run_config(self.write_config(folder))
        # A model saved by the example names no classes of its own; give it some, which record being imported
        output = Path(self.tmp_dir, "out")
        config = json.loads(Path(output, "config.json").read_text(encoding="utf-8"))
        assert "auto_map" not in config
        config["auto_map"] = {"AutoConfig": "evil_cfg.EvilConfig", "AutoModelForDecision": "evil_cfg.EvilModel"}
        Path(output, "config.json").write_text(json.dumps(config), encoding="utf-8")
        sentinel = Path(self.tmp_dir, "imported")
        Path(output, "evil_cfg.py").write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('evil')\nraise RuntimeError('evil imported')\n",
            encoding="utf-8",
        )
        for flag in (1, False):
            settings = {"trust_remote_code": flag, "output_dir": f"{self.tmp_dir}/again"}
            with pytest.raises(ValueError, match="Clef checkpoint whose config.json names classes of its own code"):
                self.run_config(self.write_config(str(output), **settings))
            assert not sentinel.exists()
            assert not any("evil_cfg" in module for module in sys.modules)
