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

import pytest
import torch
from datasets import Dataset


pytest.importorskip("transformers.decision")

from examples.decisions.clef_typed_decisions import (
    HEAD_MODULES,
    HEAD_PATTERN,
    ClefModelArguments,
    ClefScriptArguments,
    DataCollatorForClef,
    clef_decision_config,
)
from examples.decisions.clef_typed_decisions import main as clef_main
from transformers import (
    AutoModelForCausalLM,
    AutoModelForDecision,
    AutoTokenizer,
    ClefConfig,
    ClefForDecision,
    Qwen2Config,
    pipeline,
    set_seed,
)

from trl.experimental.decision import DecisionConfig, DecisionTrainer

from ..testing_utils import TrlTestCase, require_peft


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None, "alpha": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}
INSTRUCTIONS = {question["instructions"]: name for name, question in QUESTIONS.items()}
LONG_ROW = {"state": "word " * 200, "questions": QUESTIONS, "gold": {"refund": True, "team": "zeta", "urgency": 0}}


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


def tiny_clef(tokenizer, decision_config):
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
    model.decision_config = decision_config
    return model


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
        inputs = {key: value for key, value in batch.items() if key != "is_choice"}
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
    model = kwargs.pop("model", None) or tiny_clef(tokenizer, decision_config)
    trainer = DecisionTrainer(
        model=model,
        args=DecisionConfig(**{**args, **kwargs}),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
    )
    return trainer, collator, val


def max_logit_difference(first, second, batch):
    inputs = {key: value for key, value in batch.items() if key not in {"is_choice", "labels"}}
    with torch.no_grad():
        return (first.eval()(**inputs).logits - second.eval()(**inputs).logits).abs().max().item()


class TestClefTraining(TrlTestCase):
    def test_train_eval_save_reload(self, tokenizer, decision_config):
        trainer, collator, val = make_trainer(self.tmp_dir, tokenizer, decision_config)
        first = trainer.evaluate()["eval_loss"]
        trainer.train()
        trainer.save_model()
        assert trainer.evaluate()["eval_loss"] < first
        again = AutoModelForDecision.from_pretrained(self.tmp_dir).eval()
        assert isinstance(again, ClefForDecision)
        assert again.decision_config.chat_template == decision_config.chat_template
        batch = collator([val[i] for i in range(4)])
        assert max_logit_difference(again, trainer.model, batch) == 0.0

    @require_peft
    def test_lora_mode_trains_adapters_and_head_only(self, tokenizer, decision_config):
        from peft import LoraConfig

        model = tiny_clef(tokenizer, decision_config)
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
        peft_config = LoraConfig(
            r=4,
            target_modules=["q_proj", "v_proj"],
            exclude_modules=HEAD_PATTERN,
            modules_to_save=HEAD_MODULES,
            task_type="FEATURE_EXTRACTION",
        )
        trainer, _, _ = make_trainer(
            self.tmp_dir,
            tokenizer,
            decision_config,
            model=model,
            peft_config=peft_config,
            max_steps=5,
            eval_strategy="no",
        )
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        trainer.train()
        changed = {
            name for name, parameter in trainer.model.named_parameters() if not torch.equal(before[name], parameter)
        }
        trainable = {name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
        assert changed <= trainable, "a frozen backbone weight changed"
        assert any("lora_" in name for name in changed)
        assert any("modules_to_save" in name and "encoder" in name for name in changed)
        assert any("modules_to_save" in name and "scorer" in name for name in changed)
        assert all("lora_" in name or "modules_to_save" in name for name in trainable)
        # Adapters sit on the backbone, not on the head
        assert not any("lora_" in name and any(f".{head}." in name for head in HEAD_MODULES) for name in trainable)
        backbone_base = [name for name in before if ".model.layers." in name and "lora_" not in name]
        assert backbone_base
        assert not any(name in changed for name in backbone_base)

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

    def test_main_builds_plain_backbone_and_reports(self, tokenizer, decision_config, capsys):
        # A local causal LM becomes the backbone of a new head; the build report counts skipped records
        backbone = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        backbone.save_pretrained(f"{self.tmp_dir}/backbone")
        tokenizer.save_pretrained(f"{self.tmp_dir}/backbone")
        rows = [make_row(index) for index in range(12)] + [LONG_ROW]
        with open(f"{self.tmp_dir}/train.jsonl", "w") as file:
            for row in rows:
                file.write(json.dumps({key: json.dumps(value) for key, value in row.items()}) + "\n")
        script_args = ClefScriptArguments(
            dataset_name=self.tmp_dir,
            dataset_config=None,
            validation_fraction=0.25,
            mode="head",
            head_hidden_size=16,
            head_intermediate_size=32,
            head_num_layers=1,
            head_num_evidence_layers=1,
            head_num_attention_heads=2,
            max_length=400,
            truncation="skip",
            permute_fields=True,
        )
        training_args = DecisionConfig(
            output_dir=f"{self.tmp_dir}/out",
            use_cpu=True,
            bf16=False,
            gradient_checkpointing=False,
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
        )
        model_args = ClefModelArguments(model_name_or_path=f"{self.tmp_dir}/backbone", dtype="float32")
        clef_main(script_args, training_args, model_args)
        report = capsys.readouterr().out
        assert "Build report (train)" in report and "Build report (eval)" in report
        assert "longer than max_length" in report
        saved = AutoModelForDecision.from_pretrained(f"{self.tmp_dir}/out")
        assert isinstance(saved, ClefForDecision)
        assert saved.decision_config.chat_template == decision_config.chat_template
        # Head-only: the backbone is the one that was loaded
        loaded = copy.deepcopy(backbone.base_model.state_dict())
        for name, value in saved.model.state_dict().items():
            torch.testing.assert_close(value.float(), loaded[name].float())
