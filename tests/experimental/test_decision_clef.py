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
)
from examples.decisions.clef_typed_decisions import main as clef_main
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForDecision,
    AutoTokenizer,
    ClefConfig,
    ClefForDecision,
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

    def run_main(self, tokenizer, mode, **model_kwargs):
        """Run the script's `main` on a tiny local backbone and dataset; returns the trainer and the arguments."""
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
        training_args = DecisionConfig(
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
        model_args = ClefModelArguments(model_name_or_path=f"{self.tmp_dir}/backbone", dtype="float32", **model_kwargs)
        trainer = clef_main(script_args, training_args, model_args)
        set_seed(training_args.seed)
        initial = build_model(script_args, model_args, {"dtype": "float32", "revision": None})
        return trainer, initial, backbone

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
