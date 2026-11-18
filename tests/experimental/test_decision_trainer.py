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
import os
from functools import partial

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset


pytest.importorskip("transformers.decision")

from examples.decisions.clm_typed_decisions import DataCollatorForClm
from examples.decisions.jev import tokenize as tokenize_jev
from examples.decisions.jeva_banking77 import DataCollatorForJeva
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from transformers.decision import DecisionConfig as TransformersDecisionConfig
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from trl.experimental.decision import (
    DataCollatorForDecision,
    DecisionConfig,
    DecisionTrainer,
    compute_calibration_metrics,
    target_from_gold,
)

from ..testing_utils import TrlTestCase, require_peft


HAS_JEVA = "jeva" in CONFIG_MAPPING
HAS_CLM = "clm" in CONFIG_MAPPING
HAS_JEV_LOSS = "JevForDecision" in LOSS_MAPPING
# Transformers #50058 adds Clef; its training (native loss, batched forward, head choice, RLCD) comes on top of it
HAS_CLEF_TRAINING = "ClefForDecision" in LOSS_MAPPING
REQUIRES_CLEF_TRAINING = (
    "Requires a Transformers build with Clef training support (native loss and batched forward) on top of #50058"
)

if HAS_CLEF_TRAINING:
    from transformers.models.clef.configuration_clef import CLEF_TEMPLATE

if HAS_JEVA:
    from transformers import AutoModelForDecision, JevaForDecision
    from transformers import JevaConfig as JevaModelConfig


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def classification_dataset(tokenizer):
    encoded = tokenizer(["A refund", "A delivery", "A complaint", "Another refund", "Another delivery"])
    return Dataset.from_dict({**encoded, "labels": [0, 1, 2, 0, 1]})


def make_classification_model():
    set_seed(42)
    return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3, dtype=torch.float32)


def jev_docstring_command():
    """The command line of the docstring of `jev.py`, without `python examples/decisions/jev.py`."""
    import shlex

    import examples.decisions.jev as jev

    command = " ".join(line.rstrip("\\ ") for line in jev.__doc__.splitlines() if line.strip()).split("jev.py", 1)
    return shlex.split(command[1])


@pytest.mark.parametrize("example", ["jeva_banking77", "clm_typed_decisions", "jev", "clef_typed_decisions"])
def test_example_parsers_build(example):
    # The parser each example's `__main__` block builds, on the configuration it is supplied with: no option may be
    # defined twice (by the example and by `DecisionConfig`)
    import importlib

    from trl import ModelConfig, ScriptArguments, TrlParser

    module = importlib.import_module(f"examples.decisions.{example}")
    dataclass_types = {
        "jeva_banking77": lambda: (module.JevaScriptArguments, DecisionConfig, module.JevaModelArguments),
        "clm_typed_decisions": lambda: (module.ClmScriptArguments, DecisionConfig, module.ClmModelArguments),
        "jev": lambda: (ScriptArguments, DecisionConfig, ModelConfig),
        "clef_typed_decisions": lambda: (module.ClefScriptArguments, DecisionConfig, module.ClefModelArguments),
    }[example]()
    parser = TrlParser(dataclass_types)
    if example == "jev":
        args = jev_docstring_command()
    else:
        args = ["--config", os.path.join(os.path.dirname(module.__file__), f"{example}.yaml")]
    parsed = parser.parse_args_and_config(args)
    assert [type(value) for value in parsed] == list(dataclass_types)


class TestDecisionTrainer(TrlTestCase):
    @pytest.mark.parametrize("use_peft", [False, pytest.param(True, marks=require_peft)])
    def test_train_and_predict(self, tokenizer, classification_dataset, use_peft):
        peft_config = None
        if use_peft:
            from peft import LoraConfig

            peft_config = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")

        def compute_metrics(prediction):
            return {"accuracy": float((prediction.predictions.argmax(-1) == prediction.label_ids).mean())}

        trainer = DecisionTrainer(
            model=make_classification_model(),
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_steps=1,
                report_to="none",
            ),
            train_dataset=classification_dataset,
            eval_dataset=classification_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            peft_config=peft_config,
        )
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert any(not torch.equal(before[name], parameter) for name, parameter in trainer.model.named_parameters())
        prediction = trainer.predict(classification_dataset)
        assert prediction.predictions.shape == (len(classification_dataset), 3)
        np.testing.assert_allclose(prediction.predictions.sum(-1), 1, atol=1e-6)
        np.testing.assert_array_equal(prediction.label_ids, classification_dataset["labels"])
        metrics = trainer.evaluate()
        assert metrics["eval_choice_accuracy"] == pytest.approx(metrics["eval_accuracy"])
        assert "eval_coverage_loss" not in metrics
        assert "eval_null_precision" not in metrics
        trainer.save_model()

    def test_accumulation(self, tokenizer, classification_dataset):
        models = []
        for batch_size, accumulation in ((5, 1), (3, 2)):
            model = make_classification_model()
            trainer = DecisionTrainer(
                model=model,
                args=DecisionConfig(
                    output_dir=self.tmp_dir,
                    use_cpu=True,
                    bf16=False,
                    max_steps=1,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    max_grad_norm=0,
                    lr_scheduler_type="constant",
                    report_to="none",
                ),
                train_dataset=classification_dataset,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
            )
            trainer.train()
            models.append(model)
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.fixture
def dataset():
    records = [
        (
            "A billing question",
            {"type": "choice", "instructions": "Choose a team", "criteria": {"billing": None, "technical": None}},
            [0, 1, 0],
        ),
        (
            "No service applies",
            {"type": "choice", "instructions": "Choose a service", "criteria": {"shipping": None}, "allow_null": True},
            [1, 0],
        ),
        ("The answer is mostly true", {"type": "noul", "instructions": "Is the answer true?"}, [0, 0.8, 0.2]),
        (
            "A good answer",
            {"type": "score", "instructions": "Rate the answer", "criteria": ["bad", "good", "great"]},
            [0, 0, 0.5, 0.5],
        ),
        (
            "A refund",
            {
                "type": "choice",
                "instructions": "Choose a team",
                "criteria": {"yes": None, "no": None},
                "allow_null": True,
            },
            [0.1, 0.7, 0.2],
        ),
    ]
    return Dataset.from_dict(
        {
            "request": [
                json.dumps({"state": state, "questions": {"decision": question}}) for state, question, _ in records
            ],
            "target_probs": [target for _, _, target in records],
        }
    )


def make_model(model_id=MODEL_ID, dtype=torch.float32, mode="head", base_path=None):
    set_seed(42)
    encoder = AutoModel.from_pretrained(model_id, dtype=dtype)
    model = JevaForDecision(
        JevaModelConfig(text_config=encoder.config, projection_dim=16, decision_num_heads=2), model=encoder
    )
    model.base_model.requires_grad_(mode == "full")
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        model.save_pretrained(base_path)
        model.config._name_or_path = str(base_path)
        config = LoraConfig(
            r=2,
            target_modules=r"model\..*\.(q_proj|v_proj)",
            task_type="FEATURE_EXTRACTION",
            modules_to_save=["context_proj", "option_proj", "core"],
        )
        model = get_peft_model(model, config)
    return model


@pytest.fixture
def collator(tokenizer):
    return DataCollatorForJeva(tokenizer, make_model().decision_config, option_chunk_size=1)


@pytest.mark.skipif(not HAS_JEVA, reason="Requires Transformers #50614")
class TestDecisionTrainerJeva(TrlTestCase):
    @pytest.mark.parametrize("mode", ["head", "full", pytest.param("lora", marks=require_peft)])
    def test_train(self, mode, tokenizer, dataset, collator):
        dataset = dataset.map(collator.tokenize, remove_columns=dataset.column_names)
        model = make_model(mode=mode, base_path=os.path.join(self.tmp_dir, "base_model"))
        trainer = DecisionTrainer(
            model=model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                gradient_checkpointing=mode != "head",
                max_steps=2,
                per_device_train_batch_size=2,
                logging_steps=1,
                report_to="none",
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        logs = trainer.state.log_history[0]
        assert logs["loss"] == pytest.approx(logs["candidate_loss"] + logs["coverage_loss"], abs=1e-4)
        changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
        assert any("core." in name for name in changed)
        if mode == "head":
            assert not any(name.startswith("model.") for name in changed)
        elif mode == "lora":
            assert any("lora_" in name for name in changed)
            assert all("lora_" in name for name in changed if ".model.model." in name)
        else:
            assert any(name.startswith("model.") for name in changed)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_loss_and_predictions(self, tokenizer, dataset, batch_size, collator):
        model = make_model().eval()
        batch = collator(list(dataset))
        with torch.no_grad():
            outputs = model(
                **{
                    key: batch[key]
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "option_input_ids",
                        "option_attention_mask",
                        "num_options",
                        "option_mask",
                    )
                }
            )
        log_q = outputs.logits.masked_fill(~batch["option_mask"], 0)
        expected = -(batch["labels"][:, 1:] * log_q).sum(-1)
        expected += (
            F.binary_cross_entropy_with_logits(outputs.coverage_logits, 1 - batch["labels"][:, 0], reduction="none")
            * batch["allow_null"]
        )

        def compute_metrics(prediction):
            probs, labels = prediction.predictions, prediction.label_ids
            np.testing.assert_allclose(np.where(probs == -100, 0, probs).sum(-1), 1, atol=1e-6)
            assert np.array_equal(probs == -100, labels == -100)
            return {"accuracy": float((probs.argmax(-1) == labels.argmax(-1)).mean())}

        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                report_to="none",
                per_device_eval_batch_size=batch_size,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        prediction = trainer.predict(dataset)
        assert prediction.predictions.shape == (len(dataset), 4)
        assert prediction.metrics["test_loss"] == pytest.approx(expected.mean().item(), rel=1e-5)
        assert "eval_holdout_accuracy" in trainer.evaluate({"holdout": dataset})
        metrics = trainer.evaluate()
        probs = torch.tensor(np.where(prediction.predictions == -100, 0, prediction.predictions))
        predicted = probs.argmax(-1)
        target = batch["labels"].argmax(-1)
        hard_choice = batch["is_choice"] & (batch["labels"].max(-1).values == 1)
        predicted_null = (predicted == 0) & hard_choice
        target_null = (target == 0) & hard_choice
        true_null = (predicted_null & target_null).sum().item()
        assert metrics["eval_candidate_loss"] + metrics["eval_coverage_loss"] == pytest.approx(
            expected.mean().item(), rel=1e-5
        )
        assert metrics["eval_choice_accuracy"] == pytest.approx(
            ((predicted == target) & hard_choice).sum().item() / hard_choice.sum().item()
        )
        present = hard_choice & (target != 0)
        assert metrics["eval_candidate_accuracy"] == pytest.approx(
            ((outputs.logits.argmax(-1) + 1 == target) & present).sum().item() / present.sum().item()
        )
        assert metrics["eval_null_precision"] == pytest.approx(true_null / max(predicted_null.sum().item(), 1))
        assert metrics["eval_null_recall"] == pytest.approx(true_null / target_null.sum().item())
        assert metrics["eval_null_rate"] == pytest.approx(
            ((predicted == 0) & batch["allow_null"]).sum().item() / batch["allow_null"].sum().item()
        )
        assert metrics["eval_brier"] == pytest.approx(
            (probs - batch["labels"]).square().sum(-1).mean().item(), abs=1e-6
        )
        assert metrics["eval_null_brier"] == pytest.approx(
            (probs[:, 0] - batch["labels"][:, 0]).square()[batch["allow_null"]].mean().item(), abs=1e-6
        )
        assert metrics["eval_entropy"] == pytest.approx(
            -(probs * probs.clamp_min(1e-30).log()).sum(-1).mean().item(), abs=1e-6
        )

    @pytest.mark.parametrize("pooling", ["last", "mean", "cls"])
    def test_permutation_and_chunks(self, pooling, tokenizer, dataset, collator):
        model = make_model().eval()
        model.config.pooling = pooling
        row = dict(dataset[3])
        request = json.loads(row["request"])
        request["questions"]["decision"]["criteria"].reverse()
        permuted = {**row, "request": json.dumps(request), "target_probs": [0, 0.5, 0.5, 0]}
        batch = collator([row])
        reordered = collator([permuted])
        with torch.no_grad():
            outputs = model(
                **{
                    key: batch[key]
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "option_input_ids",
                        "option_attention_mask",
                        "num_options",
                        "option_mask",
                    )
                }
            )
            actual = model(
                **{
                    key: reordered[key]
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "option_input_ids",
                        "option_attention_mask",
                        "num_options",
                        "option_mask",
                    )
                },
            )
        torch.testing.assert_close(actual.logits.flip(-1), outputs.logits)
        torch.testing.assert_close(actual.coverage_logits, outputs.coverage_logits)

    def test_accumulation(self, tokenizer, dataset, collator):
        trained_models = []
        for batch_size, accumulation in ((5, 1), (3, 2)):
            model = make_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            trainer = DecisionTrainer(
                model,
                args=DecisionConfig(
                    output_dir=self.tmp_dir,
                    use_cpu=True,
                    bf16=False,
                    gradient_checkpointing=False,
                    max_steps=1,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    max_grad_norm=0,
                    lr_scheduler_type="constant",
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                data_collator=collator,
                optimizers=(optimizer, None),
            )
            trainer.train()
            trained_models.append(model)
        for a, b in zip(trained_models[0].parameters(), trained_models[1].parameters(), strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    @require_peft
    def test_checkpoint_and_resume(self, tokenizer, dataset, collator):
        args = DecisionConfig(
            output_dir=self.tmp_dir,
            use_cpu=True,
            bf16=False,
            gradient_checkpointing=False,
            max_steps=2,
            save_steps=1,
            report_to="none",
        )
        trainer = DecisionTrainer(
            make_model(mode="lora", base_path=os.path.join(self.tmp_dir, "base_model")),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        trainer.train()
        checkpoint = os.path.join(self.tmp_dir, "checkpoint-2")
        from peft import PeftModel

        base_path = os.path.join(self.tmp_dir, "base_model")
        reloaded = PeftModel.from_pretrained(AutoModelForDecision.from_pretrained(base_path), checkpoint)
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])
        assert os.path.isfile(os.path.join(checkpoint, "optimizer.pt"))
        resumed = PeftModel.from_pretrained(
            AutoModelForDecision.from_pretrained(base_path), checkpoint, is_trainable=True
        )
        trainer = DecisionTrainer(
            resumed, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=collator
        )
        trainer.train(resume_from_checkpoint=os.path.join(self.tmp_dir, "checkpoint-1"))
        assert trainer.state.global_step == 2
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])

    def test_bfloat16_checkpoint(self):
        model = make_model(dtype=torch.bfloat16)
        model.save_pretrained(self.tmp_dir)
        reloaded = AutoModelForDecision.from_pretrained(self.tmp_dir)
        assert reloaded.model.dtype == torch.bfloat16
        assert reloaded.context_proj.weight.dtype == torch.float32
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])

    @pytest.mark.parametrize(
        "model_id", ["trl-internal-testing/tiny-LlamaForCausalLM-3.2", "trl-internal-testing/tiny-Qwen3MoeForCausalLM"]
    )
    def test_backbones(self, model_id, dataset):
        model = make_model(model_id, mode="full")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForJeva(tokenizer, model.decision_config)
        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                gradient_checkpointing=False,
                max_steps=1,
                logging_steps=1,
                report_to="none",
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        assert model.get_output_embeddings() is None
        result = trainer.train()
        assert np.isfinite(result.training_loss)


@pytest.mark.skipif(not HAS_JEVA, reason="Requires Transformers #50614")
class TestDataCollatorForJeva(TrlTestCase):
    @pytest.mark.parametrize("pretokenized", [False, True])
    def test_batching(self, tokenizer, dataset, collator, pretokenized):
        rows = list(dataset)
        rows[0] = {**rows[0], "request": json.loads(rows[0]["request"])}
        request = json.loads(rows[1].pop("request"))
        rows[1].update(request)
        expected_batch = collator(rows)
        if pretokenized:
            rows = list(dataset.map(collator.tokenize, remove_columns=dataset.column_names))
        batch = collator(rows, return_tensors="pt")
        for key, value in expected_batch.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(batch[key], value)
            else:
                assert batch[key] == value
        torch.testing.assert_close(batch["num_options"], torch.tensor([2, 1, 2, 3, 2]))
        torch.testing.assert_close(
            batch["option_mask"],
            torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                    [True, True, False],
                    [True, True, True],
                    [True, True, False],
                ]
            ),
        )
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0.8, 0.2, 0], [0, 0, 0.5, 0.5], [0.1, 0.7, 0.2, 0]]),
        )
        torch.testing.assert_close(batch["allow_null"], torch.tensor([False, True, False, False, True]))
        torch.testing.assert_close(batch["is_choice"], torch.tensor([True, True, False, False, True]))
        singles = [collator([row]) for row in rows]
        for prefix in ("", "option_"):
            ids, mask = batch[f"{prefix}input_ids"], batch[f"{prefix}attention_mask"]
            assert ids.shape[0] == (5 if not prefix else 10)
            assert (ids[mask == 0] == tokenizer.pad_token_id).all()
            assert (mask[:, 1:] <= mask[:, :-1]).all()
            expected = torch.cat(
                [single[f"{prefix}input_ids"][single[f"{prefix}attention_mask"].bool()] for single in singles]
            )
            torch.testing.assert_close(ids[mask.bool()], expected)

    @pytest.mark.parametrize(
        "question, label, target",
        [
            ({"type": "choice", "criteria": {"a": None, "b": None}}, {"target_index": 1}, [0, 0, 1]),
            ({"type": "choice", "criteria": {"a": None}, "allow_null": True}, {"target_index": None}, [1, 0]),
            ({"type": "noul"}, {"p_true": 0.8}, [0, 0.2, 0.8]),
            ({"type": "score", "criteria": ["bad", "good"], "values": [1, 3]}, {"score": 2}, [0, 0.5, 0.5]),
        ],
    )
    def test_decision_labels(self, collator, question, label, target):
        batch = collator([{"state": "A state", "questions": {"decision": question}, **label}])
        torch.testing.assert_close(batch["labels"], torch.tensor([target], dtype=torch.float32))

    def test_truncation(self, tokenizer, collator):
        row = {
            "state": "A long state",
            "questions": {
                "decision": {"type": "choice", "instructions": "Decide", "criteria": {"a": None, "b": None}}
            },
            "target_index": 0,
        }
        with pytest.raises(ValueError, match="token limit"):
            DataCollatorForJeva(tokenizer, collator.decision_config, max_state_length=2)([row])
        batch = DataCollatorForJeva(tokenizer, collator.decision_config, max_state_length=2, truncation="truncate")(
            [row]
        )
        assert batch["input_ids"].shape[1] == 2
        tokenizer.truncation_side = "left"
        truncated = DataCollatorForJeva(
            tokenizer, collator.decision_config, max_state_length=2, truncation="truncate"
        )([row])
        torch.testing.assert_close(truncated["input_ids"], collator([row])["input_ids"][:, -2:])


@pytest.mark.skipif(not HAS_CLM, reason="Requires Transformers #50649")
class TestDataCollatorForClm(TrlTestCase):
    def test_contrastive_training(self, tokenizer):
        from transformers import ClmConfig, ClmForDecision

        encoder = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
        model = ClmForDecision(
            ClmConfig(text_config=encoder.config, hidden_size=24, projection_dim=8, num_hidden_layers=3), model=encoder
        )
        model.base_model.requires_grad_(False)
        collator = DataCollatorForClm(tokenizer, model.decision_config)
        records = [
            ("An invoice", {"billing": None, "delivery": None}, [0, 0.75, 0.25]),
            ("A different request", {"delivery": None, "other": None}, [0, 0.2, 0.8]),
            ("Another invoice", {"billing": None, "delivery": None}, [0, 1, 0]),
        ]
        dataset = Dataset.from_list(
            [
                {
                    "request": json.dumps(
                        {"state": state, "questions": {"intent": {"type": "choice", "criteria": criteria}}}
                    ),
                    "target_probs": target,
                }
                for state, criteria, target in records
            ]
        )
        prepared = dataset.map(collator.tokenize, remove_columns=dataset.column_names)
        batch = collator(list(prepared))
        torch.testing.assert_close(batch["num_options"], torch.tensor([[2, 1, 0]]))
        targets = torch.tensor([[0.75, 0.25, 0], [0, 0.2, 0.8], [1, 0, 0]])
        torch.testing.assert_close(batch["labels"], targets)
        torch.testing.assert_close(
            batch["option_mask"], torch.tensor([[True, True, False], [False, True, True], [True, True, False]])
        )
        output = model(**{key: value for key, value in batch.items() if key != "is_choice"})
        reverse = targets.T / targets.sum(0)[:, None]
        expected = (F.cross_entropy(output.logits, targets) + F.cross_entropy(output.logits.T, reverse)) / 2
        torch.testing.assert_close(output.loss, expected)
        before = model.state_head.input_projection.weight.detach().clone()
        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=3,
                per_device_eval_batch_size=3,
                report_to="none",
            ),
            train_dataset=prepared,
            eval_dataset=prepared,
            processing_class=tokenizer,
            data_collator=collator,
        )
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert not torch.equal(before, model.state_head.input_projection.weight)
        assert all(parameter.grad is None for parameter in model.base_model.parameters())
        prediction = trainer.predict(prepared)
        assert prediction.predictions.shape == (3, 3)
        np.testing.assert_allclose(
            np.where(prediction.predictions == -100, 0, prediction.predictions).sum(-1), 1, atol=1e-6
        )


@pytest.mark.skipif(not HAS_JEV_LOSS, reason="Requires Transformers #50586")
class TestDecisionTrainerJev(TrlTestCase):
    def test_training_and_predictions(self, tokenizer):
        from transformers import JevConfig, JevForDecision

        encoder = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
        model = JevForDecision(JevConfig(text_config=encoder.config, hidden_size=encoder.config.hidden_size))
        model.model = encoder
        collator = DataCollatorForDecision(tokenizer, model.decision_config)
        rows = [
            (
                "An invoice",
                {"type": "choice", "criteria": {"billing": None, "delivery": None}},
                {"target_probs": [0, 0.75, 0.25]},
            ),
            ("An urgent invoice", {"type": "noul", "instructions": "Is this urgent?"}, {"p_true": 0.8}),
            ("A good answer", {"type": "score", "criteria": ["bad", "good", "great"]}, {"score": 1.5}),
        ]
        dataset = Dataset.from_list(
            [
                {
                    "request": json.dumps({"state": state, "questions": {"decision": question}}),
                    "target_probs": None,
                    "p_true": None,
                    "score": None,
                    **target,
                }
                for state, question, target in rows
            ]
        )
        prepared = dataset.map(partial(tokenize_jev, collator=collator), remove_columns=dataset.column_names)
        batch = default_data_collator([prepared[0]])
        with torch.no_grad():
            expected = model(**{key: value for key, value in batch.items() if key != "is_choice"}).loss.item()
        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                gradient_checkpointing=False,
                max_steps=2,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                report_to="none",
            ),
            train_dataset=prepared,
            eval_dataset=prepared,
            processing_class=tokenizer,
            data_collator=default_data_collator,
        )
        assert trainer.evaluate(prepared.select([0]))["eval_loss"] == pytest.approx(expected)
        before = model.proj.weight.detach().clone()
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert not torch.equal(before, model.proj.weight)
        prediction = trainer.predict(prepared)
        assert prediction.predictions.shape == (3, 3)
        np.testing.assert_allclose(
            np.where(prediction.predictions == -100, 0, prediction.predictions).sum(-1), 1, atol=1e-6
        )


THREE_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None, "alpha": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}
THREE_TARGETS = {
    "refund": {"p_true": 0.75},
    "team": {"target_index": 0},
    "urgency": {"score": 1},
}


@pytest.mark.skipif(not HAS_CLEF_TRAINING, reason=REQUIRES_CLEF_TRAINING)
class TestDataCollatorForDecisionRequests(TrlTestCase):
    @staticmethod
    def make_collator(tokenizer, per_question=False):
        config = (
            TransformersDecisionConfig()
            if per_question
            else TransformersDecisionConfig(
                noul_options=["true", "false"], sort_choices=True, chat_template=CLEF_TEMPLATE
            )
        )
        return DataCollatorForDecision(tokenizer, config)

    def test_prepare_request_returns_one_target_per_question(self, tokenizer):
        collator = self.make_collator(tokenizer)
        inputs, questions, targets = collator.prepare_request(
            {"state": "s", "questions": THREE_QUESTIONS, "targets": THREE_TARGETS}
        )
        assert "input_ids" in inputs
        assert [question.name for question in questions] == ["refund", "team", "urgency"]
        assert len(targets) == 3
        assert [len(target) for target in targets] == [len(question.options) + 1 for question in questions]
        assert targets[0] == pytest.approx((0, 0.75, 0.25))
        assert targets[2] == pytest.approx((0, 0, 1, 0))

    def test_targets_follow_template_option_order(self, tokenizer):
        collator = self.make_collator(tokenizer)
        _, questions, targets = collator.prepare_request(
            {"state": "s", "questions": THREE_QUESTIONS, "targets": {**THREE_TARGETS, "team": {"target_index": 0}}}
        )
        assert questions[1].options == ["alpha", "zeta"]
        # target_index counts in template order, so index 0 is "alpha"
        assert targets[1] == (0.0, 1.0, 0.0)
        # gold names the key, so the mass follows "zeta" to the second slot
        _, _, targets = collator.prepare_request(
            {"state": "s", "questions": THREE_QUESTIONS, "gold": {"refund": True, "team": "zeta", "urgency": 2}}
        )
        assert targets[1] == (0.0, 0.0, 1.0)
        assert targets[0] == (0.0, 1.0, 0.0)
        assert targets[2] == (0.0, 0.0, 0.0, 1.0)

    def test_gold_label_and_probabilities(self, tokenizer):
        collator = self.make_collator(tokenizer)
        _, questions, _ = collator.prepare_request(
            {"state": "s", "questions": THREE_QUESTIONS, "targets": THREE_TARGETS}
        )
        refund, team, urgency = questions
        assert target_from_gold("b", team.__class__("c", "choice", None, ["a", "b"], ["A", "B"], [None, None])) == {
            "target_probs": [0.0, 0.0, 1.0]
        }
        assert target_from_gold({"noul": 0.75}, refund) == {"target_probs": [0.0, 0.75, 0.25]}
        assert target_from_gold({"label": "false"}, refund) == {"target_probs": [0.0, 0.0, 1.0]}
        assert target_from_gold(False, refund) == {"target_probs": [0.0, 0.0, 1.0]}
        assert target_from_gold({"probabilities": {"zeta": 3, "alpha": 1}}, team) == {
            "target_probs": [0.0, 0.25, 0.75]
        }
        assert target_from_gold({"probabilities": {"zeta": 1, "alpha": -5}}, team) == {"target_probs": [0.0, 0.0, 1.0]}
        assert target_from_gold("1", urgency) == {"target_probs": [0.0, 0.0, 1.0, 0.0]}
        assert target_from_gold(2, urgency) == {"target_probs": [0.0, 0.0, 0.0, 1.0]}
        with pytest.raises(ValueError, match="no usable label or probabilities"):
            target_from_gold("missing", team)
        with pytest.raises(ValueError, match="finite"):
            target_from_gold({"probabilities": {"alpha": float("nan")}}, team)
        # JSON-encoded gold works the same as a dict
        _, _, targets = collator.prepare_request(
            {
                "state": "s",
                "questions": THREE_QUESTIONS,
                "gold": json.dumps({"refund": {"noul": 0.75}, "team": "alpha", "urgency": 0}),
            }
        )
        assert targets[0] == pytest.approx((0, 0.75, 0.25))

    def test_zero_sum_probabilities_fall_back_to_the_label(self, tokenizer):
        collator = self.make_collator(tokenizer)
        _, (refund, team, urgency), _ = collator.prepare_request(
            {"state": "s", "questions": THREE_QUESTIONS, "targets": THREE_TARGETS}
        )
        zero = {"label": "zeta", "probabilities": {"alpha": 0.0, "zeta": -1.0, "omega": 5.0}}
        assert target_from_gold(zero, team) == {"target_probs": [0.0, 0.0, 1.0]}
        with pytest.raises(ValueError, match="no usable label or probabilities"):
            target_from_gold({"probabilities": {"alpha": 0.0}}, team)

    def test_answers_form_of_targets(self, tokenizer):
        collator = self.make_collator(tokenizer)
        answers = {"refund": {"noul": 0.75}, "team": "zeta", "urgency": {"label": 1}}
        _, _, from_answers = collator.prepare_request({"state": "s", "questions": THREE_QUESTIONS, "answers": answers})
        _, _, from_gold = collator.prepare_request({"state": "s", "questions": THREE_QUESTIONS, "gold": answers})
        assert from_answers == from_gold
        assert from_answers[1] == (0.0, 0.0, 1.0) and from_answers[2] == (0.0, 0.0, 1.0, 0.0)
        _, labels = collator.prepare_labeled_targets(
            {"answers": json.dumps(answers)},
            collator.prepare_request({"state": "s", "questions": THREE_QUESTIONS, "answers": answers})[1],
        )
        assert labels == [0, 1, 1]

    def test_targets_in_more_than_one_form_raise(self, tokenizer):
        collator = self.make_collator(tokenizer)
        gold = {"refund": True, "team": "zeta", "urgency": 2}
        for other in ({"gold": gold}, {"answers": gold}):
            with pytest.raises(ValueError, match="exactly one"):
                collator.prepare_request(
                    {"state": "s", "questions": THREE_QUESTIONS, "targets": THREE_TARGETS, **other}
                )
        with pytest.raises(ValueError, match="exactly one"):
            collator.prepare_request({"state": "s", "questions": THREE_QUESTIONS, "gold": gold, "answers": gold})

    @pytest.mark.parametrize("target", [1, "zeta", [0.0, 1.0], None])
    def test_a_target_that_is_not_a_mapping_is_a_value_error(self, tokenizer, target):
        collator = self.make_collator(tokenizer)
        with pytest.raises(ValueError, match="'team'"):
            collator.prepare_request(
                {"state": "s", "questions": THREE_QUESTIONS, "targets": {**THREE_TARGETS, "team": target}}
            )

    def test_missing_target_names_the_question(self, tokenizer):
        collator = self.make_collator(tokenizer)
        with pytest.raises(ValueError, match="'urgency'"):
            collator.prepare_request(
                {
                    "state": "s",
                    "questions": THREE_QUESTIONS,
                    "targets": {"refund": THREE_TARGETS["refund"], "team": THREE_TARGETS["team"]},
                }
            )

    def test_unknown_target_name_raises(self, tokenizer):
        collator = self.make_collator(tokenizer)
        with pytest.raises(ValueError, match="'colour'"):
            collator.prepare_request(
                {
                    "state": "s",
                    "questions": THREE_QUESTIONS,
                    "targets": {**THREE_TARGETS, "colour": {"target_index": 0}},
                }
            )

    def test_single_question_rows_still_work(self, tokenizer):
        collator = self.make_collator(tokenizer, per_question=True)
        question = {"type": "choice", "instructions": "Which team?", "criteria": {"zeta": None, "alpha": None}}
        inputs, parsed, target = collator.prepare_example(
            {"state": "s", "questions": {"team": question}, "target_index": 1}
        )
        assert parsed.name == "team" and parsed.options == ["zeta", "alpha"]
        assert target == (0.0, 0.0, 1.0)
        assert "input_ids" in inputs
        with pytest.raises(ValueError, match="exactly one decision question"):
            collator.prepare_example({"state": "s", "questions": THREE_QUESTIONS, "target_index": 0})

    def test_per_question_prompt_template_is_rejected(self, tokenizer):
        collator = self.make_collator(tokenizer, per_question=True)
        with pytest.raises(ValueError, match="one prompt"):
            collator.prepare_request({"state": "s", "questions": THREE_QUESTIONS, "targets": THREE_TARGETS})


def naive_calibration_metrics(rows, bins=15, kl_weight=0.25, brier_weight=0.25, gold_labels=None):
    """
    Every metric by an explicit float64 loop over questions given as (probabilities, target) lists. A question is right
    when its most probable option is its gold label: `gold_labels[i]`, else the target's first most probable.
    """
    accuracy = kl = brier = nll = confidence_total = 0.0
    confidences, corrects = [], []
    for index, (probabilities, target) in enumerate(rows):
        probabilities = [float(p) for p in probabilities]
        target = [float(t) for t in target]
        predicted = max(range(len(probabilities)), key=lambda i: probabilities[i])
        gold = max(range(len(target)), key=lambda i: target[i]) if gold_labels is None else gold_labels[index]
        confidence = probabilities[predicted]
        correct = float(predicted == gold)
        accuracy += correct
        confidence_total += confidence
        confidences.append(confidence)
        corrects.append(correct)
        for p, t in zip(probabilities, target, strict=True):
            brier += (p - t) ** 2
            if t > 0:
                nll -= t * np.log(p)
                kl += t * (np.log(t) - np.log(p))
    n = len(rows)
    ece = 0.0
    for b in range(bins):
        members = [i for i, c in enumerate(confidences) if min(int(c * bins), bins - 1) == b]
        ece += abs(sum(corrects[i] for i in members) - sum(confidences[i] for i in members))
    metrics = {
        "accuracy": accuracy / n,
        "kl": kl / n,
        "brier": brier / n,
        "ece": ece / n,
        "mean_confidence": confidence_total / n,
        "soft_nll": nll / n,
    }
    metrics["confidence_gap"] = metrics["mean_confidence"] - metrics["accuracy"]
    metrics["selection_score"] = metrics["accuracy"] - kl_weight * metrics["kl"] - brier_weight * metrics["brier"]
    return metrics


def ragged_questions(seed=0, count=40, max_options=4, soft=True):
    """Questions of 1 to `max_options` options: their probabilities and target distributions."""
    generator = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        width = 1 + index % max_options
        logits = generator.normal(size=width) * 2
        probabilities = np.exp(logits) / np.exp(logits).sum()
        if soft and index % 3 == 0:
            target = generator.dirichlet(np.ones(width))
        else:
            target = np.eye(width)[generator.integers(width)]
        rows.append((probabilities, target))
    return rows


def padded(rows, max_options=4):
    predictions = np.full((len(rows), max_options), -100.0)
    labels = np.full((len(rows), max_options), -100.0)
    for i, (probabilities, target) in enumerate(rows):
        predictions[i, : len(probabilities)] = probabilities
        labels[i, : len(target)] = target
    return predictions, labels


class EvalPrediction:
    def __init__(self, predictions, label_ids):
        self.predictions, self.label_ids = predictions, label_ids


class TestCalibrationMetrics:
    def test_calibration_metrics_match_naive(self):
        rows = ragged_questions()
        metrics = compute_calibration_metrics(EvalPrediction(*padded(rows)))
        expected = naive_calibration_metrics(rows)
        assert set(metrics) == set(expected)
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, abs=1e-9), key
        weighted = compute_calibration_metrics(EvalPrediction(*padded(rows)), bins=7, kl_weight=0.5, brier_weight=1.0)
        for key, value in naive_calibration_metrics(rows, bins=7, kl_weight=0.5, brier_weight=1.0).items():
            assert weighted[key] == pytest.approx(value, abs=1e-9), key

    def test_calibration_metrics_integer_labels(self):
        rows = ragged_questions(soft=False)
        predictions, labels = padded(rows)
        indices = np.array([int(np.argmax(target)) for _, target in rows])
        from_indices = compute_calibration_metrics(EvalPrediction(predictions, indices))
        from_distributions = compute_calibration_metrics(EvalPrediction(predictions, labels))
        expected = naive_calibration_metrics(rows)
        for key, value in expected.items():
            assert from_indices[key] == pytest.approx(value, abs=1e-9), key
            assert from_distributions[key] == pytest.approx(value, abs=1e-9), key
        # An index of -100 is a question to ignore
        ignored = compute_calibration_metrics(EvalPrediction(predictions, np.concatenate([indices[:-1], [-100]])))
        for key, value in naive_calibration_metrics(rows[:-1]).items():
            assert ignored[key] == pytest.approx(value, abs=1e-9), key
        with pytest.raises(ValueError, match="option"):
            compute_calibration_metrics(EvalPrediction(predictions[:1], np.array([3])))  # the first has one option

    def test_calibration_metrics_single_option_question(self):
        rows = [(np.array([1.0]), np.array([1.0]))] + ragged_questions(count=5)
        metrics = compute_calibration_metrics(EvalPrediction(*padded(rows)))
        for key, value in naive_calibration_metrics(rows).items():
            assert metrics[key] == pytest.approx(value, abs=1e-9), key
        only = compute_calibration_metrics(EvalPrediction(*padded(rows[:1])))
        assert only["accuracy"] == 1.0 and only["mean_confidence"] == 1.0
        assert only["kl"] == pytest.approx(0.0, abs=1e-12) and only["brier"] == 0.0 and only["ece"] == 0.0

    def test_calibration_metrics_reject_nan(self):
        predictions, labels = padded(ragged_questions(count=6))
        predictions[3, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            compute_calibration_metrics(EvalPrediction(predictions, labels))
        predictions, labels = padded(ragged_questions(count=6))
        predictions[3, 1] = np.inf  # the fourth question has four options
        with pytest.raises(ValueError, match="finite"):
            compute_calibration_metrics(EvalPrediction(predictions, labels))

    def test_calibration_metrics_packed_records(self):
        from trl.experimental.decision.decision_trainer import pack_rows

        rows = ragged_questions(count=12)
        predictions, labels = padded(rows)
        rows_per_example = torch.tensor([3, 1, 2, 1, 3, 2])
        expected = naive_calibration_metrics(rows)

        def pack(array):
            return pack_rows(torch.from_numpy(array), rows_per_example, 4, -100).numpy()

        packed = compute_calibration_metrics(EvalPrediction(pack(predictions), pack(labels)))
        # Flattened back to rows, as `DecisionTrainer` passes them, the padded slots stay
        flat = compute_calibration_metrics(
            EvalPrediction(pack(predictions).reshape(-1, 4), pack(labels).reshape(-1, 4))
        )
        indices = np.array([int(np.argmax(target)) for _, target in rows])
        hard = [(probabilities, np.eye(len(target))[np.argmax(target)]) for probabilities, target in rows]
        from_indices = compute_calibration_metrics(EvalPrediction(pack(predictions), pack(indices)))
        for key, value in expected.items():
            assert packed[key] == pytest.approx(value, abs=1e-9), key
            assert flat[key] == pytest.approx(value, abs=1e-9), key
        for key, value in naive_calibration_metrics(hard).items():
            assert from_indices[key] == pytest.approx(value, abs=1e-9), key

    def test_gold_label_decides_a_tie(self):
        # Ties in the target: the gold label is the second most probable option, which the model predicts
        rows = [
            (np.array([0.2, 0.7, 0.1]), np.array([0.5, 0.5, 0.0])),
            (np.array([0.6, 0.4]), np.array([0.5, 0.5])),
            (np.array([0.1, 0.3, 0.6]), np.array([0.0, 0.5, 0.5])),
        ] + ragged_questions(count=9, max_options=3)
        gold = [1, 0, 2] + [int(np.argmax(target)) for _, target in rows[3:]]
        predictions, labels = padded(rows, 3)
        metrics = compute_calibration_metrics(EvalPrediction(predictions, (labels, np.array(gold))))
        expected = naive_calibration_metrics(rows, gold_labels=gold)
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, abs=1e-9), key
        # Against the first most probable option, the first and third questions would count as wrong
        first_argmax = compute_calibration_metrics(EvalPrediction(predictions, labels))
        assert metrics["accuracy"] - first_argmax["accuracy"] == pytest.approx(2 / len(rows))
        assert metrics["selection_score"] > first_argmax["selection_score"]
        assert metrics["ece"] != first_argmax["ece"]
        # Record-shaped, as the trainer passes them: padded slots are -100 in both
        from trl.experimental.decision.decision_trainer import pack_rows

        rows_per_example = torch.tensor([3, 1, 2, 2, 3, 1])
        packed = (
            pack_rows(torch.from_numpy(predictions), rows_per_example, 3, -100).flatten(0, 1).numpy(),
            (
                pack_rows(torch.from_numpy(labels), rows_per_example, 3, -100).flatten(0, 1).numpy(),
                pack_rows(torch.tensor(gold), rows_per_example, 3, -100).flatten(0, 1).numpy(),
            ),
        )
        for key, value in compute_calibration_metrics(EvalPrediction(*packed)).items():
            assert value == pytest.approx(metrics[key], abs=1e-12), key
        # A gold label that is not an option of its question, and indices without distributions, are refused
        outside = list(gold)
        outside[1] = 2  # the second question has two options
        with pytest.raises(ValueError, match="gold label"):
            compute_calibration_metrics(EvalPrediction(predictions, (labels, np.array(outside))))
        with pytest.raises(ValueError, match="distributions"):
            compute_calibration_metrics(EvalPrediction(predictions, (np.array(gold), np.array(gold))))

    def test_selection_score_weights(self):
        rows = ragged_questions()
        metrics = compute_calibration_metrics(EvalPrediction(*padded(rows)), kl_weight=0, brier_weight=0)
        assert metrics["selection_score"] == metrics["accuracy"]
        default = compute_calibration_metrics(EvalPrediction(*padded(rows)))
        assert default["selection_score"] == pytest.approx(
            default["accuracy"] - 0.25 * default["kl"] - 0.25 * default["brier"], abs=1e-12
        )


class TestBestCheckpointRestore(TrlTestCase):
    """The strict checkpoint loader of `DecisionTrainer` also restores models that are not Clef."""

    def train_ranked(self, tokenizer, classification_dataset, model, **kwargs):
        # The first evaluation ranks best, so the best checkpoint is the first one, not the last
        evaluations = []

        def compute_metrics(prediction):
            evaluations.append(None)
            return {"rank": -len(evaluations)}

        trainer = DecisionTrainer(
            model=model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=6,
                learning_rate=1e-2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_rank",
                greater_is_better=True,
                report_to="none",
            ),
            train_dataset=classification_dataset,
            eval_dataset=classification_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            **kwargs,
        )
        trainer.train()
        assert trainer.state.best_model_checkpoint.endswith("checkpoint-2")
        return trainer

    @staticmethod
    def assert_restored(model, checkpoint, weights="model.safetensors"):
        from safetensors.torch import load_file

        saved = load_file(os.path.join(checkpoint, weights))
        # PEFT names the adapter `default` in the model, not in its saved file
        state = {
            name.replace(".modules_to_save.default", "").replace(".default", ""): value
            for name, value in model.state_dict().items()
            if "original_module" not in name
        }
        for name, value in saved.items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        assert saved

    def test_plain_model(self, tokenizer, classification_dataset, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="trl.experimental.decision.decision_trainer")
        trainer = self.train_ranked(tokenizer, classification_dataset, make_classification_model())
        # The line the base implementation prints
        assert f"Loading best model from {trainer.state.best_model_checkpoint} (score: -1" in caplog.text
        self.assert_restored(trainer.model, trainer.state.best_model_checkpoint)

    def test_model_with_keys_ignored_on_save(self, tokenizer, classification_dataset):
        model = make_classification_model()
        model._keys_to_ignore_on_save = {"score.weight"}
        trainer = self.train_ranked(tokenizer, classification_dataset, model)
        checkpoint = trainer.state.best_model_checkpoint
        from safetensors.torch import load_file

        assert "score.weight" not in load_file(os.path.join(checkpoint, "model.safetensors"))
        self.assert_restored(trainer.model, checkpoint)

    @require_peft
    def test_peft_model(self, tokenizer, classification_dataset):
        from peft import LoraConfig

        peft_config = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")
        trainer = self.train_ranked(
            tokenizer, classification_dataset, make_classification_model(), peft_config=peft_config
        )
        self.assert_restored(trainer.model, trainer.state.best_model_checkpoint, "adapter_model.safetensors")
