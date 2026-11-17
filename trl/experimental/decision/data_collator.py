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
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.decision import DecisionConfig as TransformersDecisionConfig
from transformers.decision import Question


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def normalize_target(row: Mapping[str, Any], question: Question) -> tuple[float, ...]:
    kind = question.type
    k = len(question.options)
    metadata = asdict(question)
    allow_null = metadata.get("allow_null", False)
    supplied = [
        name
        for name in ("target_index", "target_probs", "p_true", "score")
        if name in row and (row[name] is not None or name == "target_index")
    ]
    if len(supplied) > 1 and row.get("target_index") is None:
        supplied.remove("target_index")
    if len(supplied) != 1:
        raise ValueError("provide exactly one of target_index, target_probs, p_true, or score")
    label = supplied[0]
    values, positive_index = None, None
    if kind == "noul":
        positive_index = question.options.index("true")
    if kind == "score":
        values = tuple(metadata.get("values") or range(k))
    probs = [0.0] * (k + 1)
    if label == "target_probs":
        raw_probs = row[label]
        if not isinstance(raw_probs, (list, tuple)) or len(raw_probs) != k + 1:
            raise ValueError("target_probs must contain null probability followed by one per option")
        probs = [_finite(value, "target_probs") for value in raw_probs]
        if min(probs) < 0 or not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("target_probs must be nonnegative and sum to one")
        total = sum(probs)
        probs = [value / total for value in probs]
    elif label == "target_index":
        index = row[label]
        if index is None:
            if not allow_null:
                raise ValueError("a null target requires allow_null=true")
            probs[0] = 1.0
        elif type(index) is not int or not 0 <= index < k:
            raise ValueError("target_index must be a zero-based option index or null")
        else:
            probs[index + 1] = 1.0
    elif label == "p_true":
        if kind != "noul":
            raise ValueError("p_true is only valid for noul")
        probability = _finite(row[label], label)
        if not 0 <= probability <= 1:
            raise ValueError("p_true must be between zero and one")
        probs[positive_index + 1] = probability
        probs[2 - positive_index] = 1.0 - probability
    else:
        if kind != "score":
            raise ValueError("score is only valid for score records")
        score = _finite(row[label], label)
        if not values[0] <= score <= values[-1]:
            raise ValueError("score must fall within the supplied rubric values")
        for index, value in enumerate(values):
            if score <= value:
                if index == 0:
                    probs[1] = 1.0
                else:
                    fraction = (score - values[index - 1]) / (value - values[index - 1])
                    probs[index] = 1.0 - fraction
                    probs[index + 1] = fraction
                break
    if not allow_null and probs[0] != 0:
        raise ValueError("null target mass requires allow_null=true")
    return tuple(probs)


def _gold_key(label: Any, question: Question) -> str | None:
    """The option of `question` that a gold label names, or `None` if it names none."""
    if question.type == "noul":
        if isinstance(label, bool):
            label = "true" if label else "false"
        label = label.strip().lower() if isinstance(label, str) else None
    elif question.type == "score":
        if isinstance(label, str) and label.strip().lstrip("+-").isdigit():
            label = int(label)
        label = str(label) if type(label) is int else None
    elif not isinstance(label, str):
        label = None
    return label if label in question.options else None


def target_from_gold(gold: Any, question: Question) -> dict[str, list[float]]:
    """Turn the gold answer of one question into a `{"target_probs": [...]}` row, null probability first (zero).

    `gold` is a bare label, or a dict with `label` and/or `probabilities` (for a noul question, `noul` is also accepted
    as the probability of true). Probabilities take each option's value (missing or negative counts as zero), fill the
    missing side of a noul as `1 - p`, and are normalized to sum to one. Without probabilities, a label that names an
    option gives a one-hot target. Labels are normalized per type: noul takes booleans and "true"/"false", score takes
    an integer level or its string. Probabilities follow `question.options`.
    """
    options = list(question.options)
    row = gold if isinstance(gold, Mapping) else {"label": gold}
    probabilities = row.get("probabilities")
    if probabilities is None and question.type == "noul" and row.get("noul") is not None:
        probabilities = {"true": row["noul"]}
    if probabilities is not None:
        if not isinstance(probabilities, Mapping):
            raise ValueError("probabilities must map options to numbers")
        supplied = {}
        for key, value in probabilities.items():
            value = _finite(value, f"probability of {key!r}")
            option = _gold_key(key, question)
            if option is not None:
                supplied[option] = max(value, 0.0)
        if question.type == "noul" and len(supplied) == 1:
            (known,) = supplied
            other = "false" if known == "true" else "true"
            supplied[other] = max(1.0 - supplied[known], 0.0)
        total = sum(supplied.values())
        if total > 0:
            return {"target_probs": [0.0] + [supplied.get(option, 0.0) / total for option in options]}
    option = _gold_key(row.get("label"), question)
    if option is None:
        raise ValueError("gold has no usable label or probabilities")
    return {"target_probs": [0.0] + [1.0 if key == option else 0.0 for key in options]}


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


@dataclass
class DataCollatorForDecision(DataCollatorMixin):
    """Base for example collators that adapt Transformers decision inputs to a model's forward signature.

    Subclasses implement `tokenize` and `torch_call` for their model's input layout. Request rendering and question
    parsing are delegated to Transformers; targets are normalized independently of the model's loss.

    Args:
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer or processor used to render the decision request.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
    """

    tokenizer: PreTrainedTokenizerBase
    decision_config: TransformersDecisionConfig
    return_tensors = "pt"

    def prepare_example(self, example: dict[str, Any]):
        """Return native inputs, the question, and its normalized null-first target distribution."""
        request = (
            example["request"]
            if "request" in example
            else {"state": example["state"], "questions": example["questions"]}
        )
        prompts = self.decision_config.prepare_inputs(request, self.tokenizer)
        if len(prompts) != 1 or len(prompts[0][1]) != 1:
            raise ValueError("Each training row must contain exactly one decision question")
        inputs, questions = prompts[0]
        question = questions[0]
        return inputs, question, normalize_target(example, question)

    def prepare_request(self, example: dict[str, Any]):
        """Return native inputs of the single prompt, its questions in template order, and one null-first target each.

        The template must render all questions in one prompt. Targets come from `example["targets"]`, a mapping of
        question name to a row for `normalize_target`, or from `example["gold"]` / `example["answers"]`, a mapping of
        question name to a gold answer for [`~experimental.decision.data_collator.target_from_gold`]. Either may be a
        JSON string.
        """
        if "request" in example:
            request = example["request"]
        else:
            request = {"state": example["state"], "questions": _json_value(example["questions"])}
        prompts = self.decision_config.prepare_inputs(request, self.tokenizer)
        if len(prompts) != 1:
            raise ValueError(
                f"The template renders {len(prompts)} prompts for this row; prepare_request needs a template that "
                "renders all questions in one prompt"
            )
        inputs, questions = prompts[0]
        names = [question.name for question in questions]

        if example.get("targets") is not None:
            rows, convert = _json_value(example["targets"]), lambda row, question: row
            source = "targets"
        else:
            source = next((key for key in ("gold", "answers") if example.get(key) is not None), None)
            if source is None:
                raise ValueError("A row needs `targets`, `gold` or `answers` with one entry per question")
            rows, convert = _json_value(example[source]), target_from_gold
        if not isinstance(rows, Mapping):
            raise ValueError(f"`{source}` must map question names to targets")
        unknown = [name for name in rows if name not in names]
        if unknown:
            raise ValueError(f"`{source}` names unknown questions {unknown}; the questions are {names}")

        targets = []
        for question in questions:
            if question.name not in rows:
                raise ValueError(f"`{source}` has no entry for question {question.name!r}")
            try:
                targets.append(normalize_target(convert(rows[question.name], question), question))
            except ValueError as error:
                raise ValueError(f"Question {question.name!r}: {error}") from error
        return inputs, questions, targets
