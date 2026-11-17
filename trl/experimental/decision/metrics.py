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


import numpy as np


def compute_calibration_metrics(
    eval_pred, *, bins: int = 15, kl_weight: float = 0.25, brier_weight: float = 0.25
) -> dict[str, float]:
    """
    Accuracy and calibration of decisions, usable as the `compute_metrics` of
    [`experimental.decision.DecisionTrainer`].

    Every real question counts once, whatever its number of options. `eval_pred.predictions` holds the predicted
    probabilities of each question, `-100` in the cells of options it does not have, as `(num_questions, num_options)`
    or packed by record as `(num_records, max_questions, num_options)`. A question slot that is `-100` in every cell is
    padding and is ignored: with `rows_per_example`, the trainer passes the slots of each record, flattened to rows.
    `eval_pred.label_ids` holds, in the same layout, the target distribution of each question (`-100` in padded cells),
    or the index of its right option (`-100` for a question to ignore), or a pair `(distributions, indices)` of both,
    as [`experimental.decision.DecisionTrainer`] passes them when the collator gives `hard_labels`. Correctness
    (`accuracy`, and the metrics derived from it) is counted against the index when there is one, and otherwise against
    the most probable option of the target (the first of a tie); the other metrics compare the distributions.

    Args:
        eval_pred ([`~transformers.EvalPrediction`]):
            Predicted probabilities and labels.
        bins (`int`, *optional*, defaults to `15`):
            Number of equal-width confidence bins of the expected calibration error.
        kl_weight (`float`, *optional*, defaults to `0.25`):
            Weight of the KL divergence in `selection_score`.
        brier_weight (`float`, *optional*, defaults to `0.25`):
            Weight of the Brier score in `selection_score`.

    Returns:
        `dict[str, float]`: The means over the questions of:
        - `accuracy`: whether the most probable option is the gold label (see above).
        - `kl`: KL divergence from the target distribution to the predicted one.
        - `brier`: squared distance between the predicted and target distributions.
        - `ece`: expected calibration error of the confidence (the largest probability), over `bins` bins.
        - `mean_confidence`: confidence.
        - `soft_nll`: cross-entropy of the predicted distribution against the target distribution.
        - `confidence_gap`: `mean_confidence - accuracy`.
        - `selection_score`: `accuracy - kl_weight * kl - brier_weight * brier`, higher is better.

    Raises:
        `ValueError`: When a real cell holds a non-finite probability or label, or there is no question.
    """
    predictions = np.asarray(eval_pred.predictions, dtype=np.float64)
    labels, hard_labels = eval_pred.label_ids, None
    if isinstance(labels, (tuple, list)):
        labels, hard_labels = labels
        hard_labels = np.asarray(hard_labels).reshape(-1)
    labels = np.asarray(labels)
    num_options = predictions.shape[-1]
    # Distributions have the shape of the predictions; option indices have one dimension less
    is_distribution = labels.shape == predictions.shape
    if hard_labels is not None and not is_distribution:
        raise ValueError("Gold label indices go with target distributions, not with label indices")
    predictions = predictions.reshape(-1, num_options)
    is_option = predictions != -100
    if is_distribution:
        labels = labels.astype(np.float64).reshape(-1, num_options)
        keep = is_option.any(-1)
    else:
        labels = labels.reshape(-1)
        keep = is_option.any(-1) & (labels != -100)
    if hard_labels is not None and hard_labels.shape != keep.shape:
        raise ValueError("There must be one gold label index per question slot")
    predictions, labels, is_option = predictions[keep], labels[keep], is_option[keep]
    if hard_labels is not None:
        hard_labels = hard_labels[keep].astype(np.int64)
        valid = (hard_labels >= 0) & (hard_labels < num_options)
        if not valid.all() or not is_option[np.arange(len(hard_labels)), hard_labels].all():
            raise ValueError("A gold label is not an option of its question")
    if len(predictions) == 0:
        raise ValueError("There is no question to evaluate")
    if not np.isfinite(predictions[is_option]).all():
        raise ValueError("Predicted probabilities must be finite")
    if labels.ndim == 1:
        labels = labels.astype(np.int64)
        if ((labels < 0) | (labels >= num_options)).any() or not is_option[np.arange(len(labels)), labels].all():
            raise ValueError("A label is not an option of its question")
        targets = np.eye(num_options)[labels]
    else:
        if not np.isfinite(labels[is_option]).all():
            raise ValueError("Target probabilities must be finite")
        targets = labels
    probabilities = np.where(is_option, predictions, 0.0)
    targets = np.where(is_option, targets, 0.0)

    predicted = probabilities.argmax(-1)
    confidence = probabilities.max(-1)
    correct = predicted == (targets.argmax(-1) if hard_labels is None else hard_labels)
    log_probabilities = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
    soft_nll = -np.where(targets > 0, targets * log_probabilities, 0.0).sum(-1).mean()
    entropy = -np.where(targets > 0, targets * np.log(np.where(targets > 0, targets, 1.0)), 0.0).sum(-1).mean()
    kl = soft_nll - entropy
    brier = ((probabilities - targets) ** 2).sum(-1).mean()
    bucket = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    ece = np.abs(
        np.bincount(bucket, weights=correct, minlength=bins) - np.bincount(bucket, weights=confidence, minlength=bins)
    ).sum() / len(confidence)
    accuracy = correct.mean()
    mean_confidence = confidence.mean()
    return {
        "accuracy": float(accuracy),
        "kl": float(kl),
        "brier": float(brier),
        "ece": float(ece),
        "mean_confidence": float(mean_confidence),
        "soft_nll": float(soft_nll),
        "confidence_gap": float(mean_confidence - accuracy),
        "selection_score": float(accuracy - kl_weight * kl - brier_weight * brier),
    }
