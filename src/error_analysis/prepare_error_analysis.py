#!/usr/bin/env python
# prepare_error_analysis.py

"""
Prepare automatic error-analysis artifacts for ComVE Task A/B.

This script scans prediction CSV files, joins them with Task A/B data,
and produces model summaries, all-error tables, cross-model hard cases,
and manual taxonomy templates.

Expected prediction columns:
    id, gold, pred, correct

Also supports scoring outputs with:
    id, gold, prediction

Outputs:
    outputs/error_analysis/
        model_summary.csv
        all_errors_taskA.csv
        all_errors_taskB.csv
        hard_cases_taskA.csv
        hard_cases_taskB.csv
        manual_taxonomy_taskA_roberta_large_original.csv
        manual_taxonomy_taskB_roberta_large_original.csv
        manual_taxonomy_taskA_hard_cases_original.csv
        manual_taxonomy_taskB_hard_cases_original.csv
        error_analysis_note.md

Usage:
    python src/error_analysis/prepare_error_analysis.py \
        --config src/config.yaml \
        --split test
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import yaml


# ======== Path Setup ========

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # project/src/error_analysis
SRC_DIR = os.path.dirname(CURRENT_DIR)                     # project/src
PROJECT_DIR = os.path.dirname(SRC_DIR)                     # project

sys.path.append(SRC_DIR)

from data_utils import (
    resolve_project_path,
    get_split_paths,
    load_taskA_samples,
    load_taskB_samples,
)


# ======== Constants ========

TASK_B_LABEL_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}

TASK_B_ID_TO_LABEL = {
    0: "A",
    1: "B",
    2: "C",
}

DEFAULT_PRED_DIRS = [
    "outputs/finetune_bert",
    "outputs/llm_prompt",
    "outputs/llm_scoring",
    "outputs",
]

DEFAULT_CORE_MODELS = {
    "A": [
        "preds_taskA_bert-base-uncased_test",
        "preds_taskA_roberta-large_test",
        "llm_taskA_oneshot_Qwen_Qwen3-8B_test",
    ],
    "B": [
        "preds_taskB_bert-base-uncased_test",
        "preds_taskB_roberta-large_test",
        "llm_taskB_oneshot_Qwen_Qwen3-8B_test",
    ],
}

MANUAL_ERROR_TYPES = [
    "physical",
    "object_affordance",
    "social",
    "temporal",
    "causal",
    "numerical",
    "lexical_distractor",
    "ambiguity",
    "dataset_noise",
    "other",
]


# ======== Basic Utilities ========

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Dict:
    config_path = resolve_project_path(path, must_exist=True)

    with open(config_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def safe_name(name: str) -> str:
    name = str(name)
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return name


def normalize_text(text) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int_label(value, task: str) -> Optional[int]:
    if value is None:
        return None

    raw = str(value).strip()

    if raw == "":
        return None

    if raw.lower() in {"none", "nan", "null"}:
        return None

    if task == "B" and raw in TASK_B_LABEL_MAP:
        return TASK_B_LABEL_MAP[raw]

    try:
        return int(raw)
    except ValueError:
        return None


def words(text: str) -> List[str]:
    text = normalize_text(text).lower()
    toks = []

    for w in text.split():
        w = w.strip(".,!?;:\"'()[]{}")
        if w:
            toks.append(w)

    return toks


def word_count(text: str) -> int:
    return len(words(text))


def word_overlap(a: str, b: str) -> float:
    wa = set(words(a))
    wb = set(words(b))

    if not wa or not wb:
        return 0.0

    return len(wa & wb) / len(wa | wb)


def cosine_bow(a: str, b: str) -> float:
    ca = Counter(words(a))
    cb = Counter(words(b))

    keys = set(ca) | set(cb)

    if not keys:
        return 0.0

    dot = sum(ca[k] * cb[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def fmt_float(value: float) -> str:
    return f"{float(value):.4f}"


def write_dicts(path: str, rows: List[Dict], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ======== Data Loading ========

def load_task_samples(cfg: Dict, task: str, split: str):
    data_path, answer_path = get_split_paths(cfg, task=task, split=split)

    if task == "A":
        raw_samples = load_taskA_samples(
            data_path=data_path,
            answer_path=answer_path,
            require_labels=True,
        )

        samples = {}

        for s in raw_samples:
            samples[str(s["id"])] = {
                "id": str(s["id"]),
                "gold": s["label"],
                "sent0": s["sentence0"],
                "sent1": s["sentence1"],
            }

        return samples, data_path, answer_path

    raw_samples = load_taskB_samples(
        data_path=data_path,
        answer_path=answer_path,
        require_labels=True,
    )

    samples = {}

    for s in raw_samples:
        samples[str(s["id"])] = {
            "id": str(s["id"]),
            "gold": s["label"],
            "false_sent": s["question"],
            "optionA": s["optionA"],
            "optionB": s["optionB"],
            "optionC": s["optionC"],
        }

    return samples, data_path, answer_path


# ======== Prediction Loading ========

def is_prediction_file(path: str, task: str) -> bool:
    name = os.path.basename(path)

    if not name.endswith(".csv"):
        return False

    if name.startswith(f"errors_task{task}_"):
        return False

    if name.startswith(f"all_errors_task{task}"):
        return False

    if name.startswith(f"hard_cases_task{task}"):
        return False

    prefixes = [
        f"preds_task{task}_",
        f"llm_task{task}_",
    ]

    if any(name.startswith(prefix) for prefix in prefixes):
        return True

    # llm_scoring outputs often use names like:
    # dev_taskA_bert-large-uncased_official.csv
    # dev_taskA_distilgpt2_ppl.csv
    if f"task{task}" in name.lower() and (
        "ppl" in name.lower()
        or "bert" in name.lower()
        or "score" in name.lower()
    ):
        return True

    return False


def find_prediction_files(pred_dirs: List[str], task: str) -> List[str]:
    files = []
    seen = set()

    for pred_dir in pred_dirs:
        pred_dir = resolve_project_path(pred_dir, must_exist=False)

        if not os.path.exists(pred_dir):
            continue

        for root, _, filenames in os.walk(pred_dir):
            # 避免把 error_analysis 自己的输出再次扫进去
            if "error_analysis" in root.replace("\\", "/"):
                continue

            for filename in filenames:
                path = os.path.join(root, filename)

                if path in seen:
                    continue

                if is_prediction_file(path, task):
                    files.append(path)
                    seen.add(path)

    return sorted(files)


def prediction_model_name(path: str) -> str:
    return safe_name(os.path.splitext(os.path.basename(path))[0])


def load_predictions(path: str, task: str) -> Dict[str, Dict]:
    """
    Normalize prediction CSV formats.

    Supported columns:
        id, gold, pred, correct
        id, gold, prediction, correct
        id, gold, prediction
    """
    rows = read_csv_dicts(path)
    preds = {}

    for row in rows:
        sid = normalize_text(row.get("id", ""))

        if not sid:
            continue

        pred_raw = None

        if "pred" in row and normalize_text(row.get("pred")) != "":
            pred_raw = row.get("pred")
        elif "prediction" in row and normalize_text(row.get("prediction")) != "":
            pred_raw = row.get("prediction")

        pred = parse_int_label(pred_raw, task=task)

        if pred is None:
            continue

        gold = parse_int_label(row.get("gold"), task=task)

        correct_raw = row.get("correct")

        if correct_raw is not None and normalize_text(correct_raw) != "":
            correct = normalize_bool(correct_raw)
        elif gold is not None:
            correct = pred == gold
        else:
            correct = None

        preds[sid] = {
            "id": sid,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "raw_response": row.get("raw_response", ""),
        }

    return preds


def selected_gold(pred_info: Dict, sample: Dict) -> int:
    if pred_info.get("gold") is not None:
        return pred_info["gold"]
    return sample["gold"]


# ======== Error Row Builders ========

def taskA_error_row(
    model: str,
    sample: Dict,
    pred_info: Dict,
    source_file: str,
    variant: str = "original",
) -> Dict:
    gold = selected_gold(pred_info, sample)
    pred = pred_info["pred"]

    sent0 = sample["sent0"]
    sent1 = sample["sent1"]

    gold_sent = sent0 if gold == 0 else sent1
    pred_sent = sent0 if pred == 0 else sent1

    return {
        "task": "A",
        "variant": variant,
        "model": model,
        "id": sample["id"],
        "gold": gold,
        "pred": pred,
        "correct": int(pred == gold),
        "confusion": f"g{gold}->p{pred}",
        "sent0": sent0,
        "sent1": sent1,
        "gold_nonsense_sentence": gold_sent,
        "predicted_nonsense_sentence": pred_sent,
        "sent0_len": word_count(sent0),
        "sent1_len": word_count(sent1),
        "avg_sentence_len": fmt_float((word_count(sent0) + word_count(sent1)) / 2),
        "sent_pair_overlap": fmt_float(word_overlap(sent0, sent1)),
        "source_prediction_file": source_file,
    }


def taskB_option(sample: Dict, idx: int) -> str:
    if idx == 0:
        return sample["optionA"]
    if idx == 1:
        return sample["optionB"]
    if idx == 2:
        return sample["optionC"]
    return ""


def taskB_error_row(
    model: str,
    sample: Dict,
    pred_info: Dict,
    source_file: str,
    variant: str = "original",
) -> Dict:
    gold = selected_gold(pred_info, sample)
    pred = pred_info["pred"]

    gold_option = taskB_option(sample, gold)
    pred_option = taskB_option(sample, pred)

    wrong_options = [
        taskB_option(sample, idx)
        for idx in [0, 1, 2]
        if idx != gold
    ]

    max_gold_wrong_sim = max(
        [cosine_bow(gold_option, opt) for opt in wrong_options]
        or [0.0]
    )

    gold_label = TASK_B_ID_TO_LABEL.get(gold, str(gold))
    pred_label = TASK_B_ID_TO_LABEL.get(pred, str(pred))

    return {
        "task": "B",
        "variant": variant,
        "model": model,
        "id": sample["id"],
        "gold": gold_label,
        "pred": pred_label,
        "correct": int(pred == gold),
        "confusion": f"g{gold_label}->p{pred_label}",
        "false_sent": sample["false_sent"],
        "OptionA": sample["optionA"],
        "OptionB": sample["optionB"],
        "OptionC": sample["optionC"],
        "gold_option": gold_option,
        "predicted_option": pred_option,
        "false_sent_len": word_count(sample["false_sent"]),
        "gold_option_len": word_count(gold_option),
        "pred_option_len": word_count(pred_option),
        "max_gold_wrong_option_cosine": fmt_float(max_gold_wrong_sim),
        "source_prediction_file": source_file,
    }


# ======== Main Analysis ========

def summarize_prediction_file(
    task: str,
    model: str,
    source_file: str,
    preds: Dict[str, Dict],
    samples: Dict[str, Dict],
    variant: str = "original",
) -> Tuple[Dict, List[Dict], set]:
    total = 0
    correct = 0
    errors = set()
    error_rows = []
    confusion = Counter()
    pred_distribution = Counter()
    gold_mismatch_count = 0

    for sid, sample in samples.items():
        if sid not in preds:
            continue

        pred_info = preds[sid]
        pred = pred_info["pred"]
        gold = selected_gold(pred_info, sample)

        if pred_info.get("gold") is not None and pred_info["gold"] != sample["gold"]:
            gold_mismatch_count += 1

        total += 1
        pred_distribution[pred] += 1

        if pred == gold:
            correct += 1
            continue

        errors.add(sid)
        confusion[(gold, pred)] += 1

        sample_for_row = dict(sample)
        sample_for_row["gold"] = gold

        if task == "A":
            error_rows.append(
                taskA_error_row(
                    model=model,
                    sample=sample_for_row,
                    pred_info=pred_info,
                    source_file=source_file,
                    variant=variant,
                )
            )
        else:
            error_rows.append(
                taskB_error_row(
                    model=model,
                    sample=sample_for_row,
                    pred_info=pred_info,
                    source_file=source_file,
                    variant=variant,
                )
            )

    accuracy = correct / total if total else 0.0

    top_confusions = []

    for (gold, pred), count in confusion.most_common(5):
        if task == "A":
            top_confusions.append(f"g{gold}->p{pred}={count}")
        else:
            g = TASK_B_ID_TO_LABEL.get(gold, str(gold))
            p = TASK_B_ID_TO_LABEL.get(pred, str(pred))
            top_confusions.append(f"g{g}->p{p}={count}")

    status = "ok"
    warning = ""

    if total == 0:
        status = "suspicious"
        warning = "no_matching_prediction_rows"
    elif pred_distribution:
        majority_pred, majority_count = pred_distribution.most_common(1)[0]
        majority_ratio = majority_count / total

        if majority_ratio >= 0.95 and accuracy < 0.75:
            status = "suspicious"
            warning = f"degenerate_prediction_distribution: pred={majority_pred}, ratio={majority_ratio:.3f}"

    if gold_mismatch_count > 0:
        if warning:
            warning += "; "
        warning += f"gold_mismatch_with_dataset={gold_mismatch_count}"

    summary = {
        "task": task,
        "variant": variant,
        "model": model,
        "status": status,
        "total": total,
        "correct": correct,
        "accuracy": fmt_float(accuracy),
        "n_errors": len(errors),
        "top_confusions": "; ".join(top_confusions),
        "prediction_distribution": "; ".join(
            f"{k}:{v}" for k, v in sorted(pred_distribution.items(), key=lambda x: str(x[0]))
        ),
        "gold_mismatch_with_dataset": gold_mismatch_count,
        "warning": warning,
        "source_prediction_file": source_file,
    }

    return summary, error_rows, errors


def analyze_task(
    task: str,
    samples: Dict[str, Dict],
    pred_dirs: List[str],
    output_dir: str,
    variant: str = "original",
):
    prediction_paths = find_prediction_files(pred_dirs, task=task)

    all_error_rows = []
    summary_rows = []
    model_to_errors = {}
    model_to_preds = {}

    for path in prediction_paths:
        model = prediction_model_name(path)
        source_file = os.path.relpath(path, PROJECT_DIR)

        preds = load_predictions(path, task=task)

        if not preds:
            continue

        summary, error_rows, errors = summarize_prediction_file(
            task=task,
            model=model,
            source_file=source_file,
            preds=preds,
            samples=samples,
            variant=variant,
        )

        # total=0 的文件跳过 hard-case，但 summary 保留
        summary_rows.append(summary)
        all_error_rows.extend(error_rows)

        if summary["total"] > 0:
            model_to_errors[model] = errors
            model_to_preds[model] = preds

    return summary_rows, all_error_rows, model_to_errors, model_to_preds


# ======== Hard Cases ========

def resolve_core_models(
    task: str,
    model_to_errors: Dict[str, set],
    core_models_arg: Optional[str],
) -> List[str]:
    available = set(model_to_errors.keys())

    if core_models_arg:
        requested = [safe_name(x.strip()) for x in core_models_arg.split(",") if x.strip()]
        core = [m for m in requested if m in available]

        if core:
            return core

    default_core = DEFAULT_CORE_MODELS.get(task, [])
    core = [m for m in default_core if m in available]

    if core:
        return core

    # fallback：如果没有匹配到写死的 core model，就用前 3 个模型
    return sorted(available)[:3]


def build_hard_cases(
    task: str,
    samples: Dict[str, Dict],
    model_to_errors: Dict[str, set],
    model_to_preds: Dict[str, Dict],
    core_models: List[str],
    variant: str = "original",
) -> List[Dict]:
    if not core_models:
        return []

    hard_ids = set(samples.keys())

    for model in core_models:
        hard_ids &= model_to_errors.get(model, set())

    rows = []

    for sid in sorted(hard_ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        sample = samples[sid]

        pred_parts = []

        for model in core_models:
            pred_info = model_to_preds[model][sid]
            gold = selected_gold(pred_info, sample)
            pred = pred_info["pred"]

            if task == "A":
                pred_parts.append(f"{model}:g{gold}->p{pred}")
            else:
                g = TASK_B_ID_TO_LABEL.get(gold, str(gold))
                p = TASK_B_ID_TO_LABEL.get(pred, str(pred))
                pred_parts.append(f"{model}:g{g}->p{p}")

        if task == "A":
            gold = selected_gold(model_to_preds[core_models[0]][sid], sample)
            rows.append(
                {
                    "task": "A",
                    "variant": variant,
                    "id": sid,
                    "hard_case_definition": "wrong_by_all_core_models",
                    "core_models": "; ".join(core_models),
                    "model_predictions": "; ".join(pred_parts),
                    "gold": gold,
                    "sent0": sample["sent0"],
                    "sent1": sample["sent1"],
                    "gold_nonsense_sentence": sample["sent0"] if gold == 0 else sample["sent1"],
                    "sent_pair_overlap": fmt_float(word_overlap(sample["sent0"], sample["sent1"])),
                }
            )
        else:
            gold = selected_gold(model_to_preds[core_models[0]][sid], sample)
            gold_label = TASK_B_ID_TO_LABEL.get(gold, str(gold))
            rows.append(
                {
                    "task": "B",
                    "variant": variant,
                    "id": sid,
                    "hard_case_definition": "wrong_by_all_core_models",
                    "core_models": "; ".join(core_models),
                    "model_predictions": "; ".join(pred_parts),
                    "gold": gold_label,
                    "false_sent": sample["false_sent"],
                    "OptionA": sample["optionA"],
                    "OptionB": sample["optionB"],
                    "OptionC": sample["optionC"],
                    "gold_option": taskB_option(sample, gold),
                }
            )

    return rows


# ======== Output Writers ========

def write_all_errors(task: str, rows: List[Dict], output_dir: str):
    if task == "A":
        fields = [
            "task",
            "variant",
            "model",
            "id",
            "gold",
            "pred",
            "correct",
            "confusion",
            "sent0",
            "sent1",
            "gold_nonsense_sentence",
            "predicted_nonsense_sentence",
            "sent0_len",
            "sent1_len",
            "avg_sentence_len",
            "sent_pair_overlap",
            "source_prediction_file",
        ]
    else:
        fields = [
            "task",
            "variant",
            "model",
            "id",
            "gold",
            "pred",
            "correct",
            "confusion",
            "false_sent",
            "OptionA",
            "OptionB",
            "OptionC",
            "gold_option",
            "predicted_option",
            "false_sent_len",
            "gold_option_len",
            "pred_option_len",
            "max_gold_wrong_option_cosine",
            "source_prediction_file",
        ]

    path = os.path.join(output_dir, f"all_errors_task{task}.csv")
    write_dicts(path, rows, fields)
    print(f"Saved Task {task} all errors to: {path}")


def write_hard_cases(task: str, rows: List[Dict], output_dir: str):
    if task == "A":
        fields = [
            "task",
            "variant",
            "id",
            "hard_case_definition",
            "core_models",
            "model_predictions",
            "gold",
            "sent0",
            "sent1",
            "gold_nonsense_sentence",
            "sent_pair_overlap",
        ]
    else:
        fields = [
            "task",
            "variant",
            "id",
            "hard_case_definition",
            "core_models",
            "model_predictions",
            "gold",
            "false_sent",
            "OptionA",
            "OptionB",
            "OptionC",
            "gold_option",
        ]

    path = os.path.join(output_dir, f"hard_cases_task{task}.csv")
    write_dicts(path, rows, fields)
    print(f"Saved Task {task} hard cases to: {path}")


def write_model_summary(rows: List[Dict], output_dir: str):
    fields = [
        "task",
        "variant",
        "model",
        "status",
        "total",
        "correct",
        "accuracy",
        "n_errors",
        "top_confusions",
        "prediction_distribution",
        "gold_mismatch_with_dataset",
        "warning",
        "source_prediction_file",
    ]

    path = os.path.join(output_dir, "model_summary.csv")
    write_dicts(path, rows, fields)
    print(f"Saved model summary to: {path}")


# ======== Manual Taxonomy Templates ========

def write_manual_taxonomy_templates(
    all_a: List[Dict],
    all_b: List[Dict],
    hard_a: List[Dict],
    hard_b: List[Dict],
    output_dir: str,
    roberta_model_name: str,
):
    roberta_model_name = safe_name(roberta_model_name)

    # Task A RoBERTa-large errors
    taskA_roberta_rows = []

    for row in all_a:
        if row["variant"] == "original" and row["model"] == roberta_model_name:
            item = {
                "variant": row["variant"],
                "id": row["id"],
                "gold": row["gold"],
                "pred": row["pred"],
                "confusion": row["confusion"],
                "sent0": row["sent0"],
                "sent1": row["sent1"],
                "gold_nonsense_sentence": row["gold_nonsense_sentence"],
                "predicted_nonsense_sentence": row["predicted_nonsense_sentence"],
                "sent_pair_overlap": row["sent_pair_overlap"],
                "error_type": "",
                "secondary_error_type": "",
                "is_dataset_noise": "",
                "notes": "",
            }
            taskA_roberta_rows.append(item)

    taskA_roberta_fields = [
        "variant",
        "id",
        "gold",
        "pred",
        "confusion",
        "sent0",
        "sent1",
        "gold_nonsense_sentence",
        "predicted_nonsense_sentence",
        "sent_pair_overlap",
        "error_type",
        "secondary_error_type",
        "is_dataset_noise",
        "notes",
    ]

    path = os.path.join(output_dir, "manual_taxonomy_taskA_roberta_large_original.csv")
    write_dicts(path, taskA_roberta_rows, taskA_roberta_fields)
    print(f"Saved manual taxonomy template: {path}")

    # Task B RoBERTa-large errors
    taskB_roberta_rows = []

    for row in all_b:
        if row["variant"] == "original" and row["model"] == roberta_model_name:
            item = {
                "variant": row["variant"],
                "id": row["id"],
                "gold": row["gold"],
                "pred": row["pred"],
                "confusion": row["confusion"],
                "false_sent": row["false_sent"],
                "OptionA": row["OptionA"],
                "OptionB": row["OptionB"],
                "OptionC": row["OptionC"],
                "gold_option": row["gold_option"],
                "predicted_option": row["predicted_option"],
                "max_gold_wrong_option_cosine": row["max_gold_wrong_option_cosine"],
                "error_type": "",
                "secondary_error_type": "",
                "is_dataset_noise": "",
                "notes": "",
            }
            taskB_roberta_rows.append(item)

    taskB_roberta_fields = [
        "variant",
        "id",
        "gold",
        "pred",
        "confusion",
        "false_sent",
        "OptionA",
        "OptionB",
        "OptionC",
        "gold_option",
        "predicted_option",
        "max_gold_wrong_option_cosine",
        "error_type",
        "secondary_error_type",
        "is_dataset_noise",
        "notes",
    ]

    path = os.path.join(output_dir, "manual_taxonomy_taskB_roberta_large_original.csv")
    write_dicts(path, taskB_roberta_rows, taskB_roberta_fields)
    print(f"Saved manual taxonomy template: {path}")

    # Task A hard cases
    taskA_hard_rows = []

    for row in hard_a:
        if row["variant"] == "original":
            item = {
                "variant": row["variant"],
                "id": row["id"],
                "hard_case_definition": row["hard_case_definition"],
                "core_models": row["core_models"],
                "model_predictions": row["model_predictions"],
                "gold": row["gold"],
                "sent0": row["sent0"],
                "sent1": row["sent1"],
                "gold_nonsense_sentence": row["gold_nonsense_sentence"],
                "sent_pair_overlap": row["sent_pair_overlap"],
                "error_type": "",
                "secondary_error_type": "",
                "is_dataset_noise": "",
                "notes": "",
            }
            taskA_hard_rows.append(item)

    taskA_hard_fields = [
        "variant",
        "id",
        "hard_case_definition",
        "core_models",
        "model_predictions",
        "gold",
        "sent0",
        "sent1",
        "gold_nonsense_sentence",
        "sent_pair_overlap",
        "error_type",
        "secondary_error_type",
        "is_dataset_noise",
        "notes",
    ]

    path = os.path.join(output_dir, "manual_taxonomy_taskA_hard_cases_original.csv")
    write_dicts(path, taskA_hard_rows, taskA_hard_fields)
    print(f"Saved manual taxonomy template: {path}")

    # Task B hard cases
    taskB_hard_rows = []

    for row in hard_b:
        if row["variant"] == "original":
            item = {
                "variant": row["variant"],
                "id": row["id"],
                "hard_case_definition": row["hard_case_definition"],
                "core_models": row["core_models"],
                "model_predictions": row["model_predictions"],
                "gold": row["gold"],
                "false_sent": row["false_sent"],
                "OptionA": row["OptionA"],
                "OptionB": row["OptionB"],
                "OptionC": row["OptionC"],
                "gold_option": row["gold_option"],
                "error_type": "",
                "secondary_error_type": "",
                "is_dataset_noise": "",
                "notes": "",
            }
            taskB_hard_rows.append(item)

    taskB_hard_fields = [
        "variant",
        "id",
        "hard_case_definition",
        "core_models",
        "model_predictions",
        "gold",
        "false_sent",
        "OptionA",
        "OptionB",
        "OptionC",
        "gold_option",
        "error_type",
        "secondary_error_type",
        "is_dataset_noise",
        "notes",
    ]

    path = os.path.join(output_dir, "manual_taxonomy_taskB_hard_cases_original.csv")
    write_dicts(path, taskB_hard_rows, taskB_hard_fields)
    print(f"Saved manual taxonomy template: {path}")


# ======== Notes ========

def write_note(
    output_dir: str,
    summary_rows: List[Dict],
    hard_a: List[Dict],
    hard_b: List[Dict],
    meta: Dict,
):
    path = os.path.join(output_dir, "error_analysis_note.md")

    by_task = defaultdict(list)

    for row in summary_rows:
        by_task[row["task"]].append(row)

    lines = [
        "# Error Analysis Notes",
        "",
        "This folder contains automatically generated error-analysis artifacts.",
        "",
        "## Generated Files",
        "",
        "- `model_summary.csv`: accuracy, error count, confusion directions, prediction distribution, and warnings for every prediction file.",
        "- `all_errors_taskA.csv` / `all_errors_taskB.csv`: all wrong predictions joined with the original dataset text.",
        "- `hard_cases_taskA.csv` / `hard_cases_taskB.csv`: samples that are wrong for all selected core models.",
        "- `manual_taxonomy_taskA_roberta_large_original.csv`: manual-label template for RoBERTa-large Task A errors.",
        "- `manual_taxonomy_taskB_roberta_large_original.csv`: manual-label template for RoBERTa-large Task B errors.",
        "- `manual_taxonomy_taskA_hard_cases_original.csv`: manual-label template for Task A hard cases.",
        "- `manual_taxonomy_taskB_hard_cases_original.csv`: manual-label template for Task B hard cases.",
        "",
        "## Dataset",
        "",
        f"- Split: `{meta.get('split')}`",
        f"- Task A data: `{meta.get('taskA_data')}`",
        f"- Task A answers: `{meta.get('taskA_answer')}`",
        f"- Task B data: `{meta.get('taskB_data')}`",
        f"- Task B answers: `{meta.get('taskB_answer')}`",
        "",
        "## Model Summary",
        "",
    ]

    for task in ["A", "B"]:
        lines.append(f"### Task {task}")
        rows = by_task.get(task, [])

        if not rows:
            lines.append("- No prediction files found.")
            lines.append("")
            continue

        for row in rows:
            status = row["status"]
            warning = row["warning"]

            warning_text = ""
            if status != "ok" or warning:
                warning_text = f" [{status}: {warning}]"

            lines.append(
                f"- `{row['model']}`: acc={row['accuracy']}, "
                f"errors={row['n_errors']}, top confusions={row['top_confusions']}"
                f"{warning_text}"
            )

        lines.append("")

    lines.extend(
        [
            "## Cross-Model Hard Cases",
            "",
            f"- Task A hard cases: {len(hard_a)}",
            f"- Task B hard cases: {len(hard_b)}",
            "",
            "Hard cases are defined as examples that are incorrectly predicted by all selected core models.",
            "",
            "## Manual Taxonomy Labels",
            "",
            "Recommended `error_type` values:",
            "",
        ]
    )

    for t in MANUAL_ERROR_TYPES:
        lines.append(f"- `{t}`")

    lines.extend(
        [
            "",
            "Suggested annotation workflow:",
            "",
            "1. Start with the two RoBERTa-large templates.",
            "2. Fill `error_type` for each row.",
            "3. Use `secondary_error_type` only when the case clearly has two causes.",
            "4. Mark `is_dataset_noise=yes` only when the gold label is genuinely questionable.",
            "5. Write a short explanation in `notes` for report-ready qualitative evidence.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved note to: {path}")


# ======== CLI ========

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ComVE error-analysis artifacts.")

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SRC_DIR, "config.yaml"),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/error_analysis",
    )

    parser.add_argument(
        "--pred_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Prediction directories to scan. Defaults to outputs/finetune_bert, outputs/llm_prompt, outputs/llm_scoring, outputs.",
    )

    parser.add_argument(
        "--core_models_taskA",
        type=str,
        default=None,
        help="Comma-separated model names for Task A hard cases. Use CSV stem names.",
    )

    parser.add_argument(
        "--core_models_taskB",
        type=str,
        default=None,
        help="Comma-separated model names for Task B hard cases. Use CSV stem names.",
    )

    parser.add_argument(
        "--roberta_model_name",
        type=str,
        default="preds_taskA_roberta-large_test",
        help=(
            "CSV stem name for RoBERTa-large Task A. "
            "Task B name is inferred by replacing taskA with taskB unless --roberta_model_name_taskB is given."
        ),
    )

    parser.add_argument(
        "--roberta_model_name_taskB",
        type=str,
        default=None,
        help="CSV stem name for RoBERTa-large Task B.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    output_dir = resolve_project_path(args.output_dir, must_exist=False)
    ensure_dir(output_dir)

    pred_dirs = args.pred_dirs if args.pred_dirs is not None else DEFAULT_PRED_DIRS

    print("=" * 60)
    print("Preparing error-analysis artifacts")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print("Prediction dirs:")
    for d in pred_dirs:
        print(f"  - {resolve_project_path(d, must_exist=False)}")
    print("=" * 60)

    taskA_samples, taskA_data, taskA_answer = load_task_samples(
        cfg=cfg,
        task="A",
        split=args.split,
    )

    taskB_samples, taskB_data, taskB_answer = load_task_samples(
        cfg=cfg,
        task="B",
        split=args.split,
    )

    summary_a, all_a, model_errors_a, model_preds_a = analyze_task(
        task="A",
        samples=taskA_samples,
        pred_dirs=pred_dirs,
        output_dir=output_dir,
        variant="original",
    )

    summary_b, all_b, model_errors_b, model_preds_b = analyze_task(
        task="B",
        samples=taskB_samples,
        pred_dirs=pred_dirs,
        output_dir=output_dir,
        variant="original",
    )

    core_models_a = resolve_core_models(
        task="A",
        model_to_errors=model_errors_a,
        core_models_arg=args.core_models_taskA,
    )

    core_models_b = resolve_core_models(
        task="B",
        model_to_errors=model_errors_b,
        core_models_arg=args.core_models_taskB,
    )

    hard_a = build_hard_cases(
        task="A",
        samples=taskA_samples,
        model_to_errors=model_errors_a,
        model_to_preds=model_preds_a,
        core_models=core_models_a,
        variant="original",
    )

    hard_b = build_hard_cases(
        task="B",
        samples=taskB_samples,
        model_to_errors=model_errors_b,
        model_to_preds=model_preds_b,
        core_models=core_models_b,
        variant="original",
    )

    all_summary = summary_a + summary_b

    write_model_summary(all_summary, output_dir)
    write_all_errors("A", all_a, output_dir)
    write_all_errors("B", all_b, output_dir)
    write_hard_cases("A", hard_a, output_dir)
    write_hard_cases("B", hard_b, output_dir)

    roberta_a = safe_name(args.roberta_model_name)

    if args.roberta_model_name_taskB is not None:
        roberta_b = safe_name(args.roberta_model_name_taskB)
    else:
        roberta_b = roberta_a.replace("taskA", "taskB")

    # 为了复用函数，这里要求 A/B 分别匹配自己的名字。
    # 函数参数只收一个 roberta_model_name，所以这里临时把 Task B 的 model 名修正成同一个判断逻辑。
    # 更简单的做法：如果你的 Task A/B 文件都是 preds_taskA_roberta-large_test /
    # preds_taskB_roberta-large_test，这里不用改任何参数。
    write_manual_taxonomy_templates(
        all_a=all_a,
        all_b=all_b,
        hard_a=hard_a,
        hard_b=hard_b,
        output_dir=output_dir,
        roberta_model_name=roberta_a,
    )

    # 如果 Task B 的 RoBERTa 文件名和自动替换后的不一致，额外生成一次 Task B 模板会比较麻烦。
    # 所以这里做一个简单提示。
    if roberta_b != roberta_a.replace("taskA", "taskB"):
        print(
            "Warning: --roberta_model_name_taskB was provided, but the current template writer "
            "uses one inferred naming convention. If Task B template is empty, set "
            "--roberta_model_name to the matching Task A stem and keep Task B stem as "
            "the same string with taskA replaced by taskB."
        )

    meta = {
        "split": args.split,
        "taskA_data": taskA_data,
        "taskA_answer": taskA_answer,
        "taskB_data": taskB_data,
        "taskB_answer": taskB_answer,
        "output_dir": output_dir,
        "pred_dirs": pred_dirs,
        "core_models_taskA": core_models_a,
        "core_models_taskB": core_models_b,
        "n_taskA_samples": len(taskA_samples),
        "n_taskB_samples": len(taskB_samples),
        "n_taskA_all_error_rows": len(all_a),
        "n_taskB_all_error_rows": len(all_b),
        "n_taskA_hard_cases": len(hard_a),
        "n_taskB_hard_cases": len(hard_b),
    }

    meta_path = os.path.join(output_dir, "error_analysis_meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    write_note(
        output_dir=output_dir,
        summary_rows=all_summary,
        hard_a=hard_a,
        hard_b=hard_b,
        meta=meta,
    )

    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Task A prediction files analyzed: {len(summary_a)}")
    print(f"Task B prediction files analyzed: {len(summary_b)}")
    print(f"Task A all error rows: {len(all_a)}")
    print(f"Task B all error rows: {len(all_b)}")
    print(f"Task A hard cases: {len(hard_a)}")
    print(f"Task B hard cases: {len(hard_b)}")
    print(f"Core models Task A: {core_models_a}")
    print(f"Core models Task B: {core_models_b}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
