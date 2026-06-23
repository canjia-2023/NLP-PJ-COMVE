#!/usr/bin/env python
# shuffle_data.py

"""
Generate shuffled datasets for Task A / Task B.

This script only generates shuffled CSV files.
Evaluation should be done by existing evaluate.py / run_llm_prompt.py / run_bert_score.py / run_ppl.py.

Examples:
    python src/robustness/shuffle_data.py --task both --split test

    python src/robustness/shuffle_data.py --task A --split dev --seed 42

    python src/robustness/shuffle_data.py \
        --task both \
        --split test \
        --output_dir outputs/robustness/shuffle/test_seed42
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List

import yaml


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # project/src/robustness
SRC_DIR = os.path.dirname(CURRENT_DIR)                     # project/src
PROJECT_DIR = os.path.dirname(SRC_DIR)                     # project

sys.path.append(SRC_DIR)

from data_utils import (
    resolve_project_path,
    get_split_paths,
    load_taskA_samples,
    load_taskB_samples,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)

    candidates = [
        os.path.join(os.getcwd(), path),
        os.path.join(SRC_DIR, path),
        os.path.join(PROJECT_DIR, path),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)

    return os.path.normpath(os.path.join(SRC_DIR, path))


def load_config(path: str) -> Dict:
    config_path = resolve_config_path(path)

    with open(config_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def normalize_task_arg(task: str) -> str:
    task = task.strip()

    if task.lower() == "both":
        return "both"

    task = task.upper()

    if task not in {"A", "B"}:
        raise ValueError("--task must be A, B, or both.")

    return task


def save_taskA_csv(samples: List[Dict], data_path: str, answer_path: str):
    ensure_dir(os.path.dirname(data_path))
    ensure_dir(os.path.dirname(answer_path))

    with open(data_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "sent0", "sent1"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for s in samples:
            writer.writerow(
                {
                    "id": s["id"],
                    "sent0": s["sentence0"],
                    "sent1": s["sentence1"],
                }
            )

    with open(answer_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        for s in samples:
            writer.writerow([s["id"], s["label"]])


def save_taskB_csv(samples: List[Dict], data_path: str, answer_path: str):
    ensure_dir(os.path.dirname(data_path))
    ensure_dir(os.path.dirname(answer_path))

    with open(data_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "FalseSent", "OptionA", "OptionB", "OptionC"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for s in samples:
            writer.writerow(
                {
                    "id": s["id"],
                    "FalseSent": s["question"],
                    "OptionA": s["optionA"],
                    "OptionB": s["optionB"],
                    "OptionC": s["optionC"],
                }
            )

    with open(answer_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        for s in samples:
            writer.writerow([s["id"], s["label"]])


def shuffle_taskA_samples(samples: List[Dict], seed: int = 42):
    """
    Task A:
        50% probability to swap sentence0 and sentence1.
        If swapped, flip label: 0 <-> 1.
    """
    rng = random.Random(seed)

    shuffled = []
    shuffle_map = []

    for s in samples:
        s_new = dict(s)
        swapped = rng.random() < 0.5

        old_label = s_new["label"]

        if swapped:
            s_new["sentence0"], s_new["sentence1"] = (
                s_new["sentence1"],
                s_new["sentence0"],
            )

            if s_new["label"] in [0, 1]:
                s_new["label"] = 1 - s_new["label"]

        shuffled.append(s_new)

        shuffle_map.append(
            {
                "id": s_new["id"],
                "swapped": swapped,
                "old_label": old_label,
                "new_label": s_new["label"],
            }
        )

    return shuffled, shuffle_map


def shuffle_taskB_samples(samples: List[Dict], seed: int = 42):
    """
    Task B:
        Randomly permute OptionA / OptionB / OptionC.
        Remap label according to the new option order.
    """
    rng = random.Random(seed)

    shuffled = []
    shuffle_map = []

    for s in samples:
        s_new = dict(s)

        options = [
            s_new["optionA"],
            s_new["optionB"],
            s_new["optionC"],
        ]

        old_label = s_new["label"]

        perm = list(range(3))
        rng.shuffle(perm)

        shuffled_options = [options[i] for i in perm]

        s_new["optionA"] = shuffled_options[0]
        s_new["optionB"] = shuffled_options[1]
        s_new["optionC"] = shuffled_options[2]

        if old_label in [0, 1, 2]:
            s_new["label"] = perm.index(old_label)

        shuffled.append(s_new)

        shuffle_map.append(
            {
                "id": s_new["id"],
                "perm": perm,
                "old_label": old_label,
                "new_label": s_new["label"],
            }
        )

    return shuffled, shuffle_map


def save_map_csv(rows: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def run_taskA(cfg: Dict, split: str, output_dir: str, seed: int):
    data_path, answer_path = get_split_paths(cfg, task="A", split=split)

    samples = load_taskA_samples(
        data_path=data_path,
        answer_path=answer_path,
        require_labels=True,
    )

    shuffled_samples, shuffle_map = shuffle_taskA_samples(samples, seed=seed)

    out_data_path = os.path.join(
        output_dir,
        f"subtaskA_{split}_data_shuffled_seed{seed}.csv",
    )
    out_answer_path = os.path.join(
        output_dir,
        f"subtaskA_{split}_answers_shuffled_seed{seed}.csv",
    )
    out_map_path = os.path.join(
        output_dir,
        f"subtaskA_{split}_shuffle_map_seed{seed}.csv",
    )

    save_taskA_csv(shuffled_samples, out_data_path, out_answer_path)
    save_map_csv(shuffle_map, out_map_path)

    print(f"Task A shuffled data saved to: {out_data_path}")
    print(f"Task A shuffled answers saved to: {out_answer_path}")
    print(f"Task A shuffle map saved to: {out_map_path}")

    return {
        "taskA_original_data": data_path,
        "taskA_original_answer": answer_path,
        "taskA_shuffled_data": out_data_path,
        "taskA_shuffled_answer": out_answer_path,
        "taskA_shuffle_map": out_map_path,
        "taskA_total": len(shuffled_samples),
    }


def run_taskB(cfg: Dict, split: str, output_dir: str, seed: int):
    data_path, answer_path = get_split_paths(cfg, task="B", split=split)

    samples = load_taskB_samples(
        data_path=data_path,
        answer_path=answer_path,
        require_labels=True,
    )

    shuffled_samples, shuffle_map = shuffle_taskB_samples(samples, seed=seed)

    out_data_path = os.path.join(
        output_dir,
        f"subtaskB_{split}_data_shuffled_seed{seed}.csv",
    )
    out_answer_path = os.path.join(
        output_dir,
        f"subtaskB_{split}_answers_shuffled_seed{seed}.csv",
    )
    out_map_path = os.path.join(
        output_dir,
        f"subtaskB_{split}_shuffle_map_seed{seed}.csv",
    )

    save_taskB_csv(shuffled_samples, out_data_path, out_answer_path)
    save_map_csv(shuffle_map, out_map_path)

    print(f"Task B shuffled data saved to: {out_data_path}")
    print(f"Task B shuffled answers saved to: {out_answer_path}")
    print(f"Task B shuffle map saved to: {out_map_path}")

    return {
        "taskB_original_data": data_path,
        "taskB_original_answer": answer_path,
        "taskB_shuffled_data": out_data_path,
        "taskB_shuffled_answer": out_answer_path,
        "taskB_shuffle_map": out_map_path,
        "taskB_total": len(shuffled_samples),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate shuffled Task A/B datasets.")

    parser.add_argument(
        "--task",
        type=str,
        default="both",
        help="A, B, or both.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
    )

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SRC_DIR, "config.yaml"),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save shuffled CSV files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    task = normalize_task_arg(args.task)
    cfg = load_config(args.config)

    if args.output_dir is None:
        output_dir = cfg.get("robustness", {}).get(
            "shuffle_output_dir",
            f"outputs/robustness/shuffle/{args.split}_seed{args.seed}",
        )
    else:
        output_dir = args.output_dir

    output_dir = resolve_project_path(output_dir, must_exist=False)
    ensure_dir(output_dir)

    meta = {
        "task": task,
        "split": args.split,
        "seed": args.seed,
        "output_dir": output_dir,
    }

    if task in {"A", "both"}:
        meta.update(run_taskA(cfg, split=args.split, output_dir=output_dir, seed=args.seed))

    if task in {"B", "both"}:
        meta.update(run_taskB(cfg, split=args.split, output_dir=output_dir, seed=args.seed))

    meta_path = os.path.join(output_dir, f"shuffle_meta_{args.split}_seed{args.seed}.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Shuffle meta saved to: {meta_path}")


if __name__ == "__main__":
    main()
