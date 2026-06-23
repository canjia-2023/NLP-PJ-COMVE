#!/usr/bin/env python
# run_bert_score.py

"""
BERT LM scoring baseline for SemEval-2020 Task 4 ComVE.

Scoring modes:

1. official:
   Do not mask target tokens.
   Score each original token directly from BERT's output distribution.

2. masked:
   Leave-one-out pseudo-likelihood.
   Mask one token at a time and predict the original token.

For both modes:

    avg_log_prob = mean log p(token)

    gm_prob = exp(avg_log_prob)

    paper_score = exp(-avg_log_prob)

Higher gm_prob means the text is more natural.
Lower paper_score means the text is more natural.

Task A:
    select the lower-gm-prob sentence as nonsensical.

Task B:
    build false sentence + reason candidate;
    select the highest-gm-prob candidate as the best explanation.
"""

import argparse
import csv
import json
import math
import os
import sys
from functools import lru_cache
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from data_utils import (
    resolve_project_path,
    get_split_paths,
    load_taskA_samples,
    load_taskB_samples,
)


DEFAULT_TASKB_TEMPLATE = "{false_sent} is against common sense because {reason}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace("\\", "_")


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


def is_valid_label(label, task: str) -> bool:
    if task == "A":
        return label in [0, 1]

    if task == "B":
        return label in [0, 1, 2]

    return False


def select_paths(
    cfg: Dict,
    task: str,
    split: str,
    data_override: Optional[str] = None,
    answer_override: Optional[str] = None,
):
    if data_override is not None:
        data_path = resolve_project_path(data_override, must_exist=True)
        answer_path = (
            resolve_project_path(answer_override, must_exist=True)
            if answer_override is not None
            else None
        )
        return data_path, answer_path

    return get_split_paths(cfg, task=task, split=split)


class BertLMScorer:
    def __init__(
        self,
        model_name: str = "bert-large-uncased",
        device: str = "auto",
        scoring_mode: str = "official",
        max_length: int = 256,
        masked_batch_size: int = 32,
    ):
        if scoring_mode not in {"official", "masked"}:
            raise ValueError("--scoring_mode must be official or masked.")

        self.model_name = model_name
        self.scoring_mode = scoring_mode
        self.max_length = max_length
        self.masked_batch_size = masked_batch_size

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.scoring_mode == "masked" and self.tokenizer.mask_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name} has no mask_token_id, "
                "so masked pseudo-likelihood cannot be used."
            )

    @torch.no_grad()
    @lru_cache(maxsize=200000)
    def score_text(self, text: str) -> Dict[str, float]:
        text = str(text).strip()

        if not text:
            return {
                "avg_log_prob": float("-inf"),
                "gm_prob": 0.0,
                "paper_score": float("inf"),
                "n_tokens": 0,
            }

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )

        special_tokens_mask = encoded.pop("special_tokens_mask")[0].bool()
        model_inputs = {k: v.to(self.device) for k, v in encoded.items()}
        input_ids = model_inputs["input_ids"]

        positions = torch.where(~special_tokens_mask)[0].tolist()

        if not positions:
            return {
                "avg_log_prob": float("-inf"),
                "gm_prob": 0.0,
                "paper_score": float("inf"),
                "n_tokens": 0,
            }

        if self.scoring_mode == "official":
            avg_log_prob = self._score_official(model_inputs, positions)
        else:
            avg_log_prob = self._score_masked(model_inputs, positions)

        gm_prob = math.exp(avg_log_prob) if avg_log_prob > -745 else 0.0
        paper_score = math.exp(-avg_log_prob) if avg_log_prob < 700 else float("inf")

        return {
            "avg_log_prob": float(avg_log_prob),
            "gm_prob": float(gm_prob),
            "paper_score": float(paper_score),
            "n_tokens": int(len(positions)),
        }

    def _score_official(self, model_inputs: Dict[str, torch.Tensor], positions: List[int]) -> float:
        input_ids = model_inputs["input_ids"]

        outputs = self.model(**model_inputs)
        logits = outputs.logits[0]
        log_probs = F.log_softmax(logits, dim=-1)

        pos_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
        gold_ids = input_ids[0, pos_tensor]

        selected = log_probs[pos_tensor, gold_ids]
        return selected.mean().item()

    def _score_masked(self, model_inputs: Dict[str, torch.Tensor], positions: List[int]) -> float:
        input_ids = model_inputs["input_ids"]
        mask_id = self.tokenizer.mask_token_id

        selected_chunks = []

        for start in range(0, len(positions), self.masked_batch_size):
            batch_positions = positions[start : start + self.masked_batch_size]

            batch_input_ids = input_ids.repeat(len(batch_positions), 1)

            for row_idx, pos in enumerate(batch_positions):
                batch_input_ids[row_idx, pos] = mask_id

            batch_inputs = {}

            for key, value in model_inputs.items():
                if key == "input_ids":
                    batch_inputs[key] = batch_input_ids
                else:
                    batch_inputs[key] = value.repeat(len(batch_positions), 1)

            outputs = self.model(**batch_inputs)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            row_ids = torch.arange(len(batch_positions), device=self.device)
            pos_tensor = torch.tensor(batch_positions, dtype=torch.long, device=self.device)
            gold_ids = input_ids[0, pos_tensor]

            selected = log_probs[row_ids, pos_tensor, gold_ids]
            selected_chunks.append(selected)

        selected_all = torch.cat(selected_chunks, dim=0)
        return selected_all.mean().item()


def make_taskb_candidate(false_sent: str, reason: str, template: str) -> str:
    return template.format(
        false_sent=str(false_sent).strip(),
        reason=str(reason).strip(),
    ).strip()


def run_task_a(samples: List[Dict], scorer: BertLMScorer) -> tuple:
    rows = []
    correct_count = 0
    scored_total = 0

    for sample in tqdm(samples, desc="Task A BERT score"):
        score0 = scorer.score_text(sample["sentence0"])
        score1 = scorer.score_text(sample["sentence1"])

        probs = [score0["gm_prob"], score1["gm_prob"]]

        # Task A gold label is the index of the nonsensical sentence.
        # Lower gm_prob means less natural.
        pred = int(min(range(2), key=lambda i: probs[i]))
        gold = sample["label"]

        if is_valid_label(gold, "A"):
            correct = int(pred == gold)
            correct_count += correct
            scored_total += 1
        else:
            correct = None

        rows.append(
            {
                "id": sample["id"],
                "sentence0": sample["sentence0"],
                "sentence1": sample["sentence1"],
                "sentence0_avg_log_prob": score0["avg_log_prob"],
                "sentence1_avg_log_prob": score1["avg_log_prob"],
                "sentence0_gm_prob": score0["gm_prob"],
                "sentence1_gm_prob": score1["gm_prob"],
                "sentence0_paper_score_lower_better": score0["paper_score"],
                "sentence1_paper_score_lower_better": score1["paper_score"],
                "prediction": pred,
                "gold": gold,
                "correct": correct,
            }
        )

    metrics = {
        "taskA_total": len(samples),
        "taskA_scored_total": scored_total,
        "taskA_correct": correct_count,
        "taskA_accuracy": correct_count / scored_total if scored_total else None,
    }

    return rows, metrics


def run_task_b(samples: List[Dict], scorer: BertLMScorer, template: str) -> tuple:
    rows = []
    correct_count = 0
    scored_total = 0

    for sample in tqdm(samples, desc="Task B BERT score"):
        false_sent = sample["question"]

        reasons = [
            sample["optionA"],
            sample["optionB"],
            sample["optionC"],
        ]

        candidates = [
            make_taskb_candidate(false_sent, reason, template)
            for reason in reasons
        ]

        scores = [scorer.score_text(candidate) for candidate in candidates]
        probs = [score["gm_prob"] for score in scores]
        paper_scores = [score["paper_score"] for score in scores]

        # Task B selects the most natural false-sentence + reason combination.
        # Higher gm_prob is better.
        pred = int(max(range(3), key=lambda i: probs[i]))
        gold = sample["label"]

        if is_valid_label(gold, "B"):
            correct = int(pred == gold)
            correct_count += correct
            scored_total += 1
        else:
            correct = None

        rows.append(
            {
                "id": sample["id"],
                "false_sentence": false_sent,
                "optionA": reasons[0],
                "optionB": reasons[1],
                "optionC": reasons[2],
                "candidateA": candidates[0],
                "candidateB": candidates[1],
                "candidateC": candidates[2],
                "candidateA_gm_prob": probs[0],
                "candidateB_gm_prob": probs[1],
                "candidateC_gm_prob": probs[2],
                "candidateA_paper_score_lower_better": paper_scores[0],
                "candidateB_paper_score_lower_better": paper_scores[1],
                "candidateC_paper_score_lower_better": paper_scores[2],
                "prediction": pred,
                "gold": gold,
                "correct": correct,
            }
        )

    metrics = {
        "taskB_total": len(samples),
        "taskB_scored_total": scored_total,
        "taskB_correct": correct_count,
        "taskB_accuracy": correct_count / scored_total if scored_total else None,
    }

    return rows, metrics


def save_csv(rows: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT LM scoring baseline for Task A/B.")

    parser.add_argument("--task", type=str, default="both", help="A, B, or both.")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])

    parser.add_argument("--config", type=str, default=os.path.join(SRC_DIR, "config.yaml"))

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override data path when --task is A or B.",
    )
    parser.add_argument(
        "--answer",
        type=str,
        default=None,
        help="Override answer path when --task is A or B.",
    )

    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="bert-large-uncased")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="official",
        choices=["official", "masked"],
    )

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--masked_batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--taskb_template",
        type=str,
        default=DEFAULT_TASKB_TEMPLATE,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    task = normalize_task_arg(args.task)
    cfg = load_config(args.config)

    if task == "both" and (args.data is not None or args.answer is not None):
        raise ValueError("--data / --answer can only be used when --task is A or B.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = cfg.get("llm_scoring", {}).get(
            "bert_output_dir",
            "outputs/llm_scoring/bert_score",
        )

    output_dir = resolve_project_path(output_dir, must_exist=False)
    ensure_dir(output_dir)

    scorer = BertLMScorer(
        model_name=args.model_name,
        device=args.device,
        scoring_mode=args.scoring_mode,
        max_length=args.max_length,
        masked_batch_size=args.masked_batch_size,
    )

    safe_model = safe_name(args.model_name)
    all_metrics = {
        "model_name": args.model_name,
        "scoring_mode": args.scoring_mode,
        "split": args.split,
        "task": task,
        "max_length": args.max_length,
        "taskb_template": args.taskb_template,
    }

    if task in {"A", "both"}:
        data_path, answer_path = select_paths(
            cfg=cfg,
            task="A",
            split=args.split,
            data_override=args.data if task == "A" else None,
            answer_override=args.answer if task == "A" else None,
        )

        samples = load_taskA_samples(
            data_path=data_path,
            answer_path=answer_path,
            require_labels=False,
        )

        if args.limit is not None:
            samples = samples[: args.limit]

        rows, metrics = run_task_a(samples, scorer)
        all_metrics.update(metrics)
        all_metrics["taskA_data_path"] = data_path
        all_metrics["taskA_answer_path"] = answer_path

        out_path = os.path.join(
            output_dir,
            f"{args.split}_taskA_{safe_model}_{args.scoring_mode}.csv",
        )
        save_csv(rows, out_path)
        print(f"Saved Task A predictions to: {out_path}")
        print(f"Task A accuracy: {metrics['taskA_accuracy']}")

    if task in {"B", "both"}:
        data_path, answer_path = select_paths(
            cfg=cfg,
            task="B",
            split=args.split,
            data_override=args.data if task == "B" else None,
            answer_override=args.answer if task == "B" else None,
        )

        samples = load_taskB_samples(
            data_path=data_path,
            answer_path=answer_path,
            require_labels=False,
        )

        if args.limit is not None:
            samples = samples[: args.limit]

        rows, metrics = run_task_b(samples, scorer, template=args.taskb_template)
        all_metrics.update(metrics)
        all_metrics["taskB_data_path"] = data_path
        all_metrics["taskB_answer_path"] = answer_path

        out_path = os.path.join(
            output_dir,
            f"{args.split}_taskB_{safe_model}_{args.scoring_mode}.csv",
        )
        save_csv(rows, out_path)
        print(f"Saved Task B predictions to: {out_path}")
        print(f"Task B accuracy: {metrics['taskB_accuracy']}")

    metrics_path = os.path.join(
        output_dir,
        f"{args.split}_metrics_{safe_model}_{args.scoring_mode}.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
