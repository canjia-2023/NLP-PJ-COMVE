#!/usr/bin/env python
# run_ppl.py

"""
Causal LM next-token PPL baseline for SemEval-2020 Task 4 ComVE.

This script uses autoregressive causal language models, such as:
    gpt2
    distilgpt2
    EleutherAI/gpt-neo-125m

It does not use BERT.

Formula:

    NLL(x) = - mean_t log P(x_t | x_<t)

    PPL(x) = exp(NLL(x))

Lower PPL means the sequence is more natural.
Higher PPL means the sequence is less natural.

Task A:
    select the higher-PPL sentence as nonsensical.

Task B:
    build false sentence + reason candidate;
    select the lowest-PPL candidate as the best explanation.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


CHOICES_B = ["A", "B", "C"]
DEFAULT_TASKB_TEMPLATE = "{false_sent} This is against common sense because {reason}"


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


@dataclass
class ScoreCache:
    path: Optional[str]
    data: Dict[str, float]

    @classmethod
    def load(cls, path: Optional[str]) -> "ScoreCache":
        if path is None:
            return cls(path=None, data={})

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                path=path,
                data={str(k): float(v) for k, v in data.items()},
            )

        return cls(path=path, data={})

    def get(self, key: str) -> Optional[float]:
        return self.data.get(key)

    def set(self, key: str, value: float):
        self.data[key] = float(value)

    def save(self):
        if self.path is None:
            return

        ensure_dir(os.path.dirname(self.path))

        tmp_path = self.path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)

        os.replace(tmp_path, self.path)


class CausalLMPPLScorer:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_length: int = 128,
        fp16: bool = False,
        add_bos: bool = True,
        cache: Optional[ScoreCache] = None,
    ):
        if "bert" in model_name.lower():
            raise ValueError(
                f"Model {model_name!r} is not allowed in run_ppl.py. "
                "Use a causal LM such as gpt2 or distilgpt2."
            )

        self.model_name = model_name
        self.max_length = max_length
        self.add_bos = add_bos

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        if fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self.model.eval()

        self.bos_token_id = self.tokenizer.bos_token_id

        if self.bos_token_id is None:
            self.bos_token_id = self.tokenizer.eos_token_id

        if self.add_bos and self.bos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name} has neither bos_token_id nor eos_token_id. "
                "Run with --no_add_bos or choose another causal LM."
            )

        self.cache = cache or ScoreCache(path=None, data={})

    def _cache_key(self, text: str) -> str:
        return (
            f"causal_next_token_ppl::"
            f"{self.model_name}::"
            f"max{self.max_length}::"
            f"bos{self.add_bos}::"
            f"{text}"
        )

    @torch.inference_mode()
    def score(self, text: str) -> float:
        text = str(text).strip()

        if not text:
            return float("inf")

        key = self._cache_key(text)
        cached = self.cache.get(key)

        if cached is not None:
            return cached

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if not token_ids:
            return float("inf")

        if self.add_bos:
            token_ids = [int(self.bos_token_id)] + token_ids

        if len(token_ids) < 2:
            return float("inf")

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits.float()

        # logits at position t predict token at position t + 1.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        ppl = float(torch.exp(torch.clamp(nll, max=80.0)).item())

        self.cache.set(key, ppl)

        return ppl


def make_taskb_candidate(false_sent: str, reason: str, template: str) -> str:
    return template.format(
        false_sent=str(false_sent).strip(),
        reason=str(reason).strip(),
    ).strip()


def run_task_a(samples: List[Dict], scorer: CausalLMPPLScorer) -> tuple:
    rows = []
    correct_count = 0
    scored_total = 0

    for sample in tqdm(samples, desc="Task A PPL"):
        sent0 = sample["sentence0"]
        sent1 = sample["sentence1"]

        ppl0 = scorer.score(sent0)
        ppl1 = scorer.score(sent1)

        # Task A gold label is the index of the nonsensical sentence.
        # Higher PPL means less natural.
        pred = int(np.argmax([ppl0, ppl1]))
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
                "sentence0": sent0,
                "sentence1": sent1,
                "sentence0_ppl": ppl0,
                "sentence1_ppl": ppl1,
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


def run_task_b(samples: List[Dict], scorer: CausalLMPPLScorer, template: str) -> tuple:
    rows = []
    correct_count = 0
    scored_total = 0

    for sample in tqdm(samples, desc="Task B PPL"):
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

        ppls = [scorer.score(candidate) for candidate in candidates]

        # Task B selects the most natural false-sentence + reason combination.
        # Lower PPL is better.
        pred = int(np.argmin(ppls))
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
                "candidateA_ppl": ppls[0],
                "candidateB_ppl": ppls[1],
                "candidateC_ppl": ppls[2],
                "prediction": pred,
                "prediction_letter": CHOICES_B[pred],
                "gold": gold,
                "gold_letter": CHOICES_B[gold] if gold in [0, 1, 2] else "",
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
    parser = argparse.ArgumentParser(description="Causal LM PPL baseline for Task A/B.")

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

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_add_bos", action="store_true")

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = cfg.get("llm_scoring", {}).get(
            "ppl_output_dir",
            "outputs/llm_scoring/ppl",
        )

    output_dir = resolve_project_path(output_dir, must_exist=False)
    ensure_dir(output_dir)

    safe_model = safe_name(args.model_name)

    cache = None

    if not args.no_cache:
        cache_path = os.path.join(
            output_dir,
            f"score_cache_{args.split}_{safe_model}_max{args.max_length}_ppl.json",
        )
        cache = ScoreCache.load(cache_path)

    scorer = CausalLMPPLScorer(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        fp16=args.fp16,
        add_bos=not args.no_add_bos,
        cache=cache,
    )

    all_metrics = {
        "model_name": args.model_name,
        "model_type": "AutoModelForCausalLM",
        "scoring": "next_token_perplexity",
        "uses_bert": False,
        "split": args.split,
        "task": task,
        "max_length": args.max_length,
        "add_bos": bool(not args.no_add_bos),
        "taskb_template": args.taskb_template,
        "limit": args.limit if args.limit is not None else "none",
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
            f"{args.split}_taskA_{safe_model}_ppl.csv",
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
            f"{args.split}_taskB_{safe_model}_ppl.csv",
        )
        save_csv(rows, out_path)
        print(f"Saved Task B predictions to: {out_path}")
        print(f"Task B accuracy: {metrics['taskB_accuracy']}")

    scorer.cache.save()

    metrics_path = os.path.join(
        output_dir,
        f"{args.split}_metrics_{safe_model}_ppl.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
