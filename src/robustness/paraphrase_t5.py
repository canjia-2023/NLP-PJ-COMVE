#!/usr/bin/env python
# paraphrase_t5.py

"""
Generate T5-paraphrased robustness datasets for Task A / Task B.

No candidate-review workflow.
This script directly generates one selected paraphrase for each target sentence.

Task A:
    Paraphrase both sent0 and sent1.
    Keep the original label unchanged.

Task B:
    Paraphrase FalseSent only.
    Keep OptionA / OptionB / OptionC and label unchanged.

Examples:
    python src/robustness/paraphrase_t5.py --task A --split dev --sample_size 50

    python src/robustness/paraphrase_t5.py --task B --split dev --sample_size 50

    python src/robustness/paraphrase_t5.py --task both --split test --sample_size 100 \
        --output_dir outputs/robustness/paraphrase/test_seed42
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import set_seed as hf_set_seed


# ======== Path Setup ========

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


# ======== Basic Utilities ========

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_text(text) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(":", "_")


def load_config(path: str) -> Dict:
    config_path = resolve_project_path(path, must_exist=True)

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


def stratified_sample(samples: List[Dict], sample_size: int, seed: int = 42) -> List[Dict]:
    """
    Stratified sample by label when labels are available.
    If sample_size <= 0, use all samples.
    """
    if sample_size <= 0 or sample_size >= len(samples):
        return samples[:]

    rng = random.Random(seed)

    labels_exist = any(s.get("label") is not None for s in samples)

    if not labels_exist:
        return rng.sample(samples, sample_size)

    buckets = defaultdict(list)

    for s in samples:
        buckets[s.get("label")].append(s)

    for label in buckets:
        rng.shuffle(buckets[label])

    selected = []

    while len(selected) < sample_size:
        made_progress = False

        for label in sorted(buckets.keys(), key=lambda x: str(x)):
            if buckets[label] and len(selected) < sample_size:
                selected.append(buckets[label].pop())
                made_progress = True

        if not made_progress:
            break

    rng.shuffle(selected)

    return selected


# ======== T5 Paraphraser ========

class T5Paraphraser:
    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        device: str = "auto",
        max_input_length: int = 128,
        max_output_length: int = 128,
        num_try: int = 4,
        top_k: int = 120,
        top_p: float = 0.95,
        temperature: float = 1.0,
        do_sample: bool = True,
    ):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_try = max(1, num_try)
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded paraphrase model: {model_name}")
        print(f"Device: {self.device}")

    def _clean_generated_text(self, text: str) -> str:
        text = normalize_text(text)

        prefixes = [
            "paraphrasedoutput:",
            "paraphrased output:",
            "paraphrase:",
            "paraphrased:",
        ]

        lower = text.lower()

        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return normalize_text(text)

    def _is_valid_paraphrase(self, source: str, candidate: str) -> bool:
        source = normalize_text(source)
        candidate = normalize_text(candidate)

        if not candidate:
            return False

        if candidate.lower() == source.lower():
            return False

        # 太短的输出一般是坏结果
        if len(candidate.split()) <= 1 and len(source.split()) > 1:
            return False

        return True

    @torch.inference_mode()
    def paraphrase_one(self, text: str) -> Tuple[str, bool]:
        """
        Return:
            paraphrased_text, used_fallback

        used_fallback=True means no good paraphrase was found and original text is returned.
        """
        source = normalize_text(text)

        if not source:
            return source, True

        prompt = f"paraphrase: {source} </s>"

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        num_return_sequences = self.num_try if self.do_sample else 1

        generate_kwargs = {
            **encoding,
            "max_length": self.max_output_length,
            "num_return_sequences": num_return_sequences,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            generate_kwargs.update(
                {
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                }
            )

        outputs = self.model.generate(**generate_kwargs)

        for output in outputs:
            candidate = self.tokenizer.decode(output, skip_special_tokens=True)
            candidate = self._clean_generated_text(candidate)

            if self._is_valid_paraphrase(source, candidate):
                return candidate, False

        return source, True


# ======== CSV Saving ========

def save_taskA_data(samples: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
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


def save_taskB_data(samples: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
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


def save_answers(samples: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        for s in samples:
            writer.writerow([s["id"], s.get("label")])


def save_map_csv(rows: List[Dict], path: str):
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


def make_base_name(task: str, split: str, seed: int, actual_n: int, sample_size: int, tag: str):
    if sample_size <= 0:
        sample_part = "all"
    else:
        sample_part = f"n{actual_n}"

    return f"subtask{task}_{split}_{tag}_seed{seed}_{sample_part}"


# ======== Task Running ========

def run_taskA(
    cfg: Dict,
    split: str,
    output_dir: str,
    paraphraser: T5Paraphraser,
    seed: int,
    sample_size: int,
    tag: str,
    data_override: Optional[str] = None,
    answer_override: Optional[str] = None,
):
    data_path, answer_path = select_paths(
        cfg=cfg,
        task="A",
        split=split,
        data_override=data_override,
        answer_override=answer_override,
    )

    samples = load_taskA_samples(
        data_path=data_path,
        answer_path=answer_path,
        require_labels=answer_path is not None,
    )

    selected = stratified_sample(samples, sample_size=sample_size, seed=seed)

    paraphrased_samples = []
    map_rows = []

    for s in tqdm(selected, desc="Task A paraphrase"):
        sent0_para, sent0_fallback = paraphraser.paraphrase_one(s["sentence0"])
        sent1_para, sent1_fallback = paraphraser.paraphrase_one(s["sentence1"])

        new_s = dict(s)
        new_s["sentence0"] = sent0_para
        new_s["sentence1"] = sent1_para
        paraphrased_samples.append(new_s)

        map_rows.append(
            {
                "id": s["id"],
                "label": s.get("label"),
                "sent0_original": s["sentence0"],
                "sent0_paraphrase": sent0_para,
                "sent0_fallback": sent0_fallback,
                "sent0_changed": int(normalize_text(s["sentence0"]).lower() != normalize_text(sent0_para).lower()),
                "sent1_original": s["sentence1"],
                "sent1_paraphrase": sent1_para,
                "sent1_fallback": sent1_fallback,
                "sent1_changed": int(normalize_text(s["sentence1"]).lower() != normalize_text(sent1_para).lower()),
            }
        )

    base_name = make_base_name(
        task="A",
        split=split,
        seed=seed,
        actual_n=len(selected),
        sample_size=sample_size,
        tag=tag,
    )

    para_data_path = os.path.join(output_dir, base_name + "_data.csv")
    answer_path_out = os.path.join(output_dir, base_name + "_answers.csv")
    original_data_path = os.path.join(output_dir, base_name + "_original_subset_data.csv")
    map_path = os.path.join(output_dir, base_name + "_map.csv")

    save_taskA_data(paraphrased_samples, para_data_path)
    save_taskA_data(selected, original_data_path)
    save_answers(selected, answer_path_out)
    save_map_csv(map_rows, map_path)

    print(f"Task A paraphrased data saved to: {para_data_path}")
    print(f"Task A original subset saved to: {original_data_path}")
    print(f"Task A answers saved to: {answer_path_out}")
    print(f"Task A map saved to: {map_path}")

    return {
        "taskA_original_data": data_path,
        "taskA_original_answer": answer_path,
        "taskA_paraphrased_data": para_data_path,
        "taskA_original_subset_data": original_data_path,
        "taskA_answers": answer_path_out,
        "taskA_map": map_path,
        "taskA_total": len(selected),
    }


def run_taskB(
    cfg: Dict,
    split: str,
    output_dir: str,
    paraphraser: T5Paraphraser,
    seed: int,
    sample_size: int,
    tag: str,
    data_override: Optional[str] = None,
    answer_override: Optional[str] = None,
):
    data_path, answer_path = select_paths(
        cfg=cfg,
        task="B",
        split=split,
        data_override=data_override,
        answer_override=answer_override,
    )

    samples = load_taskB_samples(
        data_path=data_path,
        answer_path=answer_path,
        require_labels=answer_path is not None,
    )

    selected = stratified_sample(samples, sample_size=sample_size, seed=seed)

    paraphrased_samples = []
    map_rows = []

    for s in tqdm(selected, desc="Task B paraphrase"):
        false_para, false_fallback = paraphraser.paraphrase_one(s["question"])

        new_s = dict(s)
        new_s["question"] = false_para
        paraphrased_samples.append(new_s)

        map_rows.append(
            {
                "id": s["id"],
                "label": s.get("label"),
                "false_sent_original": s["question"],
                "false_sent_paraphrase": false_para,
                "false_sent_fallback": false_fallback,
                "false_sent_changed": int(normalize_text(s["question"]).lower() != normalize_text(false_para).lower()),
                "optionA": s["optionA"],
                "optionB": s["optionB"],
                "optionC": s["optionC"],
            }
        )

    base_name = make_base_name(
        task="B",
        split=split,
        seed=seed,
        actual_n=len(selected),
        sample_size=sample_size,
        tag=tag,
    )

    para_data_path = os.path.join(output_dir, base_name + "_data.csv")
    answer_path_out = os.path.join(output_dir, base_name + "_answers.csv")
    original_data_path = os.path.join(output_dir, base_name + "_original_subset_data.csv")
    map_path = os.path.join(output_dir, base_name + "_map.csv")

    save_taskB_data(paraphrased_samples, para_data_path)
    save_taskB_data(selected, original_data_path)
    save_answers(selected, answer_path_out)
    save_map_csv(map_rows, map_path)

    print(f"Task B paraphrased data saved to: {para_data_path}")
    print(f"Task B original subset saved to: {original_data_path}")
    print(f"Task B answers saved to: {answer_path_out}")
    print(f"Task B map saved to: {map_path}")

    return {
        "taskB_original_data": data_path,
        "taskB_original_answer": answer_path,
        "taskB_paraphrased_data": para_data_path,
        "taskB_original_subset_data": original_data_path,
        "taskB_answers": answer_path_out,
        "taskB_map": map_path,
        "taskB_total": len(selected),
    }


# ======== Entry Point ========

def parse_args():
    parser = argparse.ArgumentParser(description="Generate T5 paraphrased robustness datasets.")

    parser.add_argument("--task", type=str, default="both", help="A, B, or both.")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SRC_DIR, "config.yaml"),
    )

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

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save paraphrased CSV files.",
    )

    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tag", type=str, default="t5_para")

    parser.add_argument("--model_name", type=str, default="Vamsi/T5_Paraphrase_Paws")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_output_length", type=int, default=128)

    parser.add_argument(
        "--num_try",
        type=int,
        default=4,
        help="Generate internally this many outputs and keep the first valid one. Not saved as candidates.",
    )

    parser.add_argument("--top_k", type=int, default=120)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling. Usually less diverse but more deterministic.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    task = normalize_task_arg(args.task)

    if task == "both" and (args.data is not None or args.answer is not None):
        raise ValueError("--data / --answer can only be used when --task is A or B.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    hf_set_seed(args.seed)

    cfg = load_config(args.config)

    if args.output_dir is None:
        base_output_dir = cfg.get("robustness", {}).get(
            "paraphrase_output_dir",
            "outputs/robustness/paraphrase",
        )
        output_dir = os.path.join(base_output_dir, f"{args.split}_seed{args.seed}")
    else:
        output_dir = args.output_dir

    output_dir = resolve_project_path(output_dir, must_exist=False)
    ensure_dir(output_dir)

    print("=" * 60)
    print("T5 paraphrase robustness generation")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Split: {args.split}")
    print(f"Sample size: {args.sample_size}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {args.model_name}")
    print("=" * 60)

    paraphraser = T5Paraphraser(
        model_name=args.model_name,
        device=args.device,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        num_try=args.num_try,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        do_sample=not args.no_sample,
    )

    meta = {
        "task": task,
        "split": args.split,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "tag": args.tag,
        "model_name": args.model_name,
        "output_dir": output_dir,
        "max_input_length": args.max_input_length,
        "max_output_length": args.max_output_length,
        "num_try": args.num_try,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "do_sample": bool(not args.no_sample),
    }

    if task in {"A", "both"}:
        meta.update(
            run_taskA(
                cfg=cfg,
                split=args.split,
                output_dir=output_dir,
                paraphraser=paraphraser,
                seed=args.seed,
                sample_size=args.sample_size,
                tag=args.tag,
                data_override=args.data if task == "A" else None,
                answer_override=args.answer if task == "A" else None,
            )
        )

    if task in {"B", "both"}:
        meta.update(
            run_taskB(
                cfg=cfg,
                split=args.split,
                output_dir=output_dir,
                paraphraser=paraphraser,
                seed=args.seed,
                sample_size=args.sample_size,
                tag=args.tag,
                data_override=args.data if task == "B" else None,
                answer_override=args.answer if task == "B" else None,
            )
        )

    meta_path = os.path.join(
        output_dir,
        f"paraphrase_meta_{args.split}_{safe_name(args.tag)}_seed{args.seed}.json",
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Meta saved to: {meta_path}")


if __name__ == "__main__":
    main()
