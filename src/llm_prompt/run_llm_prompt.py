#!/usr/bin/env python
# run_llm_prompt.py

"""
Zero-shot / One-shot LLM prompting for Task A & Task B.

Examples:
    python src/llm_prompt/run_llm_prompt.py --task A --model qwen3:8b --shot zero

    python src/llm_prompt/run_llm_prompt.py --task A --model qwen3:8b --shot one --limit 5

    python src/llm_prompt/run_llm_prompt.py --task A --model qwen3:8b --shot one \
        --eval_tag t5_original \
        --data outputs/taskA_dev_t5_cand1_original_subset_data.csv \
        --answer outputs/taskA_dev_t5_cand1_answers.csv

    python src/llm_prompt/run_llm_prompt.py --task A --model qwen3:8b --shot one \
        --eval_tag t5_para \
        --data outputs/taskA_dev_t5_cand1_data.csv \
        --answer outputs/taskA_dev_t5_cand1_answers.csv
"""

import argparse
import copy
import gc
import json
import os
import random
import re
import sys
import time

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ======== Path Setup ========

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # project/src/llm_prompt
SRC_DIR = os.path.dirname(CURRENT_DIR)                     # project/src
PROJECT_DIR = os.path.dirname(SRC_DIR)                     # project

sys.path.append(SRC_DIR)

from data_utils import (
    resolve_project_path,
    load_taskA_samples,
    load_taskB_samples,
)
from utils import set_seed


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_config_path(path):
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


def load_config(path):
    config_path = resolve_config_path(path)

    with open(config_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ======== Model Name Mapping ========

MODEL_ALIASES = {
    "qwen3:1.7b": "Qwen/Qwen3-1.7B",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:8b": "Qwen/Qwen3-8B",

    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",

    # 如果你实际用的是 1124 版本，就只改这里
    "olmo2:7b": "allenai/OLMo-2-0725-7B-Instruct",
}


def resolve_model_name(name):
    return MODEL_ALIASES.get(name.lower(), name)


def safe_name(name):
    return name.replace("/", "_").replace(":", "_").replace("\\", "_")


def is_valid_label(label, task):
    if task == "A":
        return label in [0, 1]

    if task == "B":
        return label in [0, 1, 2]

    return False


# ======== Shuffle Robustness ========

def shuffle_task_a(samples, seed=42):
    rng = random.Random(seed)
    shuffled = []

    for s in samples:
        s_new = copy.deepcopy(s)

        if rng.random() < 0.5:
            s_new["sentence0"], s_new["sentence1"] = (
                s_new["sentence1"],
                s_new["sentence0"],
            )

            if s_new["label"] in [0, 1]:
                s_new["label"] = 1 - s_new["label"]

            s_new["_shuffled"] = True
        else:
            s_new["_shuffled"] = False

        shuffled.append(s_new)

    return shuffled


def shuffle_task_b(samples, seed=42):
    rng = random.Random(seed)
    shuffled = []

    for s in samples:
        s_new = copy.deepcopy(s)

        options = [s_new["optionA"], s_new["optionB"], s_new["optionC"]]

        perm = list(range(3))
        rng.shuffle(perm)

        s_new["optionA"], s_new["optionB"], s_new["optionC"] = [
            options[i] for i in perm
        ]

        original_label = s_new["label"]

        if original_label in [0, 1, 2]:
            s_new["label"] = perm.index(original_label)

        s_new["_shuffled"] = True
        s_new["_perm"] = perm

        shuffled.append(s_new)

    return shuffled


# ======== Prompt Construction ========

def make_prompt_taskA(sample, shot="zero"):
    if shot == "one":
        return (
            "Exactly one of the following two sentences violates common sense.\n"
            "Think step by step, then end your response with 'The answer is 0' or 'The answer is 1'.\n"
            "\n"
            "Example:\n"
            "Sentence 0: She put the milk in the refrigerator.\n"
            "Sentence 1: She put the milk in the volcano.\n"
            "Thinking: A refrigerator is a normal place to store milk. "
            "A volcano is extremely hot and dangerous, so Sentence 1 is nonsensical.\n"
            "The answer is 1\n"
            "\n"
            "Now answer this one.\n"
            "\n"
            f"Sentence 0: {sample['sentence0']}\n"
            f"Sentence 1: {sample['sentence1']}\n"
        )

    return (
        "Exactly one of the following two sentences violates common sense.\n"
        "Think step by step, then end your response with 'The answer is 0' or 'The answer is 1'.\n"
        "\n"
        f"Sentence 0: {sample['sentence0']}\n"
        f"Sentence 1: {sample['sentence1']}\n"
    )


def make_prompt_taskB(sample, shot="zero"):
    if shot == "one":
        return (
            "Given a nonsensical sentence, choose the most important reason "
            "why it does not make sense.\n"
            "Think step by step, then end your response with 'The answer is A', "
            "'The answer is B', or 'The answer is C'.\n"
            "\n"
            "Example:\n"
            "Sentence: She put the milk in the volcano.\n"
            "A: Milk is too heavy to carry.\n"
            "B: A volcano is extremely hot and would destroy the milk.\n"
            "C: Milk should be stored in a bag.\n"
            "Thinking: The sentence is about putting milk in a volcano. "
            "Option B correctly explains why this is nonsensical.\n"
            "The answer is B\n"
            "\n"
            "Now answer this one.\n"
            "\n"
            f"Sentence: {sample['question']}\n"
            f"A: {sample['optionA']}\n"
            f"B: {sample['optionB']}\n"
            f"C: {sample['optionC']}\n"
        )

    return (
        "Given a nonsensical sentence, choose the most important reason "
        "why it does not make sense.\n"
        "Think step by step, then end your response with 'The answer is A', "
        "'The answer is B', or 'The answer is C'.\n"
        "\n"
        f"Sentence: {sample['question']}\n"
        f"A: {sample['optionA']}\n"
        f"B: {sample['optionB']}\n"
        f"C: {sample['optionC']}\n"
    )


# ======== Model Loading & Generation ========

def load_model(model_id, load_in_4bit=True):
    print(f"Loading model: {model_id}")

    if load_in_4bit and not torch.cuda.is_available():
        print("CUDA is not available, so 4-bit quantization is disabled.")
        load_in_4bit = False

    print(f"4-bit quantization: {load_in_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory — allocated: {allocated:.1f} GB, reserved: {reserved:.1f} GB")

    print("Model loaded successfully.")

    return model, tokenizer


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_chat_input(tokenizer, prompt, model_id):
    messages = [{"role": "user", "content": prompt}]
    model_id_lower = model_id.lower()

    # Qwen3 支持关闭 thinking
    if "qwen3" in model_id_lower or "qwen-3" in model_id_lower:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
        except Exception:
            pass

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt,
    model_id,
    max_new_tokens=512,
    max_input_length=2048,
):
    chat_text = build_chat_input(tokenizer, prompt, model_id)

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    input_device = get_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    outputs = model.generate(**generate_kwargs)

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


# ======== Response Parsing ========

def remove_think_block(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_response_taskA(response_text):
    text = remove_think_block(response_text).strip()

    if not text:
        return -1

    matches = re.findall(r"\bthe answer is\s*([01])\b", text, re.IGNORECASE)
    if matches:
        return int(matches[-1])

    match = re.search(r"\banswer\s*(?:is|:|：)?\s*([01])\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.match(r"^\s*([01])\b", text)
    if match:
        return int(match.group(1))

    return -1


def parse_response_taskB(response_text):
    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
    }

    text = remove_think_block(response_text).strip()

    if not text:
        return -1

    matches = re.findall(r"\bthe answer is\s*([ABCabc])\b", text, re.IGNORECASE)
    if matches:
        return mapping[matches[-1].upper()]

    match = re.search(r"\banswer\s*(?:is|:|：)?\s*([ABCabc])\b", text, re.IGNORECASE)
    if match:
        return mapping[match.group(1).upper()]

    match = re.match(r"^\s*([ABCabc])\b", text)
    if match:
        return mapping[match.group(1).upper()]

    return -1


# ======== Result Saving ========

def save_results(results, meta, out_dir):
    ensure_dir(out_dir)

    model_tag = safe_name(meta["model"])
    shuffle_tag = "_shuffled" if meta.get("shuffle") else ""

    base_name = (
        f"llm_task{meta['task']}_"
        f"{meta['shot']}shot_"
        f"{model_tag}_"
        f"{meta.get('eval_tag', 'test')}"
        f"{shuffle_tag}"
    )

    json_path = os.path.join(out_dir, base_name + ".json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": meta["task"],
                "model": meta["model"],
                "model_hf": meta["model_hf"],
                "shot": meta["shot"],
                "shuffle": meta.get("shuffle", False),
                "shuffle_seed": meta.get("shuffle_seed"),
                "eval_tag": meta.get("eval_tag", "test"),
                "data_path": meta.get("data_path"),
                "answer_path": meta.get("answer_path"),
                "accuracy": meta["accuracy"],
                "total": meta["total"],
                "scored_total": meta["scored_total"],
                "correct": meta["correct_count"],
                "parse_failures": meta["parse_failures"],
                "elapsed_seconds": meta.get("elapsed", 0),
                "predictions": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"JSON saved to: {json_path}")

    csv_path = os.path.join(out_dir, base_name + ".csv")

    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "gold",
                "pred",
                "correct",
                "shuffled",
                "perm",
                "raw_response",
            ],
            quoting=csv.QUOTE_ALL,
        )

        writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "id": r["id"],
                    "gold": r["gold"],
                    "pred": r["pred"],
                    "correct": r["correct"],
                    "shuffled": r.get("shuffled", ""),
                    "perm": r.get("perm", ""),
                    "raw_response": r["raw_response"],
                }
            )

    print(f"CSV saved to: {csv_path}")


# ======== Main Evaluation Loop ========

def run_eval(
    task,
    model_name,
    shot,
    cfg,
    limit=None,
    shuffle=False,
    shuffle_seed=42,
    load_in_4bit=True,
    data_path=None,
    answer_path=None,
    eval_tag="test",
    output_dir=None,
    max_new_tokens=512,
    max_input_length=2048,
):
    model_hf = resolve_model_name(model_name)

    print(f"Model alias: {model_name} -> {model_hf}")

    if task == "A":
        actual_data_path = data_path if data_path is not None else cfg["taskA"]["test_data"]
        actual_answer_path = answer_path if answer_path is not None else cfg["taskA"].get("test_answer")

        actual_data_path = resolve_project_path(actual_data_path, must_exist=True)
        actual_answer_path = resolve_project_path(actual_answer_path, must_exist=True)

        samples = load_taskA_samples(
            data_path=actual_data_path,
            answer_path=actual_answer_path,
            require_labels=False,
        )

        if shuffle:
            print("Applying sentence order shuffle for Task A.")
            samples = shuffle_task_a(samples, seed=shuffle_seed)

        make_prompt = lambda s: make_prompt_taskA(s, shot)
        parse_fn = parse_response_taskA

    elif task == "B":
        actual_data_path = data_path if data_path is not None else cfg["taskB"]["test_data"]
        actual_answer_path = answer_path if answer_path is not None else cfg["taskB"].get("test_answer")

        actual_data_path = resolve_project_path(actual_data_path, must_exist=True)
        actual_answer_path = resolve_project_path(actual_answer_path, must_exist=True)

        samples = load_taskB_samples(
            data_path=actual_data_path,
            answer_path=actual_answer_path,
            require_labels=False,
        )

        if shuffle:
            print("Applying option shuffle for Task B.")
            samples = shuffle_task_b(samples, seed=shuffle_seed)

        make_prompt = lambda s: make_prompt_taskB(s, shot)
        parse_fn = parse_response_taskB

    else:
        raise ValueError(f"Unknown task: {task}")

    if limit is not None:
        samples = samples[:limit]
        print(f"Debug mode: only running first {limit} samples.")

    if output_dir is None:
        output_dir = cfg.get("llm_prompt", {}).get("output_dir", "outputs/llm_prompt")

    output_dir = resolve_project_path(output_dir, must_exist=False)

    print("\n" + "=" * 60)
    print(f"Running {shot}-shot LLM prompting")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Model: {model_hf}")
    print(f"Data: {actual_data_path}")
    print(f"Answer: {actual_answer_path}")
    print(f"Eval tag: {eval_tag}")
    print(f"Total samples: {len(samples)}")
    print(f"Shuffle: {shuffle}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    model, tokenizer = load_model(model_hf, load_in_4bit=load_in_4bit)

    results = []
    correct_count = 0
    scored_total = 0
    parse_failures = 0
    start_time = time.time()

    for i, sample in enumerate(tqdm(samples, desc=f"Task {task} {shot}-shot")):
        prompt = make_prompt(sample)

        try:
            raw_response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                model_id=model_hf,
                max_new_tokens=max_new_tokens,
                max_input_length=max_input_length,
            )
        except Exception as e:
            print(f"\nGeneration failed for sample {sample['id']}: {e}")
            raw_response = ""

        pred = parse_fn(raw_response)
        gold = sample["label"]

        if pred == -1:
            parse_failures += 1

        if is_valid_label(gold, task):
            correct = pred == gold
            scored_total += 1

            if correct:
                correct_count += 1
        else:
            correct = None

        results.append(
            {
                "id": sample["id"],
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "shuffled": sample.get("_shuffled", ""),
                "perm": sample.get("_perm", ""),
                "raw_response": raw_response,
            }
        )

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)

            if scored_total > 0:
                current_acc = correct_count / scored_total
                acc_text = f"{current_acc:.4f}"
            else:
                acc_text = "N/A"

            gpu_mem = 0.0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3

            print(
                f"\n{i + 1}/{len(samples)} | "
                f"Acc: {acc_text} | "
                f"Avg: {avg_time:.1f}s/sample | "
                f"GPU: {gpu_mem:.1f}GB"
            )

    elapsed = time.time() - start_time

    if scored_total > 0:
        acc = correct_count / scored_total
    else:
        acc = None

    print("\n" + "=" * 60)
    print(f"Task {task} | {shot}-shot | Model: {model_hf}")
    print(f"Eval tag: {eval_tag}")
    print(f"Shuffle: {shuffle}")

    if acc is not None:
        print(f"Accuracy: {acc:.4f} ({correct_count}/{scored_total})")
    else:
        print("Accuracy: N/A, because no valid gold labels were provided.")

    print(f"Total samples: {len(samples)}")
    print(f"Parse failures: {parse_failures}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if len(samples) > 0:
        print(f"Avg per sample: {elapsed / len(samples):.2f}s")

    print("=" * 60)

    meta = {
        "task": task,
        "model": model_name,
        "model_hf": model_hf,
        "shot": shot,
        "shuffle": shuffle,
        "shuffle_seed": shuffle_seed,
        "eval_tag": eval_tag,
        "data_path": actual_data_path,
        "answer_path": actual_answer_path,
        "accuracy": acc,
        "total": len(samples),
        "scored_total": scored_total,
        "correct_count": correct_count,
        "parse_failures": parse_failures,
        "elapsed": round(elapsed, 2),
    }

    save_results(results, meta, out_dir=output_dir)

    print(f"\n{'id':>8} | {'gold':>5} | {'pred':>5} | {'ok':>5} | response preview")
    print("-" * 90)

    for r in results[:7]:
        resp_preview = r["raw_response"][:100].replace("\n", " ")
        print(
            f"{str(r['id']):>8} | "
            f"{str(r['gold']):>5} | "
            f"{str(r['pred']):>5} | "
            f"{str(r['correct']):>5} | "
            f"{resp_preview}"
        )

    del model, tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nModel unloaded, GPU memory released.")

    return acc


# ======== Entry Point ========

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Prompting for Task A/B")

    parser.add_argument("--task", type=str, required=True, choices=["A", "B"])

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name, e.g. qwen3:8b / Qwen/Qwen3-8B / llama3.1:8b",
    )

    parser.add_argument("--shot", type=str, required=True, choices=["zero", "one"])

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SRC_DIR, "config.yaml"),
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override test data path",
    )

    parser.add_argument(
        "--answer",
        type=str,
        default=None,
        help="Override answer/gold label path",
    )

    parser.add_argument(
        "--eval_tag",
        type=str,
        default="test",
        help="Tag used in output filename, e.g. test, shuffle, t5_original, t5_para",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate first N samples for debugging",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Apply sentence/option shuffle for robustness evaluation",
    )

    parser.add_argument("--shuffle_seed", type=int, default=42)

    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models like Llama",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens per sample",
    )

    parser.add_argument(
        "--max_input_length",
        type=int,
        default=2048,
        help="Maximum input token length",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("HF_TOKEN set.")

    run_eval(
        task=args.task,
        model_name=args.model,
        shot=args.shot,
        cfg=cfg,
        limit=args.limit,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        load_in_4bit=not args.no_4bit,
        data_path=args.data,
        answer_path=args.answer,
        eval_tag=args.eval_tag,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
    )


if __name__ == "__main__":
    main()
