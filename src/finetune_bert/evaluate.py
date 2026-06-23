# evaluate.py

import os
import sys
import csv
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
)

# 让当前脚本可以正常 import src 下面的公共文件
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from dataset import TaskADataset, TaskBDataset
from utils import set_seed, get_device, ensure_dir


def to_project_path(path):
    """把相对路径转换成相对于项目根目录的路径。"""
    if path is None:
        return None

    if os.path.isabs(path):
        return path

    return os.path.join(PROJECT_DIR, path)


def evaluate_taskA(args):
    device = get_device()

    model_short_name = args.model_name.split("/")[-1]
    model_path = os.path.join(args.output_dir, f"best_taskA_{model_short_name}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Task A model not found: {model_path}")

    print(f"Loading Task A model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_dataset = TaskADataset(
        data_path=args.test_data,
        answer_path=args.test_answer,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    has_answer = args.test_answer is not None

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())

            if has_answer:
                labels = batch["labels"].to(device)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().tolist())

    if has_answer:
        acc = correct / total
        print(f"Task A Accuracy: {acc:.4f} ({correct}/{total})")
    else:
        acc = None
        print("Task A prediction finished. No answer file provided, so accuracy is not computed.")

    # 保存预测结果
    ensure_dir(args.output_dir)

    pred_file = os.path.join(
        args.output_dir,
        f"preds_taskA_{model_short_name}_{args.eval_tag}.csv",
    )

    with open(pred_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if has_answer:
            writer.writerow(["id", "gold", "pred", "correct"])

            for i, pred in enumerate(all_preds):
                gold = all_labels[i]
                sample_id = test_dataset.samples[i]["id"]
                writer.writerow([sample_id, gold, pred, int(pred == gold)])
        else:
            writer.writerow(["id", "pred"])

            for i, pred in enumerate(all_preds):
                sample_id = test_dataset.samples[i]["id"]
                writer.writerow([sample_id, pred])

    print(f"Predictions saved to: {pred_file}")

    return acc, all_preds, all_labels


def evaluate_taskB(args):
    device = get_device()

    model_short_name = args.model_name.split("/")[-1]
    model_path = os.path.join(args.output_dir, f"best_taskB_{model_short_name}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Task B model not found: {model_path}")

    print(f"Loading Task B model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMultipleChoice.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_dataset = TaskBDataset(
        data_path=args.test_data,
        answer_path=args.test_answer,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    has_answer = args.test_answer is not None

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())

            if has_answer:
                labels = batch["labels"].to(device)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().tolist())

    if has_answer:
        acc = correct / total
        print(f"Task B Accuracy: {acc:.4f} ({correct}/{total})")
    else:
        acc = None
        print("Task B prediction finished. No answer file provided, so accuracy is not computed.")

    # 保存预测结果
    ensure_dir(args.output_dir)

    pred_file = os.path.join(
        args.output_dir,
        f"preds_taskB_{model_short_name}_{args.eval_tag}.csv",
    )

    with open(pred_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if has_answer:
            writer.writerow(["id", "gold", "pred", "correct"])

            for i, pred in enumerate(all_preds):
                gold = all_labels[i]
                sample_id = test_dataset.samples[i]["id"]
                writer.writerow([sample_id, gold, pred, int(pred == gold)])
        else:
            writer.writerow(["id", "pred"])

            for i, pred in enumerate(all_preds):
                sample_id = test_dataset.samples[i]["id"]
                writer.writerow([sample_id, pred])

    print(f"Predictions saved to: {pred_file}")

    return acc, all_preds, all_labels


def load_config_args(cli_args):
    config_path = to_project_path(cli_args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cli_args.model_name if cli_args.model_name is not None else cfg["model_name"]

    argsA = argparse.Namespace(
        model_name=model_name,
        max_length=cfg["max_length"],
        output_dir=to_project_path(cfg["output_dir"]),

        test_data=to_project_path(
            cli_args.taskA_data
            if cli_args.taskA_data is not None
            else cfg["taskA"].get("test_data")
        ),
        test_answer=to_project_path(
            cli_args.taskA_answer
            if cli_args.taskA_answer is not None
            else cfg["taskA"].get("test_answer")
        ),

        batch_size=cfg["taskA"]["batch_size"],
        eval_tag=cli_args.eval_tag,
    )

    argsB = argparse.Namespace(
        model_name=model_name,
        max_length=cfg["max_length"],
        output_dir=to_project_path(cfg["output_dir"]),

        test_data=to_project_path(
            cli_args.taskB_data
            if cli_args.taskB_data is not None
            else cfg["taskB"].get("test_data")
        ),
        test_answer=to_project_path(
            cli_args.taskB_answer
            if cli_args.taskB_answer is not None
            else cfg["taskB"].get("test_answer")
        ),

        batch_size=cfg["taskB"]["batch_size"],
        eval_tag=cli_args.eval_tag,
    )

    return model_name, argsA, argsB


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SRC_DIR, "config.yaml"),
        help="Path to config.yaml",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model_name in config.yaml",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["A", "B", "all"],
        default="all",
        help="Which task to evaluate",
    )

    parser.add_argument(
        "--eval_tag",
        type=str,
        default="test",
        help="Tag used in prediction file name",
    )

    parser.add_argument("--taskA_data", type=str, default=None)
    parser.add_argument("--taskA_answer", type=str, default=None)
    parser.add_argument("--taskB_data", type=str, default=None)
    parser.add_argument("--taskB_answer", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)

    cli_args = parse_args()
    model_name, argsA, argsB = load_config_args(cli_args)

    acc_a = None
    acc_b = None

    if cli_args.task in ["A", "all"]:
        print("=" * 50)
        print("Evaluating Task A")
        print("=" * 50)
        print(f"Task A data:   {argsA.test_data}")
        print(f"Task A answer: {argsA.test_answer}")

        acc_a, _, _ = evaluate_taskA(argsA)

    if cli_args.task in ["B", "all"]:
        print("\n" + "=" * 50)
        print("Evaluating Task B")
        print("=" * 50)
        print(f"Task B data:   {argsB.test_data}")
        print(f"Task B answer: {argsB.test_answer}")

        acc_b, _, _ = evaluate_taskB(argsB)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Model: {model_name.split('/')[-1]}")

    if acc_a is not None:
        print(f"Task A Acc: {acc_a:.4f}")
    else:
        if cli_args.task in ["A", "all"]:
            print("Task A Acc: N/A")

    if acc_b is not None:
        print(f"Task B Acc: {acc_b:.4f}")
    else:
        if cli_args.task in ["B", "all"]:
            print("Task B Acc: N/A")
