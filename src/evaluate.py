# evaluate.py
import os
import csv
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMultipleChoice
from dataset import TaskADataset, TaskBDataset
from utils import set_seed,get_device


def evaluate_taskA(args):
    device = get_device()

    model_short_name = args.model_name.split("/")[-1]
    model_path = os.path.join(args.output_dir, f"best_taskA_{model_short_name}")

    print(f"Loading Task A model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    test_dataset = TaskADataset(
        args.test_data, args.test_answer, tokenizer, args.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"].to(device)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    print(f"📊 Task A Test Accuracy: {acc:.4f} ({correct}/{total})")

    # 保存预测结果
    pred_file = os.path.join(args.output_dir, f"preds_taskA_{model_short_name}.csv")
    with open(pred_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gold", "pred", "correct"])
        for i, (p, l) in enumerate(zip(all_preds, all_labels)):
            writer.writerow([
                test_dataset.samples[i]["id"], l, p, int(p == l)
            ])
    print(f"💾 Predictions saved to: {pred_file}")

    return acc, all_preds, all_labels


def evaluate_taskB(args):
    device = get_device()
    
    model_short_name = args.model_name.split("/")[-1]
    model_path = os.path.join(args.output_dir, f"best_taskB_{model_short_name}")

    print(f"Loading Task B model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMultipleChoice.from_pretrained(model_path).to(device)
    model.eval()

    test_dataset = TaskBDataset(
        args.test_data, args.test_answer, tokenizer, args.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"].to(device)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    print(f"📊 Task B Test Accuracy: {acc:.4f} ({correct}/{total})")

    # 保存预测结果
    pred_file = os.path.join(args.output_dir, f"preds_taskB_{model_short_name}.csv")
    with open(pred_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gold", "pred", "correct"])
        for i, (p, l) in enumerate(zip(all_preds, all_labels)):
            writer.writerow([
                test_dataset.samples[i]["id"], l, p, int(p == l)
            ])
    print(f"💾 Predictions saved to: {pred_file}")

    return acc, all_preds, all_labels


if __name__ == "__main__":
    set_seed(42)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 评估 Task A ----
    print("=" * 50)
    print("Evaluating Task A")
    print("=" * 50)
    argsA = argparse.Namespace(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        output_dir=cfg["output_dir"],
        test_data=cfg["taskA"]["test_data"],
        test_answer=cfg["taskA"]["test_answer"],
        batch_size=cfg["taskA"]["batch_size"],
    )
    acc_a, _, _ = evaluate_taskA(argsA)

    # ---- 评估 Task B ----
    print("\n" + "=" * 50)
    print("Evaluating Task B")
    print("=" * 50)
    argsB = argparse.Namespace(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        output_dir=cfg["output_dir"],
        test_data=cfg["taskB"]["test_data"],
        test_answer=cfg["taskB"]["test_answer"],
        batch_size=cfg["taskB"]["batch_size"],
    )
    acc_b, _, _ = evaluate_taskB(argsB)

    # ---- 汇总 ----
    print("\n" + "=" * 50)
    print("📋 Summary")
    print("=" * 50)
    model_name = cfg["model_name"].split("/")[-1]
    print(f"Model:        {model_name}")
    print(f"Task A Acc:   {acc_a:.4f}")
    print(f"Task B Acc:   {acc_b:.4f}")
