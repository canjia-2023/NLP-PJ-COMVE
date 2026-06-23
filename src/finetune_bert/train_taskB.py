# train_taskB.py

import os
import sys
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup,
)

# 让当前脚本可以正常 import src 下面的公共文件
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from dataset import TaskBDataset
from utils import set_seed, get_device, ensure_dir, get_logger


def to_project_path(path):
    """把相对路径转换成相对于项目根目录的路径。"""
    if path is None:
        return None

    if os.path.isabs(path):
        return path

    return os.path.join(PROJECT_DIR, path)


def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            kwargs = {
                "input_ids": batch["input_ids"].to(device),              # (B, 3, L)
                "attention_mask": batch["attention_mask"].to(device),    # (B, 3, L)
            }

            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train(args):
    set_seed(args.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    model.to(device)

    model_short_name = args.model_name.split("/")[-1]
    save_dir = os.path.join(args.output_dir, f"best_taskB_{model_short_name}")
    ensure_dir(save_dir)

    logger = get_logger(
        name="train_taskB",
        log_file=os.path.join(save_dir, "train.log"),
    )

    logger.info(f"Model: {args.model_name}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Save dir: {save_dir}")

    # Dataset & DataLoader
    train_dataset = TaskBDataset(
        data_path=args.train_data,
        answer_path=args.train_answer,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    dev_dataset = TaskBDataset(
        data_path=args.dev_data,
        answer_path=args.dev_answer,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Optimizer & Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training Loop
    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }

            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            optimizer.zero_grad()

            outputs = model(**kwargs)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % args.log_steps == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        dev_acc = evaluate(model, dev_loader, device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Dev Acc: {dev_acc:.4f}"
        )

        if dev_acc > best_acc:
            best_acc = dev_acc

            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            logger.info(f"Best model saved. Dev Acc = {best_acc:.4f}")

    logger.info(f"Task B Best Dev Accuracy: {best_acc:.4f}")


def load_args_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    args = argparse.Namespace(
        seed=cfg.get("seed", 42),
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],

        train_data=to_project_path(cfg["taskB"]["train_data"]),
        train_answer=to_project_path(cfg["taskB"]["train_answer"]),
        dev_data=to_project_path(cfg["taskB"]["dev_data"]),
        dev_answer=to_project_path(cfg["taskB"]["dev_answer"]),

        output_dir=to_project_path(cfg["output_dir"]),

        batch_size=cfg["taskB"]["batch_size"],
        lr=float(cfg["taskB"]["lr"]),
        epochs=cfg["taskB"]["epochs"],

        weight_decay=cfg["taskB"].get("weight_decay", 0.01),
        warmup_ratio=cfg["taskB"].get("warmup_ratio", 0.1),
        max_grad_norm=cfg["taskB"].get("max_grad_norm", 1.0),
        log_steps=cfg["taskB"].get("log_steps", 50),
    )

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SRC_DIR, "config.yaml"),
        help="Path to config.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_args = load_args_from_config(cli_args.config)
    train(train_args)
