# train_taskB.py
import os
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup,
)
from dataset import TaskBDataset
from utils import set_seed, get_device


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"].to(device)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(args):
    set_seed(42)
    device = get_device()

    # --- Tokenizer & Model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name).to(device)

    print(f"Model: {args.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Dataset & DataLoader ---
    train_dataset = TaskBDataset(
        args.train_data, args.train_answer, tokenizer, args.max_length
    )
    dev_dataset = TaskBDataset(
        args.dev_data, args.dev_answer, tokenizer, args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # --- Optimizer & Scheduler ---
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # --- 保存目录 ---
    model_short_name = args.model_name.split("/")[-1]
    save_dir = os.path.join(args.output_dir, f"best_taskB_{model_short_name}")

    # --- Training Loop ---
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": batch["labels"].to(device),
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        dev_acc = evaluate(model, dev_loader, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}"
        )

        if dev_acc > best_acc:
            best_acc = dev_acc
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✅ Best model saved to {save_dir} (acc={best_acc:.4f})")

    print(f"\n🎯 Task B Best Dev Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    args = argparse.Namespace(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        output_dir=cfg["output_dir"],
        train_data=cfg["taskB"]["train_data"],
        train_answer=cfg["taskB"]["train_answer"],
        dev_data=cfg["taskB"]["dev_data"],
        dev_answer=cfg["taskB"]["dev_answer"],
        batch_size=cfg["taskB"]["batch_size"],
        lr=float(cfg["taskB"]["lr"]),
        epochs=cfg["taskB"]["epochs"],
    )
    train(args)