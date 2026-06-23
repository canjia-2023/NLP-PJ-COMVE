# dataset.py

import torch
from torch.utils.data import Dataset

from data_utils import load_taskA_samples, load_taskB_samples


class TaskADataset(Dataset):
    """
    Task A: 给定两句话，判断哪一句违反常识。

    本类只负责:
        原始样本 -> tokenizer 编码 -> PyTorch tensor

    原始 CSV 读取逻辑放在 data_utils.py。
    """

    def __init__(self, data_path, answer_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = load_taskA_samples(
            data_path=data_path,
            answer_path=answer_path,
            require_labels=answer_path is not None,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["sentence0"],
            sample["sentence1"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # BERT 有 token_type_ids，RoBERTa 没有
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if sample["label"] is not None:
            item["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return item


class TaskBDataset(Dataset):
    """
    Task B: 给定违反常识的句子和三个候选解释，选择最佳解释。

    本类只负责:
        原始样本 -> tokenizer 编码 -> PyTorch tensor

    原始 CSV 读取逻辑放在 data_utils.py。
    """

    def __init__(self, data_path, answer_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = load_taskB_samples(
            data_path=data_path,
            answer_path=answer_path,
            require_labels=answer_path is not None,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        options = [
            sample["optionA"],
            sample["optionB"],
            sample["optionC"],
        ]

        for option in options:
            encoding = self.tokenizer(
                sample["question"],
                option,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids_list.append(encoding["input_ids"].squeeze(0))
            attention_mask_list.append(encoding["attention_mask"].squeeze(0))

            if "token_type_ids" in encoding:
                token_type_ids_list.append(encoding["token_type_ids"].squeeze(0))

        item = {
            "input_ids": torch.stack(input_ids_list),            # (3, L)
            "attention_mask": torch.stack(attention_mask_list),  # (3, L)
        }

        # BERT 有 token_type_ids，RoBERTa 没有
        if len(token_type_ids_list) > 0:
            item["token_type_ids"] = torch.stack(token_type_ids_list)  # (3, L)

        if sample["label"] is not None:
            item["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return item
