# dataset.py
import csv
import torch
from torch.utils.data import Dataset


class TaskADataset(Dataset):
    """
    Task A: 给定两句话，判断哪句违反常识 (二分类, label=0 or 1)
    BERT 输入:    [CLS] sent0 [SEP] sent1 [SEP]
    RoBERTa 输入: <s> sent0 </s></s> sent1 </s>
    """

    def __init__(self, data_path, answer_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 判断 tokenizer 是否支持 token_type_ids（RoBERTa 不支持）
        self.use_token_type_ids = hasattr(tokenizer, "create_token_type_ids_from_sequences")
        self.samples = []

        # 读取答案
        labels = {}
        if answer_path is not None:
            with open(answer_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    labels[row[0].strip()] = int(row[1].strip())

        # 读取数据
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    "id": row["id"].strip(),
                    "sent0": row["sent0"].strip(),
                    "sent1": row["sent1"].strip(),
                }
                if answer_path is not None:
                    sample["label"] = labels[sample["id"]]
                else:
                    sample["label"] = None
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["sent0"],
            sample["sent1"],
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
    Task B: 给定违反常识的句子 + 3个候选解释，选出最佳解释 (三选一)
    """

    LABEL_MAP = {"A": 0, "B": 1, "C": 2, "0": 0, "1": 1, "2": 2}

    def __init__(self, data_path, answer_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 读取答案
        labels = {}
        if answer_path is not None:
            with open(answer_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    raw_label = row[1].strip()
                    if raw_label in self.LABEL_MAP:
                        labels[row[0].strip()] = self.LABEL_MAP[raw_label]
                    else:
                        labels[row[0].strip()] = int(raw_label)

        # 读取数据
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    "id": row["id"].strip(),
                    "false_sent": row["FalseSent"].strip(),
                    "options": [
                        row["OptionA"].strip(),
                        row["OptionB"].strip(),
                        row["OptionC"].strip(),
                    ],
                }
                if answer_path is not None:
                    sample["label"] = labels[sample["id"]]
                else:
                    sample["label"] = None
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        has_token_type_ids = False

        for option in sample["options"]:
            encoding = self.tokenizer(
                sample["false_sent"],
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
                has_token_type_ids = True

        item = {
            "input_ids": torch.stack(input_ids_list),           # (3, L)
            "attention_mask": torch.stack(attention_mask_list),  # (3, L)
        }

        if has_token_type_ids:
            item["token_type_ids"] = torch.stack(token_type_ids_list)  # (3, L)

        if sample["label"] is not None:
            item["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return item