# data_utils.py

import csv
import os
from typing import Dict, List, Optional, Tuple


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)


def resolve_project_path(path: Optional[str], must_exist: bool = False) -> Optional[str]:
    """
    将相对路径解析为项目根目录下的路径。
    """
    if path is None:
        return None

    if os.path.isabs(path):
        resolved = os.path.normpath(path)
    else:
        candidates = [
            os.path.join(PROJECT_DIR, path),
            os.path.join(os.getcwd(), path),
            os.path.join(SRC_DIR, path),
        ]

        resolved = None
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved = os.path.normpath(candidate)
                break

        if resolved is None:
            resolved = os.path.normpath(os.path.join(PROJECT_DIR, path))

    if must_exist and not os.path.exists(resolved):
        raise FileNotFoundError(f"File not found: {resolved}")

    return resolved


def clean_text(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def clean_id(value) -> str:
    """
    统一 id 格式，避免 1 和 1.0 之类的问题。
    """
    text = clean_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_field_name(name) -> str:
    if name is None:
        return ""
    return str(name).strip().lstrip("\ufeff")


def normalize_row(row: Dict) -> Dict:
    return {
        normalize_field_name(k): v
        for k, v in row.items()
    }


def get_first(row: Dict, keys: List[str], default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def parse_label_value(value, task: str) -> Optional[int]:
    """
    Task A:
        0 / 1

    Task B:
        A / B / C 或 0 / 1 / 2
    """
    if value is None:
        return None

    text = clean_text(value)

    if text == "":
        return None

    if text.lower() in {"label", "answer", "gold"}:
        return None

    if task.upper() == "B":
        label_map = {
            "A": 0,
            "B": 1,
            "C": 2,
            "a": 0,
            "b": 1,
            "c": 2,
        }

        if text in label_map:
            return label_map[text]

    # 兼容 "0.0" 这种情况
    return int(float(text))


def load_answer_file(answer_path: Optional[str], task: str) -> Dict[str, int]:
    """
    读取答案文件。

    支持:
        id,label
        id,answer
        无表头: 123,0
        Task B: 123,A / 123,0
    """
    labels = {}

    if answer_path is None:
        return labels

    answer_path = resolve_project_path(answer_path, must_exist=True)

    with open(answer_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if not row or len(row) < 2:
                continue

            sample_id = clean_id(row[0])
            raw_label = row[1]

            # 跳过表头
            if sample_id.lower() in {"id", "index"}:
                continue

            label = parse_label_value(raw_label, task)

            if label is None:
                # 只允许第一行是非标签内容，其余行如果解析不了，应当报错
                if row_idx == 0:
                    continue
                raise ValueError(
                    f"Cannot parse label in {answer_path}, row {row_idx + 1}: {row}"
                )

            labels[sample_id] = label

    return labels


def load_taskA_samples(
    data_path: str,
    answer_path: Optional[str] = None,
    require_labels: bool = False,
) -> List[Dict]:
    """
    读取 Task A 原始样本。

    返回格式:
        {
            "id": "...",
            "sentence0": "...",
            "sentence1": "...",
            "label": 0 or 1 or None
        }
    """
    data_path = resolve_project_path(data_path, must_exist=True)
    labels = load_answer_file(answer_path, task="A") if answer_path else {}

    samples = []

    with open(data_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row = normalize_row(row)

            sample_id = clean_id(get_first(row, ["id", "ID"], ""))

            label_raw = get_first(row, ["label", "Label", "answer", "Answer", "gold", "Gold"], None)
            label = parse_label_value(label_raw, task="A") if label_raw is not None else None

            if answer_path is not None:
                if sample_id not in labels:
                    if require_labels:
                        raise KeyError(
                            f"Missing label for sample id={sample_id} in answer file: {answer_path}"
                        )
                    label = -1
                else:
                    label = labels[sample_id]

            if require_labels and label is None:
                raise ValueError(f"Sample id={sample_id} has no label.")

            sample = {
                "id": sample_id,
                "sentence0": clean_text(
                    get_first(row, ["sent0", "sentence0", "Sentence0", "sentence_0"], "")
                ),
                "sentence1": clean_text(
                    get_first(row, ["sent1", "sentence1", "Sentence1", "sentence_1"], "")
                ),
                "label": label,
            }

            samples.append(sample)

    return samples


def load_taskB_samples(
    data_path: str,
    answer_path: Optional[str] = None,
    require_labels: bool = False,
) -> List[Dict]:
    """
    读取 Task B 原始样本。

    返回格式:
        {
            "id": "...",
            "question": "...",
            "optionA": "...",
            "optionB": "...",
            "optionC": "...",
            "label": 0 / 1 / 2 / None
        }
    """
    data_path = resolve_project_path(data_path, must_exist=True)
    labels = load_answer_file(answer_path, task="B") if answer_path else {}

    samples = []

    with open(data_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row = normalize_row(row)

            sample_id = clean_id(get_first(row, ["id", "ID"], ""))

            label_raw = get_first(row, ["label", "Label", "answer", "Answer", "gold", "Gold"], None)
            label = parse_label_value(label_raw, task="B") if label_raw is not None else None

            if answer_path is not None:
                if sample_id not in labels:
                    if require_labels:
                        raise KeyError(
                            f"Missing label for sample id={sample_id} in answer file: {answer_path}"
                        )
                    label = -1
                else:
                    label = labels[sample_id]

            if require_labels and label is None:
                raise ValueError(f"Sample id={sample_id} has no label.")

            sample = {
                "id": sample_id,
                "question": clean_text(
                    get_first(row, ["FalseSent", "false_sent", "question", "Question"], "")
                ),
                "optionA": clean_text(
                    get_first(row, ["OptionA", "optionA", "A", "reasonA"], "")
                ),
                "optionB": clean_text(
                    get_first(row, ["OptionB", "optionB", "B", "reasonB"], "")
                ),
                "optionC": clean_text(
                    get_first(row, ["OptionC", "optionC", "C", "reasonC"], "")
                ),
                "label": label,
            }

            samples.append(sample)

    return samples


def get_split_paths(cfg: Dict, task: str, split: str) -> Tuple[str, Optional[str]]:
    """
    从 config.yaml 中取某个 task 和 split 的 data / answer 路径。

    示例:
        get_split_paths(cfg, "A", "dev")
        get_split_paths(cfg, "B", "test")
    """
    task = task.upper()
    split = split.lower()

    if task not in {"A", "B"}:
        raise ValueError("task must be 'A' or 'B'.")

    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be 'train', 'dev', or 'test'.")

    section = cfg[f"task{task}"]

    data_key = f"{split}_data"
    answer_key = f"{split}_answer"

    data_path = section[data_key]
    answer_path = section.get(answer_key)

    return (
        resolve_project_path(data_path, must_exist=True),
        resolve_project_path(answer_path, must_exist=True) if answer_path else None,
    )
