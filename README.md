# Commonsense Validation and Explanation: From Perplexity-Based Scoring to Supervised Fine-Tuning

This project revisits the commonsense reasoning tasks from Wang et al. (ACL 2019) — "Does It Make Sense? And Why?" — and proposes a paradigm shift from unsupervised perplexity-based scoring to supervised fine-tuning of pre-trained Transformers.

## Task Description

We focus on two subtasks from SemEval-2020 Task 4:

### Task A — Commonsense Validation (Binary Classification)
Given two sentences with similar wording, identify which one violates common sense.

| Statement 1 | Statement 2 |
|---|---|
| He put a turkey into the fridge. ✅ | He put an elephant into the fridge. ❌ |

### Task B — Commonsense Explanation (Multiple Choice)
Given a nonsensical statement and three candidate reasons, select the correct explanation.

- **Statement:** He put an elephant into the fridge.
- **A:** An elephant is much bigger than a fridge. ✅
- **B:** Elephants are usually white while fridges are usually white.
- **C:** An elephant cannot eat a fridge.

## Dataset

We use the official [SemEval-2020 Task 4](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation) dataset (~10,000 instances per task) with standard train/dev/test splits, located at `dataset/ALL_data/`.

## Approach

### Stage 1: Perplexity-Based Baseline (Paper Reproduction)
Following Wang et al. (2019):
- **Task A:** Compute perplexity of both sentences under a frozen pre-trained LM and select the higher-PPL sentence as nonsensical.
- **Task B:** Concatenate the statement with each candidate explanation and select the lowest-PPL combination.

### Stage 2: Supervised Fine-Tuning (Our Contribution)
We reformulate both tasks as supervised classification:
- **Task A:** Encode the sentence pair with a Transformer and fine-tune end-to-end with a binary classification head.
- **Task B:** Encode each statement–explanation pair and fine-tune with a three-way multiple-choice head.

Models experimented with: **BERT-base-uncased**, **RoBERTa-base**, **RoBERTa-large**.

## Project Structure

```
code/
├── src/
│   ├── config.yaml          # Model & training hyperparameters
│   ├── dataset.py           # TaskADataset, TaskBDataset (data loading & tokenization)
│   ├── utils.py             # set_seed, get_device, EarlyStopping, get_logger
│   ├── train_taskA.py       # Training script for Task A (binary classification)
│   ├── train_taskB.py       # Training script for Task B (multiple choice)
│   ├── evaluate.py          # Test-set evaluation for both tasks
│   └── outputs/             # Saved model checkpoints
│       ├── best_taskA_bert-base-uncased/
│       ├── best_taskA_roberta-base/
│       ├── best_taskA_roberta-large/
│       ├── best_taskB/
│       ├── best_taskA/
│       └── best_taskB_roberta-large/
├── dataset/                 # Training/dev/test CSV files
├── data_explore.ipynb       # Exploratory data analysis notebook
├── environment.yml          # Conda environment specification
├── proposal.md              # Project proposal
└── README.md
```

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate comve

# Install PyTorch & Transformers (via pip)
pip install torch transformers pyyaml
```

## Usage

### Training

Configure model and hyperparameters in `src/config.yaml`, then run:

```bash
# Train Task A (binary classification)
cd src
python train_taskA.py

# Train Task B (multiple choice)
python train_taskB.py
```

Key configuration options in `config.yaml`:

| Parameter | Description |
|---|---|
| `model_name` | HuggingFace model ID (`bert-base-uncased`, `roberta-base`, `roberta-large`) |
| `max_length` | Maximum token length (default: 128) |
| `taskA.batch_size` | Batch size for Task A (default: 16) |
| `taskB.batch_size` | Batch size for Task B (default: 8) |
| `taskA.epochs` / `taskB.epochs` | Number of training epochs (default: 3–4) |
| `taskA.lr` / `taskB.lr` | Learning rate (default: 2e-5) |

### Evaluation

```bash
cd src
python evaluate.py
```

This loads the best checkpoint for each task and reports test accuracy. Predictions are saved to `src/outputs/preds_taskA_*.csv` and `src/outputs/preds_taskB_*.csv`.

### Data Exploration

Open `data_explore.ipynb` in Jupyter for dataset statistics and visualization.

## Model Architecture

| Task | HuggingFace Model Class | Output |
|---|---|---|
| Task A | `AutoModelForSequenceClassification` | 2-class logits (sensible / nonsensical) |
| Task B | `AutoModelForMultipleChoice` | 3-class logits (A / B / C) |

Both BERT and RoBERTa are supported — the dataset and training scripts automatically handle `token_type_ids` differences.

## Results

| Model | Task A (Dev) | Task B (Dev) |
|---|---|---|
| BERT-base-uncased | See outputs | See outputs |
| RoBERTa-base | See outputs | See outputs |
| RoBERTa-large | See outputs | See outputs |

*Final test set results are available after running `evaluate.py`.*

## References

- Wang et al. (2019). [Does It Make Sense? And Why? A Pilot Study for Sense Making and Explanation](https://arxiv.org/abs/1906.00363). ACL 2019.
- Wang et al. (2020). [SemEval-2020 Task 4: Commonsense Validation and Explanation](https://arxiv.org/abs/2007.00236). SemEval 2020.
- [Official Dataset Repository](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation)
