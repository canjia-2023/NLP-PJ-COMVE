# Commonsense Validation and Explanation

This project studies commonsense validation and explanation using the SemEval-2020 Task 4 ComVE benchmark. It compares four approaches:

1. **Supervised Transformer fine-tuning** with BERT/RoBERTa;
2. **Zero-shot and one-shot LLM prompting**;
3. **Unsupervised language-model scoring** with masked-LM scores and causal-LM perplexity;
4. **Robustness evaluation** under sentence/option shuffling and T5-based paraphrasing.

The project is based on the commonsense reasoning tasks introduced by Wang et al. (2019) and the SemEval-2020 Task 4 benchmark.

## Tasks

### Task A — Commonsense Validation

Each instance contains two sentences. Exactly one violates common sense.

| Sentence 0 | Sentence 1 | Label |
|---|---|---|
| She put the milk in the refrigerator. | She put the milk in the volcano. | `1` |

For Task A, the label is the index of the nonsensical sentence:

- `0`: Sentence 0 is nonsensical.
- `1`: Sentence 1 is nonsensical.

### Task B — Commonsense Explanation

Each instance contains one nonsensical sentence and three candidate explanations. The goal is to select the best explanation.

| Field | Example |
|---|---|
| False sentence | She put the milk in the volcano. |
| Option A | Milk is too heavy to carry. |
| Option B | A volcano is extremely hot and would destroy the milk. |
| Option C | Milk should be stored in a bag. |
| Gold label | `1` / `B` |

For Task B, labels are mapped as follows:

- `0`: Option A
- `1`: Option B
- `2`: Option C

## Project Structure

```text
project/
├── README.md
├── environment.yml
├── dataset/
│   └── ALL_data/
│       ├── Training_Data/
│       ├── Dev_Data/
│       └── Test_Data/
├── outputs/
│   ├── finetune_bert/
│   ├── llm_prompt/
│   ├── llm_scoring/
│   └── robustness/
└── src/
    ├── README.md
    ├── config.yaml
    ├── data_utils.py
    ├── dataset.py
    ├── utils.py
    ├── finetune_bert/
    │   ├── train_taskA.py
    │   ├── train_taskB.py
    │   └── evaluate.py
    ├── llm_prompt/
    │   └── run_llm_prompt.py
    ├── llm_scoring/
    │   ├── run_bert_score.py
    │   └── run_ppl.py
    ├── robustness/
    │   ├── shuffle_data.py
    │   └── paraphrase_t5.py
    └── error_analysis/
        ├── prepare_error_analysis.py
        └── summarize_manual_labels.py
```

See [`src/README.md`](src/README.md) for detailed commands, parameters, output files, and experiment workflows.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate comve
```

If required, install missing Python packages:

```bash
pip install torch transformers accelerate bitsandbytes pyyaml tqdm sentencepiece
```

GPU is recommended for fine-tuning, LLM prompting, and T5 paraphrase generation.


## Dataset

The dataset used in this project is downloaded from the original SemEval-2020 Task 4 ComVE repository:

[https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation)

The dataset is not included in this repository because it is an external benchmark resource. After downloading the dataset, place the `ALL_data/` directory under the project-level `dataset/` folder:

```text
project/
└── dataset/
    └── ALL_data/
        ├── Training_Data/
        ├── Dev_Data/
        └── Test_Data/
```

The expected paths are configured in `src/config.yaml`. For example:

```yaml
taskA:
  train_data: "dataset/ALL_data/Training_Data/subtaskA_data_all.csv"
  train_answer: "dataset/ALL_data/Training_Data/subtaskA_answers_all.csv"
  dev_data: "dataset/ALL_data/Dev_Data/subtaskA_dev_data.csv"
  dev_answer: "dataset/ALL_data/Dev_Data/subtaskA_gold_answers.csv"
  test_data: "dataset/ALL_data/Test_Data/subtaskA_test_data.csv"
  test_answer: "dataset/ALL_data/Test_Data/subtaskA_gold_answers.csv"
```

Since `dataset/` is ignored by `.gitignore`, each user should download the dataset locally before running experiments.


## Configuration

All dataset paths, model settings, training hyperparameters, and output directories are configured in:

```text
src/config.yaml
```

Dataset paths should be written relative to the project root.

## Quick Start

Run all commands from the project root.

### 1. Train supervised models

```bash
python src/finetune_bert/train_taskA.py
python src/finetune_bert/train_taskB.py
```

The best checkpoints are saved under the configured output directory, typically:

```text
outputs/finetune_bert/
├── best_taskA_<model-name>/
└── best_taskB_<model-name>/
```

### 2. Evaluate fine-tuned models

Evaluate both tasks on the default test split:

```bash
python src/finetune_bert/evaluate.py \
  --task all \
  --config src/config.yaml \
  --eval_tag test
```

Evaluate a single task:

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --config src/config.yaml \
  --eval_tag test
```

Prediction files are saved to the fine-tuning output directory:

```text
preds_taskA_<model-name>_<eval-tag>.csv
preds_taskB_<model-name>_<eval-tag>.csv
```

### 3. Run LLM prompting

Zero-shot prompting:

```bash
python src/llm_prompt/run_llm_prompt.py \
  --task A \
  --model qwen3:8b \
  --shot zero \
  --config src/config.yaml
```

One-shot prompting:

```bash
python src/llm_prompt/run_llm_prompt.py \
  --task B \
  --model qwen3:8b \
  --shot one \
  --config src/config.yaml
```

Available aliases include:

```text
qwen3:1.7b
qwen3:4b
qwen3:8b
llama3.1:8b
olmo2:7b
```

### 4. Run language-model scoring baselines

Masked-LM scoring with BERT:

```bash
python src/llm_scoring/run_bert_score.py \
  --task both \
  --split dev \
  --model_name bert-large-uncased \
  --scoring_mode official \
  --config src/config.yaml
```

Causal-LM perplexity scoring:

```bash
python src/llm_scoring/run_ppl.py \
  --task both \
  --split dev \
  --model_name distilgpt2 \
  --config src/config.yaml
```

### 5. Generate shuffled robustness datasets

```bash
python src/robustness/shuffle_data.py \
  --task both \
  --split test \
  --seed 42 \
  --config src/config.yaml \
  --output_dir outputs/robustness/shuffle/test_seed42
```

### 6. Generate T5 paraphrase robustness datasets

Generate a stratified subset of 50 examples per task:

```bash
python src/robustness/paraphrase_t5.py \
  --task both \
  --split dev \
  --sample_size 50 \
  --seed 42 \
  --config src/config.yaml
```

Generate paraphrases for all examples:

```bash
python src/robustness/paraphrase_t5.py \
  --task both \
  --split test \
  --sample_size 0 \
  --seed 42 \
  --config src/config.yaml
```

## Evaluation Variants

The same evaluation script supports original, shuffled, paraphrased, and custom CSV files.

Example: evaluate a shuffled Task A dataset.

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --data outputs/robustness/shuffle/test_seed42/subtaskA_test_data_shuffled_seed42.csv \
  --answer outputs/robustness/shuffle/test_seed42/subtaskA_test_answers_shuffled_seed42.csv \
  --eval_tag shuffled_seed42 \
  --config src/config.yaml
```

Example: evaluate a T5-paraphrased Task A subset.

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --data outputs/robustness/paraphrase/dev_seed42/subtaskA_dev_t5_para_seed42_n50_data.csv \
  --answer outputs/robustness/paraphrase/dev_seed42/subtaskA_dev_t5_para_seed42_n50_answers.csv \
  --eval_tag t5_para_seed42_n50 \
  --config src/config.yaml
```

## Main Approaches

| Approach | Task A Decision Rule | Task B Decision Rule |
|---|---|---|
| Fine-tuned Transformer | Binary classifier prediction | Multiple-choice prediction |
| LLM prompting | Parse generated `0` or `1` | Parse generated `A`, `B`, or `C` |
| Causal-LM PPL | Higher PPL = nonsensical | Lower PPL = better explanation |
| BERT LM score | Lower naturalness score = nonsensical | Higher naturalness score = better explanation |

## Robustness Evaluation

Two robustness settings are included.

### Shuffle Robustness

- **Task A:** Sentence 0 and Sentence 1 are swapped with 50% probability, and the gold label is flipped accordingly.
- **Task B:** Options A/B/C are randomly permuted, and the gold label is remapped to the new option position.

### T5 Paraphrase Robustness

- **Task A:** Both sentences are paraphrased.
- **Task B:** The nonsensical sentence is paraphrased while the three explanation options remain unchanged.
- Labels are preserved.
- The script internally generates several T5 outputs and directly retains one valid paraphrase; it does not create candidate-review files.

## Error Analysis

After generating prediction files, run the automatic error-analysis pipeline:

```bash
python src/error_analysis/prepare_error_analysis.py \
  --config src/config.yaml \
  --split test \
  --output_dir outputs/error_analysis
```

This script scans prediction CSV files from:

```text
outputs/finetune_bert/
outputs/llm_prompt/
outputs/llm_scoring/
outputs/
```

It supports prediction files with either:

```text
id,gold,pred,correct
```

or:

```text
id,gold,prediction
```

The generated files are:

```text
outputs/error_analysis/
├── model_summary.csv
├── all_errors_taskA.csv
├── all_errors_taskB.csv
├── hard_cases_taskA.csv
├── hard_cases_taskB.csv
├── manual_taxonomy_taskA_roberta_large_original.csv
├── manual_taxonomy_taskB_roberta_large_original.csv
├── manual_taxonomy_taskA_hard_cases_original.csv
├── manual_taxonomy_taskB_hard_cases_original.csv
├── error_analysis_meta.json
└── error_analysis_note.md
```

The four `manual_taxonomy_*.csv` files are designed for manual annotation. Fill in:

```text
error_type
secondary_error_type
is_dataset_noise
notes
```

Recommended `error_type` values:

```text
physical
object_affordance
social
temporal
causal
numerical
lexical_distractor
ambiguity
dataset_noise
other
```

After manual annotation, summarize the labels:

```bash
python src/error_analysis/summarize_manual_labels.py \
  --input_dir outputs/error_analysis \
  --output_dir outputs/error_analysis/manual_summary
```

This generates:

```text
outputs/error_analysis/manual_summary/
├── manual_error_labels_normalized.csv
├── annotation_coverage.csv
├── error_type_distribution.csv
├── representative_error_examples.csv
├── manual_error_analysis_summary.md
└── fig_error_type_distribution.png
```

The final figure `fig_error_type_distribution.png` visualizes the distribution of manually annotated error categories.



## References

- Wang, C., et al. (2019). *Does It Make Sense? And Why? A Pilot Study for Sense Making and Explanation*. ACL 2019.
- Wang, C., et al. (2020). *SemEval-2020 Task 4: Commonsense Validation and Explanation*. SemEval 2020.
- Official dataset repository: [SemEval2020-Task4-Commonsense-Validation-and-Explanation](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation)
