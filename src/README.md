# Source Code Guide

This document describes the source modules, command-line arguments, expected outputs, and recommended experiment workflow.

Run all commands from the project root unless otherwise noted.

```bash
python src/<module>/<script>.py ...
```

## 1. Shared Modules

### `data_utils.py`

Centralized data loading and path handling.

Main responsibilities:

- Resolve project-relative paths;
- Load Task A and Task B CSV files;
- Load answer files with or without headers;
- Normalize IDs and field names;
- Convert Task B labels from `A/B/C` to `0/1/2`;
- Retrieve train/dev/test paths from `config.yaml`.

Main functions:

```python
resolve_project_path(path, must_exist=False)
load_taskA_samples(data_path, answer_path=None, require_labels=False)
load_taskB_samples(data_path, answer_path=None, require_labels=False)
get_split_paths(cfg, task, split)
```

All other modules should use these functions instead of implementing their own CSV readers.

### `dataset.py`

Converts raw samples into PyTorch datasets.

```python
TaskADataset
TaskBDataset
```

- `TaskADataset` tokenizes `(sentence0, sentence1)`.
- `TaskBDataset` tokenizes three `(question, option)` pairs.
- Both support BERT and RoBERTa by conditionally handling `token_type_ids`.

### `utils.py`

Shared utility functions:

```python
set_seed(...)
get_device(...)
ensure_dir(...)
```

## 2. Fine-Tuning

Source files:

```text
src/finetune_bert/
├── train_taskA.py
├── train_taskB.py
└── evaluate.py
```

### Train Task A

```bash
python src/finetune_bert/train_taskA.py
```

Task A is formulated as binary sequence-pair classification.

Input:

```text
Sentence 0 + Sentence 1
```

Output:

```text
0 or 1
```

### Train Task B

```bash
python src/finetune_bert/train_taskB.py
```

Task B is formulated as three-way multiple-choice prediction.

Input:

```text
(FalseSent, OptionA)
(FalseSent, OptionB)
(FalseSent, OptionC)
```

Output:

```text
0, 1, or 2
```

### Evaluate Fine-Tuned Checkpoints

Evaluate both tasks using default test data from `src/config.yaml`:

```bash
python src/finetune_bert/evaluate.py \
  --task all \
  --config src/config.yaml \
  --eval_tag test
```

Evaluate only Task A:

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --config src/config.yaml \
  --eval_tag test
```

Evaluate only Task B:

```bash
python src/finetune_bert/evaluate.py \
  --task B \
  --config src/config.yaml \
  --eval_tag test
```

Override the fine-tuning model name:

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --model_name roberta-large \
  --eval_tag roberta_large_test \
  --config src/config.yaml
```

Override batch size or token length:

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --batch_size 32 \
  --max_length 128 \
  --eval_tag batch32 \
  --config src/config.yaml
```

### Evaluate Custom Data

Use `--data` and `--answer` when evaluating one task.

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --data <custom_taskA_data.csv> \
  --answer <custom_taskA_answers.csv> \
  --eval_tag custom \
  --config src/config.yaml
```

Use the task-specific arguments when evaluating both tasks simultaneously:

```bash
python src/finetune_bert/evaluate.py \
  --task all \
  --taskA_data <taskA_data.csv> \
  --taskA_answer <taskA_answers.csv> \
  --taskB_data <taskB_data.csv> \
  --taskB_answer <taskB_answers.csv> \
  --eval_tag custom_all \
  --config src/config.yaml
```

Expected prediction output:

```text
outputs/finetune_bert/
├── preds_taskA_<model>_<eval_tag>.csv
└── preds_taskB_<model>_<eval_tag>.csv
```

Each scored prediction CSV contains:

```text
id,gold,pred,correct
```

## 3. LLM Prompting

Source file:

```text
src/llm_prompt/run_llm_prompt.py
```

### Zero-Shot Prompting

```bash
python src/llm_prompt/run_llm_prompt.py \
  --task A \
  --model qwen3:8b \
  --shot zero \
  --config src/config.yaml
```

### One-Shot Prompting

```bash
python src/llm_prompt/run_llm_prompt.py \
  --task B \
  --model qwen3:8b \
  --shot one \
  --config src/config.yaml
```

### Useful Arguments

```text
--task                 A or B
--model                Model alias or Hugging Face model ID
--shot                 zero or one
--limit                Evaluate only the first N examples
--shuffle              Apply in-memory shuffle robustness transformation
--shuffle_seed         Random seed for in-memory shuffle
--data                 Override evaluation data path
--answer               Override gold-label path
--eval_tag             Tag written into output filenames
--output_dir           Override output directory
--no_4bit              Disable 4-bit model quantization
--hf_token             Hugging Face token for gated checkpoints
--max_new_tokens       Maximum generated output length
--max_input_length     Maximum prompt token length
```

Debug example:

```bash
python src/llm_prompt/run_llm_prompt.py \
  --task A \
  --model qwen3:8b \
  --shot one \
  --limit 5 \
  --eval_tag debug5 \
  --config src/config.yaml
```

Outputs:

```text
outputs/llm_prompt/
├── llm_taskA_zeroshot_<model>_<eval_tag>.json
├── llm_taskA_zeroshot_<model>_<eval_tag>.csv
├── llm_taskB_oneshot_<model>_<eval_tag>.json
└── llm_taskB_oneshot_<model>_<eval_tag>.csv
```

The JSON output includes accuracy, parsing failures, elapsed time, metadata, prompts, and raw model responses.

## 4. Language-Model Scoring Baselines

Source files:

```text
src/llm_scoring/
├── run_bert_score.py
└── run_ppl.py
```

### BERT Masked-LM Score

```bash
python src/llm_scoring/run_bert_score.py \
  --task both \
  --split dev \
  --model_name bert-large-uncased \
  --scoring_mode official \
  --config src/config.yaml
```

Available scoring modes:

```text
official    Score original tokens directly from BERT output distributions.
masked      Use leave-one-out masked pseudo-likelihood.
```

Task A decision rule:

```text
Lower naturalness score → nonsensical sentence.
```

Task B decision rule:

```text
Higher naturalness score → best sentence–reason combination.
```

### Causal-LM Perplexity

```bash
python src/llm_scoring/run_ppl.py \
  --task both \
  --split dev \
  --model_name distilgpt2 \
  --config src/config.yaml
```

Task A decision rule:

```text
Higher PPL → nonsensical sentence.
```

Task B decision rule:

```text
Lower PPL → best sentence–reason combination.
```

Run only the first 20 Task A examples:

```bash
python src/llm_scoring/run_ppl.py \
  --task A \
  --split dev \
  --model_name distilgpt2 \
  --limit 20 \
  --config src/config.yaml
```

## 5. Shuffle Robustness

Source file:

```text
src/robustness/shuffle_data.py
```

Generate shuffled Task A and Task B test sets:

```bash
python src/robustness/shuffle_data.py \
  --task both \
  --split test \
  --seed 42 \
  --output_dir outputs/robustness/shuffle/test_seed42 \
  --config src/config.yaml
```

Task A transformation:

```text
50% probability of swapping Sentence 0 and Sentence 1.
If swapped, label 0 ↔ 1.
```

Task B transformation:

```text
Randomly permute Option A, Option B, and Option C.
Remap the gold label to the new option index.
```

Generated files include:

```text
*_data_shuffled_seed42.csv
*_answers_shuffled_seed42.csv
*_shuffle_map_seed42.csv
shuffle_meta_*.json
```

### Evaluate Shuffled Data

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --data outputs/robustness/shuffle/test_seed42/subtaskA_test_data_shuffled_seed42.csv \
  --answer outputs/robustness/shuffle/test_seed42/subtaskA_test_answers_shuffled_seed42.csv \
  --eval_tag shuffled_seed42 \
  --config src/config.yaml
```

Use the same shuffled CSV files for fine-tuned models, prompting models, BERT scores, and PPL scores to ensure a fair comparison.

## 6. T5 Paraphrase Robustness

Source file:

```text
src/robustness/paraphrase_t5.py
```

The script directly generates one usable paraphrase per target sentence. It does not save candidate pools or require manual candidate selection.

### Generate a Stratified Subset

```bash
python src/robustness/paraphrase_t5.py \
  --task both \
  --split dev \
  --sample_size 50 \
  --seed 42 \
  --config src/config.yaml
```

### Generate All Examples

Use `--sample_size 0` to paraphrase the complete split.

```bash
python src/robustness/paraphrase_t5.py \
  --task both \
  --split test \
  --sample_size 0 \
  --seed 42 \
  --config src/config.yaml
```

### Important Arguments

```text
--task                 A, B, or both
--split                train, dev, or test
--sample_size          Number of stratified examples; 0 means all
--seed                 Random seed
--model_name           T5 paraphrase checkpoint
--num_try              Internal generation attempts before fallback
--top_k                Top-k sampling parameter
--top_p                Nucleus sampling parameter
--temperature          Sampling temperature
--no_sample            Use deterministic decoding
--output_dir           Override output directory
--tag                  Identifier included in generated filenames
```

Task A transformation:

```text
Paraphrase Sentence 0 and Sentence 1.
Keep the original label unchanged.
```

Task B transformation:

```text
Paraphrase FalseSent only.
Keep Option A, Option B, Option C, and the gold label unchanged.
```

Generated files include:

```text
*_data.csv
*_answers.csv
*_original_subset_data.csv
*_map.csv
paraphrase_meta_*.json
```

The original subset file contains exactly the same selected instances as the paraphrased file. It should be used as the matched control condition.

### Evaluate T5 Paraphrase Data

Paraphrased condition:

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --data outputs/robustness/paraphrase/dev_seed42/subtaskA_dev_t5_para_seed42_n50_data.csv \
  --answer outputs/robustness/paraphrase/dev_seed42/subtaskA_dev_t5_para_seed42_n50_answers.csv \
  --eval_tag t5_para_seed42_n50 \
  --config src/config.yaml
```

Matched original-subset control:

```bash
python src/finetune_bert/evaluate.py \
  --task A \
  --data outputs/robustness/paraphrase/dev_seed42/subtaskA_dev_t5_para_seed42_n50_original_subset_data.csv \
  --answer outputs/robustness/paraphrase/dev_seed42/subtaskA_dev_t5_para_seed42_n50_answers.csv \
  --eval_tag t5_original_subset_seed42_n50 \
  --config src/config.yaml
```


## 7. Error Analysis

Source files:

```text
src/error_analysis/
├── __init__.py
├── prepare_error_analysis.py
└── summarize_manual_labels.py
```

The error-analysis pipeline has two stages.

### Stage 1: Prepare Automatic Error-Analysis Artifacts

Run:

```bash
python src/error_analysis/prepare_error_analysis.py \
  --config src/config.yaml \
  --split test \
  --output_dir outputs/error_analysis
```

This script:

```text
1. Loads Task A and Task B data from config.yaml.
2. Scans prediction CSV files.
3. Normalizes prediction columns.
4. Computes model-level accuracy and confusion directions.
5. Extracts all wrong predictions.
6. Finds cross-model hard cases.
7. Creates manual taxonomy templates.
```

Default prediction directories:

```text
outputs/finetune_bert
outputs/llm_prompt
outputs/llm_scoring
outputs
```

Custom prediction directories can be passed with `--pred_dirs`:

```bash
python src/error_analysis/prepare_error_analysis.py \
  --config src/config.yaml \
  --split test \
  --pred_dirs outputs/finetune_bert outputs/llm_prompt outputs/llm_scoring \
  --output_dir outputs/error_analysis
```

Prediction files should contain either:

```text
id,gold,pred,correct
```

or:

```text
id,gold,prediction
```

For Task B, labels may be either numeric or letter-based:

```text
0 / 1 / 2
A / B / C
```

Generated files:

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

### Model Summary

`model_summary.csv` contains:

```text
task
variant
model
status
total
correct
accuracy
n_errors
top_confusions
prediction_distribution
gold_mismatch_with_dataset
warning
source_prediction_file
```

This file is useful for checking whether a prediction file was loaded correctly and whether a model has suspicious behavior, such as predicting the same label for almost all examples.

### All Errors

`all_errors_taskA.csv` contains every wrong Task A prediction from every model.

Important fields:

```text
model
id
gold
pred
confusion
sent0
sent1
gold_nonsense_sentence
predicted_nonsense_sentence
sent_pair_overlap
source_prediction_file
```

`all_errors_taskB.csv` contains every wrong Task B prediction from every model.

Important fields:

```text
model
id
gold
pred
confusion
false_sent
OptionA
OptionB
OptionC
gold_option
predicted_option
max_gold_wrong_option_cosine
source_prediction_file
```

### Hard Cases

`hard_cases_taskA.csv` and `hard_cases_taskB.csv` contain examples that are wrong for all selected core models.

By default, the script tries to use:

```text
Task A:
- preds_taskA_bert-base-uncased_test
- preds_taskA_roberta-large_test
- llm_taskA_oneshot_Qwen_Qwen3-8B_test

Task B:
- preds_taskB_bert-base-uncased_test
- preds_taskB_roberta-large_test
- llm_taskB_oneshot_Qwen_Qwen3-8B_test
```

Custom core models can be specified as CSV stem names:

```bash
python src/error_analysis/prepare_error_analysis.py \
  --config src/config.yaml \
  --split test \
  --core_models_taskA preds_taskA_bert-base-uncased_test,preds_taskA_roberta-large_test,llm_taskA_oneshot_Qwen_Qwen3-8B_test \
  --core_models_taskB preds_taskB_bert-base-uncased_test,preds_taskB_roberta-large_test,llm_taskB_oneshot_Qwen_Qwen3-8B_test
```

### Manual Taxonomy Templates

The script creates four files for manual labeling:

```text
manual_taxonomy_taskA_roberta_large_original.csv
manual_taxonomy_taskB_roberta_large_original.csv
manual_taxonomy_taskA_hard_cases_original.csv
manual_taxonomy_taskB_hard_cases_original.csv
```

Fill in these columns manually:

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

Suggested meanings:

| Error Type | Meaning |
|---|---|
| `physical` | Size, weight, space, material, body, or physical feasibility |
| `object_affordance` | What an object can or cannot be used for |
| `social` | Human behavior, social norms, roles, intentions |
| `temporal` | Time order, duration, frequency, before/after contradictions |
| `causal` | Implausible, missing, or reversed cause-effect relation |
| `numerical` | Quantity, scale, age, distance, money, or count |
| `lexical_distractor` | A small lexical change creates the commonsense contrast |
| `ambiguity` | Multiple readings or multiple options seem plausible |
| `dataset_noise` | Gold label seems wrong, weak, or ill-formed |
| `other` | Does not clearly fit another category |

### Stage 2: Summarize Manual Labels

After filling the manual taxonomy files, run:

```bash
python src/error_analysis/summarize_manual_labels.py \
  --input_dir outputs/error_analysis \
  --output_dir outputs/error_analysis/manual_summary
```

Generated files:

```text
outputs/error_analysis/manual_summary/
├── manual_error_labels_normalized.csv
├── annotation_coverage.csv
├── error_type_distribution.csv
├── representative_error_examples.csv
├── manual_error_analysis_summary.md
└── fig_error_type_distribution.png
```

`annotation_coverage.csv` reports how many rows have been labeled.

`error_type_distribution.csv` reports the count and percentage of each error type.

`representative_error_examples.csv` selects representative examples from the most frequent error types.

`manual_error_analysis_summary.md` provides a readable summary for reporting.

`fig_error_type_distribution.png` visualizes the error-type distribution across:

```text
Task A RoBERTa-large errors
Task B RoBERTa-large errors
Task A hard cases
Task B hard cases
```

To skip figure generation:

```bash
python src/error_analysis/summarize_manual_labels.py \
  --input_dir outputs/error_analysis \
  --output_dir outputs/error_analysis/manual_summary \
  --skip_plot
```

## 8. Recommended Experiment Workflow

A consistent experiment sequence is:

```text
1. Train BERT/RoBERTa models on original training data.
2. Evaluate on the original dev/test split.
3. Run LLM prompting baselines.
4. Run BERT-score and causal-LM PPL baselines.
5. Generate shuffled datasets and evaluate every method.
6. Generate matched T5 paraphrase subsets and evaluate every method.
7. Compare original, shuffled, and paraphrased performance.
8. Perform error analysis using prediction CSV files and robustness maps.
```

## 9. Reproducibility Notes

- Keep `seed` fixed across experiments.
- Use the same generated shuffled/paraphrased CSV files for all compared methods.
- Preserve `*_map.csv` and `*_meta.json` files for traceability.
- Compare paraphrase performance against the corresponding `*_original_subset_data.csv`, not against the full split.
- Record model name, seed, split, evaluation tag, and dataset path in experiment tables.
