# Kenya Clinical Reasoning Challenge (Zindi) — MVP Seq2Seq Model

This repository contains a Jupyter Notebook (`Complete_MVPInstall.ipynb`) for the Kenya Clinical Reasoning Challenge hosted on Zindi. It walks through preprocessing clinical text, training a sequence-to-sequence (Seq2Seq) model with Hugging Face Transformers, evaluating with ROUGE metrics, and generating predictions for submission to Zindi.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Notebook Overview](#notebook-overview)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Data](#data)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Inference & Submission](#inference--submission)
10. [License](#license)

---

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Internet connection to install dependencies
- Zindi account and challenge access: [https://zindi.africa/competitions/kenya-clinical-reasoning-challenge](https://zindi.africa/competitions/kenya-clinical-reasoning-challenge)

---

## Installation

Install required Python packages:

```bash
pip install transformers datasets pandas torch rouge_score sentencepiece accelerate evaluate
```

> **Tip:** Use a virtual environment (e.g., `venv` or `conda`).

---

## Notebook Overview

The `Complete_MVPInstall.ipynb` notebook is organized into:

1. **Library Installation**: Ensures all required packages are available.
2. **Imports**: Loads Python modules and classes.
3. **Data Cleaning**: Defines `clean_tagged_dataframe` and `clean_text` to preprocess clinical source/target pairs.
4. **Load Dataset**: Reads challenge CSVs (`train.csv`, `test.csv`) and applies cleaning.
5. **Tokenization & Dataset Prep**:
   - Initializes `MvpTokenizer`.
   - Tokenizes clinical notes and prepares Hugging Face `DatasetDict`.
6. **Model Initialization**:
   - Loads `MvpForConditionalGeneration` pre-trained model.
   - Sets generation parameters.
7. **Training Arguments**:
   - Configures `Seq2SeqTrainingArguments` (batch size, learning rate, epochs, checkpointing).
8. **Trainer & Callbacks**:
   - Creates `Seq2SeqTrainer` with early stopping.
9. **Training Loop**:
   - Fine-tunes the model on the training split.
10. **Evaluation**:
    - Computes ROUGE-1, ROUGE-2, and ROUGE-L on validation data.
11. **Inference & Submission**:
    - Generates predictions on the test set.
    - Formats and saves a submission CSV for Zindi.

---

## Usage

1. **Clone this repository**:
   ```bash
   ```

git clone .git cd&#x20;

````

2. **Install dependencies** (see [Installation](#installation)).

3. **Download challenge data** from the Zindi competition page and place `train.csv` and `test.csv` in the root directory.

4. **Launch the notebook**:
   ```bash
jupyter notebook Complete_MVPInstall.ipynb
````

5. **Run cells in order** to preprocess data, train the model, evaluate, and generate submission.

---

## Configuration

Adjust hyperparameters in the **Training Arguments** cell:

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./zindi-model-v1",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_total_limit=2,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=128,
)
```

---

## Data

- **train.csv**: Contains `source` (clinical notes) and `target` (expected clinician reasoning).
- **test.csv**: Contains only `source` for which predictions are required.

Ensure both files are in your working directory or update file paths in the notebook accordingly.

---

## Model Training

We use Hugging Face’s `Seq2SeqTrainer`:

- **Tokenizer**: `MvpTokenizer` for clinical text.
- **Data Collator**: `DataCollatorForSeq2Seq` to batch tokenized inputs.
- **Callbacks**: `EarlyStoppingCallback` to stop training when metrics plateau.

Checkpoints and logs are saved under `./zindi-model-v1`.

---

## Evaluation

Evaluate on validation split and view ROUGE scores:

```python
metrics = trainer.evaluate()
print(metrics)
```

---

## Inference & Submission

1. **Switch to evaluation mode**:
   ```python
   ```

trainer.model.eval()

````
2. **Generate predictions** on `test.csv`.
3. **Format submission**:
   ```python
submission = test_df.copy()
submission["clinician_reasoning"] = predictions
submission.to_csv("submission.csv", index=False)
````

4. **Submit** the `submission.csv` on Zindi.

---

## License

MIT License — see `LICENSE`.

