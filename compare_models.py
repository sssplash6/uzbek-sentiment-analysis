"""
Train and evaluate three Uzbek sentiment classifiers (binary pos/neg) and
print a comparison table alongside a reference XLM-RoBERTa score row.

Models:
- TF-IDF + Logistic Regression (sklearn)
- mBERT (bert-base-multilingual-cased)
- DistilBERT multilingual (distilbert-base-multilingual-cased)

Environment: Google Colab T4 GPU, Python 3.10+, transformers/datasets/evaluate/sklearn installed.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ----------------------------
# Configuration
# ----------------------------
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LABEL2ID = {"neg": 0, "pos": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ----------------------------
# Utilities
# ----------------------------
def tokenize_function(example: Dict[str, str], tokenizer: AutoTokenizer):
    """Tokenize a single example with padding/truncation."""
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted cross-entropy."""

    def __init__(self, class_weights=None, processing_class=None, **kwargs):
        # Support both new and older Trainer signatures; prefer processing_class as requested.
        try:
            super().__init__(processing_class=processing_class, **kwargs)
        except TypeError:
            # Fallback: older versions lack processing_class; set tokenizer manually.
            super().__init__(**kwargs)
            self.tokenizer = processing_class
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def load_and_encode_datasets() -> Dict[str, Dict]:
    """Load train/val/test CSVs and map string labels to ints."""
    data_files = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
    raw = load_dataset("csv", data_files=data_files)

    def encode_label(example):
        example["label"] = LABEL2ID[example["label"]]
        return example

    return raw.map(encode_label)


def compute_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(LABEL2ID.values())),
        y=train_labels,
    )
    return torch.tensor(weights, dtype=torch.float)


def compute_metrics_builder():
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        return {"accuracy": acc, "f1": f1}

    return compute_metrics


def eval_classification_report(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Return per-class F1 plus accuracy/weighted-F1 from sklearn report."""
    report = classification_report(
        labels,
        preds,
        target_names=[ID2LABEL[i] for i in sorted(ID2LABEL.keys())],
        output_dict=True,
        digits=4,
    )
    acc = report["accuracy"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    neg_f1 = report["neg"]["f1-score"]
    pos_f1 = report["pos"]["f1-score"]
    return {"accuracy": acc, "weighted_f1": weighted_f1, "neg_f1": neg_f1, "pos_f1": pos_f1}


# ----------------------------
# Model: TF-IDF + Logistic Regression
# ----------------------------
def run_tfidf_lr(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    # Combine train+val for final fit (common when validation only needed for deep models).
    combined_texts = pd.concat([train_df, val_df], axis=0)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    pipe = make_pipeline(vectorizer, clf)
    pipe.fit(combined_texts["text"], combined_texts["label"])

    test_preds = pipe.predict(test_df["text"])
    return eval_classification_report(test_df["label"].to_numpy(), test_preds)


# ----------------------------
# Model: Transformer fine-tuning helper
# ----------------------------
def run_transformer(model_name: str, output_dir: str, tokenized_datasets, class_weights: torch.Tensor) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenize_function(batch, tokenizer)

    tokenized = tokenized_datasets.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
        class_weights=class_weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        processing_class=tokenizer,
    )

    trainer.train()

    # Evaluate on test set
    test_output = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids

    # Persist final model/tokenizer
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return eval_classification_report(test_labels, test_preds)


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)

    # Load CSVs as dataframes for sklearn path
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    # Map labels to ints
    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].map(LABEL2ID)

    # Load datasets for HF
    tokenized_datasets = load_and_encode_datasets()
    class_weights = compute_class_weights(np.array(tokenized_datasets["train"]["label"]))

    results = {}

    # TF-IDF + LR
    print("Training TF-IDF + Logistic Regression...")
    results["TF-IDF + Logistic Regression"] = run_tfidf_lr(train_df, val_df, test_df)

    # mBERT
    print("Training mBERT (bert-base-multilingual-cased)...")
    results["mBERT"] = run_transformer(
        model_name="bert-base-multilingual-cased",
        output_dir="./mbert-model",
        tokenized_datasets=tokenized_datasets,
        class_weights=class_weights,
    )

    # DistilBERT multilingual
    print("Training DistilBERT multilingual (distilbert-base-multilingual-cased)...")
    results["DistilBERT multilingual"] = run_transformer(
        model_name="distilbert-base-multilingual-cased",
        output_dir="./distilbert-model",
        tokenized_datasets=tokenized_datasets,
        class_weights=class_weights,
    )

    # Add reference row
    results["XLM-RoBERTa (reference)"] = {
        "accuracy": 0.9153,
        "weighted_f1": 0.9175,
        "neg_f1": 0.8148,
        "pos_f1": 0.9451,
    }

    # ----------------------------
    # Print comparison table
    # ----------------------------
    header = (
        "Model",
        "Accuracy",
        "Weighted F1",
        "Neg F1",
        "Pos F1",
    )
    print("\\nModel                        | Accuracy | Weighted F1 | Neg F1 | Pos F1")
    print("-----------------------------|----------|-------------|--------|-------")
    for model_name in [
        "TF-IDF + Logistic Regression",
        "mBERT",
        "DistilBERT multilingual",
        "XLM-RoBERTa (reference)",
    ]:
        r = results[model_name]
        print(
            f"{model_name:<29}|  {r['accuracy']:.4f}  |    {r['weighted_f1']:.4f}   | {r['neg_f1']:.4f} | {r['pos_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
