"""
Fine-tune xlm-roberta-base for Uzbek sentiment analysis (pos/neg).
Intended environment: Google Colab w/ GPU, Python 3.10+, HuggingFace Transformers.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from sklearn.metrics import classification_report
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
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
OUTPUT_DIR = "./uzbek-sentiment-model"
SEED = 42

# ----------------------------
# Helper: map string labels to integers
# ----------------------------
LABEL2ID = {"neg": 0, "pos": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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

    def __init__(self, class_weights=None, tokenizer=None, **kwargs):
        # Pass tokenizer when supported by the installed Transformers version; otherwise set manually.
        try:
            super().__init__(tokenizer=tokenizer, **kwargs)
        except TypeError:
            super().__init__(**kwargs)
            self.tokenizer = tokenizer
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


def main():
    # Ensure reproducibility
    set_seed(SEED)

    # ----------------------------
    # Load data from CSVs
    # ----------------------------
    data_files = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }
    raw_datasets = load_dataset("csv", data_files=data_files)

    # ----------------------------
    # Map labels to integers
    # ----------------------------
    def encode_label(example):
        example["label"] = LABEL2ID[example["label"]]
        return example

    encoded = raw_datasets.map(encode_label)

    # ----------------------------
    # Initialize tokenizer and tokenize datasets
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        return tokenize_function(batch, tokenizer)

    tokenized = encoded.map(preprocess, batched=True)

    # ----------------------------
    # Compute class weights from the training split
    # ----------------------------
    train_labels = tokenized["train"]["label"]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(LABEL2ID.values())),
        y=np.array(train_labels),
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # ----------------------------
    # Prepare model and data collator
    # ----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----------------------------
    # Metrics: accuracy and weighted F1
    # ----------------------------
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"]
        return {"accuracy": acc, "f1": f1}

    # ----------------------------
    # Training arguments
    # ----------------------------
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_strategy="epoch",
        report_to="none",  # change to "wandb" or "tensorboard" if desired
    )

    # ----------------------------
    # Initialize trainer with weighted loss
    # ----------------------------
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ),
    )

    # ----------------------------
    # Train
    # ----------------------------
    trainer.train()

    # ----------------------------
    # Evaluate on test set with full classification report
    # ----------------------------
    test_results = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    print("\nClassification report (test):")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=[ID2LABEL[i] for i in sorted(ID2LABEL.keys())],
            digits=4,
        )
    )

    # ----------------------------
    # Final metrics (accuracy, weighted F1) from test set
    # ----------------------------
    final_acc = accuracy_metric.compute(predictions=test_preds, references=test_labels)[
        "accuracy"
    ]
    final_f1 = f1_metric.compute(
        predictions=test_preds, references=test_labels, average="weighted"
    )["f1"]
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test Weighted F1: {final_f1:.4f}")

    # ----------------------------
    # Save model and tokenizer
    # ----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
