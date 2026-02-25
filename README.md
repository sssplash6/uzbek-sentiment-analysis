# 🇺🇿 Uzbek Sentiment Analysis

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Models%20on%20Hub-yellow)](https://huggingface.co/sant1x)
[![Status](https://img.shields.io/badge/status-complete-success.svg)]()

**A benchmark study of 4 NLP models for binary sentiment classification of Uzbek-language text**

[Overview](#-overview) • [Results](#-results) • [Dataset](#-dataset) • [Models](#-models) • [Usage](#-usage) • [Reproduce](#-reproduce-training) • [Limitations](#-limitations)

</div>

---

## 🎯 Overview

Uzbek is a low-resource language with very limited NLP tooling. This project benchmarks 4 models — ranging from a classical TF-IDF baseline to fine-tuned multilingual transformers — on a self-collected dataset of Uzbek product reviews for binary sentiment classification (positive / negative).

All models, the dataset, and training code are open-source, contributing reusable resources for Uzbek NLP research.

**🤗 Models on HuggingFace:**
- [sant1x/uzbek-sentiment-xlm-roberta](https://huggingface.co/sant1x/uzbek-sentiment-xlm-roberta)
- [sant1x/uzbek-sentiment-mbert](https://huggingface.co/sant1x/uzbek-sentiment-mbert)
- [sant1x/uzbek-sentiment-distilbert](https://huggingface.co/sant1x/uzbek-sentiment-distilbert)

---

## 📊 Results

All models evaluated on the same held-out test set of 118 examples:

| Model | Accuracy | Weighted F1 | Neg F1 | Pos F1 |
|-------|----------|-------------|--------|--------|
| **TF-IDF + Logistic Regression** | **0.9576** | **0.9566** | **0.8936** | **0.9735** |
| mBERT | 0.9492 | 0.9492 | 0.8800 | 0.9677 |
| DistilBERT multilingual | 0.9407 | 0.9411 | 0.8627 | 0.9622 |
| XLM-RoBERTa | 0.9153 | 0.9175 | 0.8148 | 0.9451 |

### Discussion

The TF-IDF + Logistic Regression baseline outperformed all transformer models — a result consistent with findings in low-resource NLP literature. With only 1,172 training examples, transformers do not have enough data to fully leverage their pretraining, while TF-IDF benefits from the relatively predictable vocabulary of e-commerce reviews. This suggests that for production use on similarly small Uzbek datasets, classical methods remain competitive. Transformer models would likely surpass the baseline with a larger, more diverse dataset.

### XLM-RoBERTa Training Progress

| Epoch | Val Accuracy | Val F1 | Val Loss |
|-------|-------------|--------|----------|
| 1 | 91.45% | 91.79% | 0.4457 |
| 2 | 92.31% | 92.56% | 0.2276 |
| 3 | 94.02% | 94.14% | 0.2053 |

---

## 🗃️ Dataset

| Property | Value |
|----------|-------|
| **Source** | Uzum Market (uzum.uz) — Uzbekistan's largest e-commerce platform |
| **Total size** | 1,172 reviews |
| **Positive** | 930 (5-star reviews) |
| **Negative** | 242 (1-star reviews) |
| **Labeling method** | Distant supervision via star ratings |
| **Splits** | 80% train / 10% val / 10% test (stratified) |

Reviews were collected manually and labeled using distant supervision — 5-star ratings mapped to positive, 1-star to negative. A sample of 100 examples was manually verified for label quality. 2, 3, and 4-star reviews were excluded to keep class boundaries clean.

---

## 🛠️ Models

| Model | Size | HuggingFace |
|-------|------|-------------|
| TF-IDF + Logistic Regression | — | N/A (sklearn) |
| mBERT (`bert-base-multilingual-cased`) | 714MB | [sant1x/uzbek-sentiment-mbert](https://huggingface.co/sant1x/uzbek-sentiment-mbert) |
| DistilBERT multilingual | 542MB | [sant1x/uzbek-sentiment-distilbert](https://huggingface.co/sant1x/uzbek-sentiment-distilbert) |
| XLM-RoBERTa (`xlm-roberta-base`) | 1.12GB | [sant1x/uzbek-sentiment-xlm-roberta](https://huggingface.co/sant1x/uzbek-sentiment-xlm-roberta) |

**Tech stack:** Python 3.10+, HuggingFace Transformers + Trainer API, scikit-learn, Google Colab T4 GPU

---

## 💡 Usage

### Quick inference via HuggingFace pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sant1x/uzbek-sentiment-xlm-roberta"  # or mbert / distilbert variant
)

# Positive example
classifier("Mahsulot juda yaxshi, sifati a'lo!")
# [{'label': 'pos', 'score': 0.97}]

# Negative example
classifier("Sifat past, umuman yoqmadi.")
# [{'label': 'neg', 'score': 0.91}]
```

### Manual loading

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "sant1x/uzbek-sentiment-xlm-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("Mahsulot yaxshi!", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = logits.argmax().item()
print(model.config.id2label[predicted_class])  # 'pos'
```

---

## 🔁 Reproduce Training

### 1. Clone the repo

```bash
git clone https://github.com/sant1x/uzbek-sentiment-analysis.git
cd uzbek-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install transformers datasets evaluate scikit-learn torch
```

### 3. Prepare data

Place `train.csv`, `val.csv`, and `test.csv` in the root directory. Each file should have two columns: `text` and `label` (values: `pos` or `neg`).

### 4. Run training

To train only XLM-RoBERTa:
```bash
python train_xlm_roberta.py
```

To run the full 4-model benchmark:
```bash
python compare_models.py
```

> **Note**: Training was done on Google Colab with a T4 GPU. A GPU is strongly recommended — CPU training will be significantly slower.

---

## ⚠️ Limitations

- **Domain**: Trained on e-commerce reviews only. May not generalize to social media, news, or formal text.
- **Class imbalance**: Negative examples are underrepresented (~20% of data), affecting negative class performance despite class-weighted loss.
- **Dataset size**: 1,172 examples is small — a larger dataset would likely allow transformers to outperform the TF-IDF baseline.
- **Code-switching**: Uzbek speakers often mix Russian into writing; the models were not explicitly trained to handle this.
- **Dialects**: No coverage of regional Uzbek dialect variation.

---

## 🗺️ Future Work

- [ ] Expand dataset with social media and news comment sources
- [ ] Add neutral class for 3-class classification
- [ ] Evaluate on out-of-domain Uzbek text
- [ ] Experiment with Uzbek-specific pretrained models if/when available
- [ ] Build a simple demo on HuggingFace Spaces

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **HuggingFace** — for the Transformers library and model hosting
- **Uzum Market** — as the source of publicly available review data
- The broader **multilingual NLP research community** for making xlm-roberta-base, mBERT, and DistilBERT openly available
