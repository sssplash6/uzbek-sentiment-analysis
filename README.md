# 🇺🇿 Uzbek Sentiment Analysis

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20on%20Hub-yellow)](https://huggingface.co/sant1x/uzbek-sentiment-xlm-roberta)
[![XLM-RoBERTa](https://img.shields.io/badge/model-xlm--roberta--base-orange)](https://huggingface.co/xlm-roberta-base)
[![Status](https://img.shields.io/badge/status-complete-success.svg)]()

**Fine-tuned XLM-RoBERTa for binary sentiment classification of Uzbek-language text**

[Overview](#-overview) • [Results](#-results) • [Dataset](#-dataset) • [Usage](#-usage) • [Reproduce](#-reproduce-training) • [Limitations](#-limitations)

</div>

---

## 🎯 Overview

Uzbek is a low-resource language with very limited NLP tooling. This project fine-tunes `xlm-roberta-base` — a multilingual transformer trained on 100 languages — on a self-collected dataset of Uzbek product reviews to perform binary sentiment classification (positive / negative).

The model and dataset are both open-source, contributing a reusable resource for Uzbek NLP research.

**🤗 Model available on HuggingFace**: [sant1x/uzbek-sentiment-xlm-roberta](https://huggingface.co/sant1x/uzbek-sentiment-xlm-roberta)

---

## 📊 Results

Evaluated on a held-out test set of 118 examples:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.7586 | 0.8800 | 0.8148 | 25 |
| Positive | 0.9663 | 0.9247 | 0.9451 | 93 |
| **Weighted avg** | **0.9223** | **0.9153** | **0.9175** | **118** |

**Test Accuracy: 91.53% | Weighted F1: 91.75%**

### Training Progress

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

Reviews were scraped manually and labeled using distant supervision — 5-star ratings mapped to positive, 1-star to negative. A sample of 100 examples was manually verified for label quality.

3-star and 4-star reviews were excluded to keep class boundaries clean for binary classification.

---

## 🛠️ Tech Stack

| Category | Details |
|----------|---------|
| **Base Model** | xlm-roberta-base |
| **Framework** | HuggingFace Transformers + Trainer API |
| **Language** | Python 3.10+ |
| **Key Libraries** | `transformers`, `datasets`, `evaluate`, `scikit-learn`, `torch` |
| **Hardware** | Google Colab T4 GPU |
| **Training Time** | ~3.5 minutes (3 epochs) |

---

## 💡 Usage

### Quick inference via HuggingFace pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sant1x/uzbek-sentiment-xlm-roberta"
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

```bash
python train_xlm_roberta.py
```

The fine-tuned model will be saved to `./uzbek-sentiment-model`.

> **Note**: Training was done on Google Colab with a T4 GPU. A GPU is strongly recommended — CPU training will be significantly slower.

---

## ⚠️ Limitations

- **Domain**: Trained on e-commerce reviews only. May not generalize well to social media, news, or formal text.
- **Class imbalance**: Negative examples are underrepresented (~20% of data), which affects negative class recall despite class-weighted loss.
- **Code-switching**: Uzbek speakers often mix Russian into their writing. The model was not explicitly trained to handle this.
- **Dialects**: No coverage of regional Uzbek dialect variation.
- **Dataset size**: 1,172 examples is small by NLP standards. A larger dataset would improve robustness.

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
- The broader **multilingual NLP research community** for making xlm-roberta-base openly available
