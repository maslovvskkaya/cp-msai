# Detecting Aggressive Rhetoric with DistilBERT

This project implements a **binary text classification system** for detecting **aggressive / toxic rhetoric** in short texts.  
It combines a **fine-tuned DistilBERT model**, **Explainable AI (LIME)**, and an **interactive Streamlit web application**.

The project demonstrates a **complete NLP pipeline**: data preprocessing, model training, evaluation, model persistence, and deployment.

---

## Key Features

- **Binary classification**
  - Aggressive  — `1`
  - Non-Aggressive — `0`

- **Transformer-based NLP model**
  - Fine-tuned `DistilBERT (distilbert-base-uncased)`

- **Evaluation metrics**
  - Accuracy
  - F1-score

- **Explainable AI (XAI)**
  - Local explanations of predictions using **LIME**

- **Interactive Web Application**
  - Built with **Streamlit**
  - Real-time text analysis and visualization

- **Dataset insights**
  - WordCloud of aggressive vocabulary
  - Top-frequency aggressive terms

---

## Tech Stack

- **Language:** Python  
- **Deep Learning:** PyTorch  
- **NLP / Transformers:** Hugging Face Transformers  
- **Model:** DistilBERT  
- **Web Framework:** Streamlit  
- **Explainability:** LIME  
- **Data Processing:** pandas, NumPy  
- **Visualization:** matplotlib, Plotly, WordCloud  
- **Evaluation:** scikit-learn  

---

## Project Structure

cp-msai/
- app.py - Streamlit web application
- train_model.py - Model training and evaluation
- labeled_data.csv - Labeled text dataset
- model_with_metrics/ - Saved trained model and tokenizer
-- config.json - Model configuration
-- pytorch_model.bin - Weights of the trained DistilBERT
-- tokenizer.json , vocab.txt - Tokenizer
-- tokenizer_config.json, special_tokens_map.json - Service files
-- requirements.txt - Python dependencies
- README.md - Project documentation

---

## Dataset

Link: https://www.kaggle.com/datasets/yashdogra/toxic-tweets
- **Input:** texts 
- **Text column:** `tweet` 
- **Label column:** `class`
- **Target variable:**
  - `1` — Aggressive 
  - `0` — Non-Aggressive

### Preprocessing steps:


- removal of missing values;
- conversion to binary labels;
- stratified train/validation split;
- tokenization using DistilBERT tokenizer.

---

## Model Description

The project uses **DistilBERT**, a compact and efficient version of BERT obtained through **knowledge distillation**.  
It preserves most of BERT’s performance while being significantly faster and lighter, making it suitable for real-time applications.

### Training configuration:
- **Batch size:** 16  
- **Epochs:** 5  
- **Maximum sequence length:** 128  
- **Optimizer:** AdamW  
- **Learning rate:** 2e-5  
- **Loss function:** Cross-Entropy  

### Evaluation metrics:
- Accuracy
- F1-score

---

## Model Training

To train the model from scratch:

```bash
git clone https://github.com/maslovvskkaya/cp-msai.git
cd cp-msai
python train_model.py
