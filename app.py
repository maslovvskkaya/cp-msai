import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
import plotly.express as px

MODEL_DIR = "model_with_metrics"
CSV_PATH = "labeled_data.csv"
MAX_LEN = 128
st.set_page_config(page_title="Detecting aggressive rhetoric", layout="wide")

st.title("Detecting aggressive rhetoric")
st.caption("Binary classification: Aggressive = 1, Not Aggressive = 0")

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

@st.cache_resource
def load_accuracy():
    metrics_file = os.path.join(MODEL_DIR, "metrics.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            for line in f:
                if line.startswith("accuracy="):
                    return float(line.split("=")[1])
    return None


@st.cache_resource
def load_dataset():
    df = pd.read_csv(CSV_PATH).dropna()

    if "tweet" in df.columns:
        text_col = "tweet"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError("Dataset must contain 'tweet' or 'text' column")

    if "class" in df.columns:
        label_col = "class"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError("Dataset must contain 'class' or 'label' column")

    df["binary_label"] = df[label_col].apply(lambda x: 1 if x in [0, 1] else 0)
    df[text_col] = df[text_col].astype(str)

    return df, text_col


model, tokenizer, device = load_model()
accuracy = load_accuracy()
df, text_col = load_dataset()


def predict_proba(texts):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        ).logits

    return torch.softmax(logits, dim=1).cpu().numpy()


def classify(text):
    probs = predict_proba([text])[0]
    return int(np.argmax(probs)), probs


tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Info about DistilBERT & LIME",
    "Dataset insights"
])

with tab1:
    st.subheader("Enter text to analyze:")
    text_input = st.text_area(
        "Your text:", height=140, placeholder="Type a tweet or message here..."
    )

    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text!")
            st.stop()

        pred, probs = classify(text_input)

        if accuracy is not None:
            st.info(f"Model accuracy: {accuracy:.4f}")

        col1, col2 = st.columns([1, 2])

        with col1:
            if pred == 1:
                st.error("Aggressive / Toxic (Class 1)")
            else:
                st.success("Non-Aggressive (Class 0)")

        with col2:
            st.markdown("### Confidence Scores")
            st.write(f"Aggressive (1): {probs[1]:.4f}")
            st.progress(float(probs[1]))
            st.write(f"Non-Aggressive (0): {probs[0]:.4f}")
            st.progress(float(probs[0]))

        st.divider()
        st.subheader("LIME Explanation")
        explainer = LimeTextExplainer(class_names=["Non-Aggression", "Aggression"])
        explanation = explainer.explain_instance(
            text_input,
            predict_proba,
            num_features=10,
            labels=[1]
        )
        st.components.v1.html(explanation.as_html(), height=650, scrolling=True)


with tab2:
    st.header("DistilBERT & LIME")
    st.markdown("""
DistilBERT is a compact version of the BERT model, developed by the Hugging Face team using knowledge distillation technique. In this technique, a smaller model ("student") is trained to imitate the behavior of a larger model ("teacher"), in this case BERT-base. DistilBERT has 40% fewer parameters (about 66 million instead of 110 million), making it lighter and faster.
The model retains about 97% of the original BERT's performance on many natural language processing (NLP) tasks, such as text classification, named entity recognition, or question answering. It achieves this by halving the number of layers and using a triple loss function during training (including distillation loss and cosine distance).
DistilBERT is an ideal choice for applications with limited resources, such as on mobile devices or in real-time, as inference is 60% faster. The model is available in the Hugging Face Transformers library and is widely used for fine-tuning on specific tasks.

LIME (Local Interpretable Model-agnostic Explanations) is an explainable AI (XAI) method proposed in 2016 that allows explaining predictions of any "black box" machine learning model. It is model-agnostic, meaning it works with any model regardless of its internal structure, and focuses on local explanations for individual predictions.
The core idea of LIME is to approximate a complex model with a simpler interpretable model (e.g., linear regression or decision tree) in the local neighborhood of a specific instance. To do this, perturbations (changes) are generated around the input data, the original model's predictions on these changes are evaluated, and then a surrogate model is trained with weights depending on proximity to the original instance.
LIME is particularly useful for tabular data, text, and images, where it highlights the most influential features for a specific prediction. This helps build trust in models, detect biases, and understand why the model made a certain decision. The method is implemented in Python (lime) and R libraries and is widely used in tasks requiring transparency, such as medicine or finance.
""")


with tab3:
    st.header("☁️ Dataset Vocabulary Insights")
    st.write(
        """
        This section visualizes the most frequent aggressive words
        from the training dataset (Class 1).
        Stop-words and neutral vocabulary are removed.
        """
    )

    if st.button("Generate Visualization", key="wordcloud_btn"):
        with st.spinner("Processing dataset..."):

            aggressive_df = df[df["binary_label"] == 1]

            if aggressive_df.empty:
                st.warning("No aggressive samples found in the dataset.")
                st.stop()

            toxic_text = " ".join(aggressive_df[text_col].astype(str))

            import re
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            from collections import Counter

            words = re.findall(r"\b[a-zA-Z]{4,}\b", toxic_text.lower())

            AGGRESSIVE_WORDS = {
                "hate", "kill", "stupid", "idiot", "fuck", "fucking",
                "shit", "bitch", "asshole", "racist", "retard",
                "moron", "trash", "dumb", "loser", "die",
                "cunt", "nigger"
            }

            filtered_words = [
                w for w in words
                if w not in ENGLISH_STOP_WORDS
                and (w in AGGRESSIVE_WORDS)
            ]

            if len(filtered_words) == 0:
                st.warning("No aggressive vocabulary found after filtering.")
                st.stop()

            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                colormap="Reds",
                max_words=100
            ).generate(" ".join(filtered_words))

            fig, ax = plt.subplots(figsize=(15, 7))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            st.divider()

            st.subheader("Top 10 most frequent aggressive words")

            word_counts = Counter(filtered_words).most_common(10)
            labels, values = zip(*word_counts)

            import plotly.express as px

            fig_bar = px.bar(
                x=labels,
                y=values,
                labels={"x": "Word", "y": "Frequency"},
                color=values,
                color_continuous_scale="Reds"
            )

            st.plotly_chart(fig_bar, use_container_width=True)