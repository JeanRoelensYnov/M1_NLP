from pathlib import Path
import sys
ROOT_FOLDER = Path(__file__).parent.parent
sys.path.append(str(ROOT_FOLDER))
import streamlit as st
import torch
import pickle
import joblib
import numpy as np


# Local import
from models.generate_dl_model import GenreClassifier, EMBEDDING_DIM
from models.summarize_utils import summarize
from data.datasets_preprocess import convert_all_numbers

# For typing
from sklearn.preprocessing import LabelEncoder
from typing import Dict
from sklearn.pipeline import Pipeline

# For preprocessing
import spacy
from data.datasets_preprocess import convert_all_numbers
import re
from string import punctuation


NLP = spacy.load("en_core_web_trf")

# For "summarization"
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import heapq
from typing import Dict

nltk.download('punkt')
nltk.download('stopwords')


# Load necessary assets

with open(ROOT_FOLDER / "models" / "ml_model.pkl", "rb") as f:
    ml_model: Pipeline = joblib.load(f)

with open(ROOT_FOLDER / "models" / "word2idx.pkl", "rb") as f:
    word2idx: Dict[str, int] = pickle.load(f)


with open(ROOT_FOLDER / "models" / "label_encoder.pkl", "rb") as f:
    label_encoder: LabelEncoder = pickle.load(f)

dl_model = GenreClassifier(
    vocab_size=len(word2idx),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=128,
    output_dim=len(label_encoder.classes_),
    pad_idx=0,
)
dl_model.load_state_dict(
    torch.load(
        ROOT_FOLDER / "models" / "dl_model.pkl", map_location=torch.device("cpu")
    )
)

# Summarizer

def ml_summarize(text: str, num_sentences = 6):
    sentences = sent_tokenize(text)

    stop_words = set(stopwords.words('english'))
    word_frequencies = {}

    for word in word_tokenize(text.lower()):
        if word not in stop_words and word not in punctuation:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    sentences_score : Dict[str,int] = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if len(sentence.split(" ")) < 30: # Avoid long sentences
                    if sentence not in sentences_score:
                        sentences_score[sentence] = word_frequencies[word]
                    else:
                        sentences_score[sentence] += word_frequencies[word]
    sentence_ranks = heapq.nlargest(num_sentences, enumerate(sentences_score), key=lambda x: x[1])
    sorted_sentences = sorted(sentence_ranks, key=lambda x: x[0])  # sort by original order
    summary = ' '.join([sentences[i] for i, _ in sorted_sentences])
    return summary

# Streamlit UI

st.title("Movie Genre Classifier")

model_type = st.radio("Choose a model type:", ["ML Classification", "DL Classification", "ML Summarize", "DL Summarize"])
user_input = st.text_area("Enter a movie synopsis:", height=200)
submit = st.button("Predict")

# Logic


def preprocess(user_input: str) -> str:
    user_input = convert_all_numbers(user_input)
    user_input = re.sub(rf"[{re.escape(punctuation)}]", "", user_input)
    user_input = re.sub(r"\s+", " ", user_input)
    doc = NLP(user_input)
    return " ".join(token.lemma_ for token in doc if not token.is_stop)


if submit and user_input.strip():
    if model_type in ["ML Classification", "DL Classification"]:
        preprocessed_input = preprocess(user_input)

        if model_type == "ML Classification":
            pred = ml_model.predict([preprocessed_input])[0]
        else:  # DL
            tokens = preprocessed_input.split()
            indices = [word2idx.get(word, 0) for word in tokens]
            tensor_input = torch.LongTensor(indices)
            padded_input = torch.nn.utils.rnn.pad_sequence(
                [tensor_input], batch_first=True, padding_value=0
            )

            dl_model.eval()
            with torch.no_grad():
                output = dl_model(padded_input)
                predicted_index = torch.argmax(output, dim=1).item()
                pred = label_encoder.inverse_transform([predicted_index])[0]

        st.write(f"Predicted genre: {pred}")
    else: # Summary
        if model_type == "DL Summarize":
            with st.spinner("Generating summary ..."):
                summary = summarize(user_input)
            st.write(f"DL Summary found : \n{summary}")
        else:
            pred = ml_summarize(user_input)
            st.write(f"ML Summary found : \n{pred}")
