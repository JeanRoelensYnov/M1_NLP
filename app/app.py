import streamlit as st
import torch
import pickle
import joblib
import numpy as np
from pathlib import Path
import sys
ROOT_FOLDER = Path(__file__).parent.parent
sys.path.append(str(ROOT_FOLDER))

# Local import
from models.generate_dl_model import GenreClassifier, EMBEDDING_DIM

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

# Streamlit UI

st.title("Movie Genre Classifier")

model_type = st.radio("Choose a model type:", ["ML", "DL"])
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
    preprocessed_input = preprocess(user_input)

    if model_type == "ML":
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

    st.success(f"Predicted genre: {pred}")
