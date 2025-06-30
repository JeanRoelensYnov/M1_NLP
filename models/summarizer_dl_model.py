"""
File to generate the DL model which I'll use to summarize movie plot
"""

import polars as pl
from pathlib import Path
import logging
from typing import Union
from datasets import load_dataset
import re
import contractions
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    try:
        text = contractions.fix(text)  # type: ignore
    except Exception as e:
        logging.warning(f"Contraction expansion failed for text: {repr(text)}\nError: {e}")
        return ""

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_id)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        embedded_tgt = self.embedding(tgt)

        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)
        decoder_outputs, _= self.decoder(embedded_tgt, (hidden, cell))

        attn_input = torch.cat((decoder_outputs, encoder_outputs[:, -1:, :].expand_as(decoder_outputs)), dim=2)
        attn_hidden = torch.tanh(self.attn(attn_input))

        output_logits = self.out(attn_hidden)
        return self.softmax(output_logits)

def train(
    model: Seq2SeqModel, 
    dataloader: DataLoader, 
    optimizer, 
    criterion, 
    device: torch.device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        output = output.reshape(-1, output.size(-1))
        target = tgt[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)



if __name__ == "__main__":
    # Retrieve data 
    logger.info("Loading dataset")
    dataset = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries", split="train")
    df : pl.DataFrame = dataset.to_polars() # type: ignore
    logger.info("Dataset loaded")
    X = df.select(["Plot"])
    y = df.select(["PlotSummary"])

    logger.info("Preprocessing both X and y")
    X_texts = [preprocess_text(X[i, 0]) for i in range(X.height)]
    y_texts = [preprocess_text(y[i, 0]) for i in range(y.height)]

    raw_text_path = "tokenizer_input.txt"
    logger.info("Saving combined texts for tokenizer training")
    with open(raw_text_path, "w", encoding="utf-8") as f:
        for line in X_texts + y_texts:
            f.write(line + "\n")
    
    model_prefix = "movie_tokenizer"
    vocab_size = 8000


    if not Path(model_prefix + ".model").exists():
        logger.info("Training SentencePiece tokenizer")
        spm.SentencePieceTrainer.train( # type: ignore
            input=raw_text_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe"
        )
    else:
        logger.info("SentencePiece model already exists. Skipping training.")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model") # type: ignore

    logger.info("Tokenizing input and targets texts")

    X_tokenized = [sp.Encode(text, out_type=int) for text in X_texts]
    y_tokenized = [sp.Encode(text, out_type=int) for text in y_texts]

    MAX_LEN = 512
    X_tokenized = [x[:MAX_LEN] for x in X_tokenized]
    y_tokenized = [y[:MAX_LEN] for y in y_tokenized]

    PAD_ID = sp.pad_id() if sp.pad_id() >= 0 else 0

    X_tensor = pad_sequence([torch.tensor(x) for x in X_tokenized], batch_first=True, padding_value=PAD_ID)
    y_tensor = pad_sequence([torch.tensor(y) for y in y_tokenized], batch_first=True, padding_value=PAD_ID)

    torch.save((X_tensor, y_tensor), "movie_summary_data.pt")
    logger.info("Saved tokenized dataset")

    tensor_dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqModel(vocab_size=vocab_size, pad_id=PAD_ID).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss(ignore_index=PAD_ID)
    logger.info("Training start..")
    for epoch in range(10):
        loss = train(model, dataloader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch + 1}: loss = {loss:.4f}")
    logger.info("Training end.")
    model_path = "movie_summarizer_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved.")