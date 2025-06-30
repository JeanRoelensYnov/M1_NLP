"""
File to generate the DL model which I'll use to predict the genre of a synopsis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from generate_ml_model import load_data
from sklearn.preprocessing import LabelEncoder
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "dl_model.pkl"
EMBEDDING_DIM = 50


def create_vocabulary(tokens: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {}
    idx2word = {}
    for sentence in tokens:
        for word in sentence:
            if word not in word2idx:
                idx2word[len(word2idx)] = word
                word2idx[word] = len(word2idx)
    return word2idx, idx2word


class GenreDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class GenreClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim: int,
        pad_idx: torch.Tensor,
        output_dim=1,
        embedding_dim=EMBEDDING_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        out = self.dropout(pooled)
        return self.fc(out)


def train_epoch(
    model: GenreClassifier,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    total_loss, total_correct = 0, 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        total_correct += (preds.argmax(1) == y_batch).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def eval_epoch(
    model: GenreClassifier,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            total_correct += (preds.argmax(1) == y_batch).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


if __name__ == "__main__":

    """
    Model preparation
    """
    logger.info("Model preparation start..")
    X_train, X_test, y_train, y_test = load_data()

    # We can tokenize with split since X_train is already preprocessed data
    tokens = [sentence.split() for sentence in X_train]

    word2idx, idx2word = create_vocabulary(tokens)

    indices = [[word2idx[word] for word in sentence] for sentence in tokens]

    padded_indices = nn.utils.rnn.pad_sequence(
        [torch.LongTensor(sentence) for sentence in indices], batch_first=True
    )

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)

    train_labels = torch.tensor(y_train_encoded, dtype=torch.long)

    train_dataset = GenreDataset(padded_indices, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    logger.info("Model preparation done.")
    """
    Model training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_encoder.classes_)
    model = GenreClassifier(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=128,
        output_dim=num_classes,
        pad_idx=0,
    ).to(device)
    logger.info("Model instantiation done.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info("Training phase start..")
    for epoch in range(2):
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        logger.info(
            f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}"
        )
    logger.info("Training phase end..")
    logger.info("Test phase..")
    test_tokens = [s.split() for s in X_test]
    test_indices = [[word2idx.get(word, 0) for word in s] for s in test_tokens]
    test_padded = nn.utils.rnn.pad_sequence(
        [torch.LongTensor(sentence) for sentence in test_indices], batch_first=True
    )

    test_labels = torch.tensor(label_encoder.transform(y_test), dtype=torch.long)
    test_dataset = GenreDataset(test_padded, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"Test loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(PROJECT_ROOT / "models" / "word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)
    with open(PROJECT_ROOT / "models" / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info("Deep Learning model, word2idx, label_encoder saved.")
