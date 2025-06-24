"""
File to generate the DL model which I'll use to predict the genre of a synopsis
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import Dataset, DataLoader

PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 100


def load_glove_embeddings(glove_path: str, word2idx : dict, embedding_dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    embeddings[word2idx['PAD']] = np.zeros(embedding_dim)

    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                vect = np.array(parts[1:], dtype=np.float32)
                embeddings[word2idx[word]] = vect
    
    return torch.tensor(embeddings, dtype=torch.float)

def text_to_sequence(text : str, word2idx: dict, max_len=MAX_LEN):
    tokens = text.split() # We only use split since we preprocess data ourselves
    seq = [word2idx.get(tok, UNK_IDX) for tok in tokens]
    if len(seq) < MAX_LEN:
        seq += [PAD_IDX] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx], self.word2idx, self.max_len)
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class TextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, freeze_embeddings=True):
        super().__init__()
        vocab_size, embeddings_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embeddings_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        dropped = self.dropout(pooled)
        out = self.fc(dropped)
        return out
    
def train_epochs(model : TextClassifier, dataloader, criterion, optimizer : optim.Optimizer, device : torch.device):
    model.train()
    epoch_loss = 0
    preds = []
    trues = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0) 
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    return epoch_loss / len(dataloader.dataset), acc

def eval_epoch(model : TextClassifier, dataloader, criterion, device : torch.device):
    model.eval()
    epoch_loss = 0
    preds = []
    trues = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item() * inputs.size(0)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    return epoch_loss / len(dataloader.dataset), acc


if __name__ == ("__main__"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load your embeddings
    embedding_matrix = load_glove_embeddings("path/to/glove.6B.100d.txt", word2idx, 100)

    # Prepare datasets and dataloaders
    train_dataset = TextDataset(train_texts, train_labels, word2idx)
    val_dataset = TextDataset(val_texts, val_labels, word2idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Build model
    model = TextClassifier(embedding_matrix, hidden_dim=128, output_dim=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.3f} Acc {train_acc:.3f} | Val Loss {val_loss:.3f} Acc {val_acc:.3f}")

    # Save the model
    torch.save(model.state_dict(), "models/dl_model.pth")
