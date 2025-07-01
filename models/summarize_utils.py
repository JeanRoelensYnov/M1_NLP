import torch
import sentencepiece as spm
from models.summarizer_dl_model import Seq2SeqModel, preprocess_text
from torch.nn.utils.rnn import pad_sequence

MODEL_PATH = "models/movie_summarizer_model.pt"
TOKENIZER_PATH = "models/movie_tokenizer.model"
VOCAB_SIZE = 8000
MAX_LEN = 512

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)

PAD_ID = sp.pad_id() if sp.pad_id() >= 0 else 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2SeqModel(
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def summarize(text: str) -> str:
    text = preprocess_text(text)
    tokens = sp.Encode(text, out_type=int)[:MAX_LEN]
    input_tensor = torch.tensor(tokens, dtype=torch.long)
    input_tensor = pad_sequence([input_tensor], batch_first=True, padding_value=PAD_ID).to(device)

    with torch.no_grad():
        decoder_input = torch.tensor([[sp.bos_id()]], device=device) # bos -> begginning of sentence
        outputs = []
        hidden, cell = None, None

        for _ in range(MAX_LEN):
            embedded_src = model.embedding(input_tensor)
            encoder_outputs, (hidden, cell) = model.encoder(embedded_src)

            embedded_tgt = model.embedding(decoder_input)
            decoder_output, (hidden, cell) = model.decoder(embedded_tgt, (hidden, cell))

            attn_input = torch.cat((decoder_output, encoder_outputs[:, -1:, :].expand_as(decoder_output)), dim=2)
            attn_hidden = torch.tanh(model.attn(attn_input))

            output_logits = model.out(attn_hidden)
            output_probs = model.softmax(output_logits)
            next_token = output_probs.argmax(dim=-1)[:, -1]

            if next_token.item() == sp.eos_id():
                break

            outputs.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        return sp.Decode(outputs)
