import json
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Load training data from JSON
# -----------------------------
def load_training_pairs(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [(item["input"], item["output"]) for item in data]

training_pairs = load_training_pairs("training_data.json")


# -----------------------------
# 2. Build vocabulary
# -----------------------------
def build_vocab(pairs):
    vocab = {"<pad>":0, "<sos>":1, "<eos>":2}
    idx = 3
    for inp, out in pairs:
        for word in (inp + " " + out).split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

vocab = build_vocab(training_pairs)
inv_vocab = {v:k for k,v in vocab.items()}


# -----------------------------
# 3. Encode sentences
# -----------------------------
def encode(sentence, vocab):
    return [vocab[word] for word in sentence.split()] + [vocab["<eos>"]]


# -----------------------------
# 4. Define the GRU chatbot model
# -----------------------------
class Chatbot(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden


# -----------------------------
# 5. Train the model
# -----------------------------
model = Chatbot(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = 0
    for inp, out in training_pairs:
        inp_ids = torch.tensor([encode(inp, vocab)])
        out_ids = torch.tensor([encode(out, vocab)])

        optimizer.zero_grad()
        logits, _ = model(inp_ids)

        loss = criterion(logits.squeeze(0), out_ids.squeeze(0))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {total_loss:.4f}")


# -----------------------------
# 6. Generate replies
# -----------------------------
def generate_reply(model, text, max_len=10):
    model.eval()
    ids = torch.tensor([encode(text, vocab)])
    _, hidden = model(ids)

    input_id = torch.tensor([[vocab["<sos>"]]])
    output_words = []

    for _ in range(max_len):
        logits, hidden = model(input_id, hidden)
        next_id = torch.argmax(logits[0, -1]).item()

        if next_id == vocab["<eos>"]:
            break

        output_words.append(inv_vocab[next_id])
        input_id = torch.tensor([[next_id]])

    return " ".join(output_words)


# -----------------------------
# 7. Chat loop
# -----------------------------
print("PyTorch Chatbot Ready!")

while True:
    user = input("You: ").lower()
    if user == "bye":
        print("Bot: goodbye")
        break
    print("Bot:", generate_reply(model, user))
