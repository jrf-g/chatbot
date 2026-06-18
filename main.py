import json
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# 1. Load training data from JSON
# ---------------------------------------------------------
def load_training_pairs(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [(item["input"], item["output"]) for item in data]

training_pairs = load_training_pairs("training_data.json")


# ---------------------------------------------------------
# 2. Build vocabulary
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 3. Encode sentences
# ---------------------------------------------------------
def encode(sentence, vocab):
    return [vocab[word] for word in sentence.split()] + [vocab["<eos>"]]


# ---------------------------------------------------------
# 4. Attention Layer
# ---------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)

        hidden = hidden.repeat(1, seq_len, 1).transpose(0, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


# ---------------------------------------------------------
# 5. Encoder
# ---------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embed(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


# ---------------------------------------------------------
# 6. Decoder with Attention
# ---------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.embed(input_token).unsqueeze(1)

        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(rnn_input, hidden)

        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        output = self.fc(output)

        return output, hidden


# ---------------------------------------------------------
# 7. Seq2Seq wrapper
# ---------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_outputs, hidden = self.encoder(src)

        input_token = torch.tensor([vocab["<sos>"]])
        outputs = []

        for t in range(trg.size(1)):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs.append(output)
            input_token = trg[0][t].unsqueeze(0)

        return torch.stack(outputs)


# ---------------------------------------------------------
# 8. Train the model
# ---------------------------------------------------------
encoder = Encoder(len(vocab), 32, 64)
decoder = Decoder(len(vocab), 32, 64)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    total_loss = 0
    for inp, out in training_pairs:
        inp_ids = torch.tensor([encode(inp, vocab)])
        out_ids = torch.tensor([encode(out, vocab)])

        optimizer.zero_grad()
        logits = model(inp_ids, out_ids)

        loss = criterion(logits.view(-1, len(vocab)), out_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {total_loss:.4f}")


# ---------------------------------------------------------
# 9. Generate replies
# ---------------------------------------------------------
def generate_reply(model, text, max_len=10):
    model.eval()
    inp_ids = torch.tensor([encode(text, vocab)])

    encoder_outputs, hidden = model.encoder(inp_ids)

    input_token = torch.tensor([vocab["<sos>"]])
    output_words = []

    for _ in range(max_len):
        output, hidden = model.decoder(input_token, hidden, encoder_outputs)
        next_id = torch.argmax(output).item()

        if next_id == vocab["<eos>"]:
            break

        output_words.append(inv_vocab[next_id])
        input_token = torch.tensor([next_id])

    return " ".join(output_words)


# ---------------------------------------------------------
# 10. Chat loop
# ---------------------------------------------------------
print("Chatbot with Attention Ready!")

while True:
    user = input("You: ").lower()
    if user == "bye":
        print("Bot: goodbye")
        break
    print("Bot:", generate_reply(model, user))
