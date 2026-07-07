import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken



# Hyperparameters
epochs = 5000
eval_epochs = 200
learning_rate = 3e-4
batch_size = 16
block_size = 128
n_embd=192
n_head=6
n_layer=6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout=0.2

print("Using device:", device)

# Read data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer (GPT-2 BPE via tiktoken)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# Before: Character level tokenisation...

# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}

# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

torch.manual_seed(1337)
B,T,C=4,8,2
x=torch.randn(B,T,C)
x.shape

head_size=16


class head(nn.Module):
    

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    

    def forward(self, x):
        B,T,C=x.shape
        k=self.key(x)   
        q=self.query(x) 
        head_size = k.shape[-1]

        wei=q @ k.transpose(-2,-1) * head_size**-0.5 
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1) 
        v=self.value(x) 
        out=wei @ v 
        return out
class MultiHeadAttention(nn.Module):
   

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    

    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self, x):
        # x = self.ln1(x)
        x = x + self.sa_head(self.ln1(x))  
        # x = self.ln2(x)
        x = x + self.ffwd(self.ln2(x))    
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = MultiHeadAttention(num_heads=6, head_size=n_embd//6)
        self.ffwd = FeedForward(n_embd)
        self.blocks=nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            nn.LayerNorm(n_embd)
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # (B,T,C)
        B,T=idx.shape
        tok_emb = self.token_embedding_table(idx)   # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  
        x = self.blocks(x)
        logits = self.lm_head(x)               
       
        if targets is None:
            loss = None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] 

            logits ,loss = self(idx_cond)
            logits=logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled
           
        return idx
    


# Create model
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train
for epoch in range(epochs):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss = {loss.item():.4f} | Val Loss = {model(*get_batch('val'))[1].item():.4f}")
        # print(f"Epoch {epoch:4d} | Validation Loss = {model(xb, yb)[1].item():.4f} ")

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print("\nGenerated text:\n")
print(decode(generated[0].tolist()))
