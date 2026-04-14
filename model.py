# -*- coding: utf-8 -*-
"""
Original reference implementation (from Colab notebook).

This is the karpathy-style decoder-only transformer that served
as the basis for train_model.py. Kept here for reference.
"""

import torch, math
import torch.nn as nn
from torch.nn import functional as F

torch.set_printoptions(sci_mode=False)

batch_size = 16
block_size = 64
max_iters = 5000
SZ = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
sz_token = 64 * 6
CNT_HEADS = 6
n_layers = 3
dropout = 0.2

text1 = open('input.txt', 'r').read()
all = sorted(list(set({x for x in text1})))
# print(all)
stoi = { ch:i for i, ch in enumerate(all)}
itos = { i:ch for i, ch in enumerate(all)}
# print(itos)
encode = lambda l : [stoi[x] for x in l]
decode = lambda l : "".join(itos[x] for x in l)
# print(decode(encode(text1)[:1000]))

data = torch.tensor(encode(text1), dtype=torch.int64)
# print(torch.version)
n = int(0.9 * len(text1))
train_data = data[:n]
validate_data = data[n:]

torch.manual_seed(5)


def get_random(flag):
  data = train_data
  if flag == "validate":
    data = validate_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


@torch.no_grad()
def estimate_loss():
  out = {}
  m1.eval()
  for cur in ['train', 'validate']:
    now = torch.zeros(eval_iters)
    for i in range(eval_iters):
      a, b = get_random(cur)
      logits, loss = m1(a, b)
      now[i] = loss
    out[cur] = now.mean()
  m1.train()
  return out


class Head(nn.Module):
    def __init__(self, sz_head):
      super().__init__()
      self.query = nn.Linear(sz_token, sz_head, bias=False)
      self.key = nn.Linear(sz_token, sz_head, bias=False)
      self.weight = nn.Linear(sz_token, sz_head, bias=False)
      self.dropout = nn.Dropout(dropout)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, logits):
      B, T, C = logits.shape # C == sz_head
      q = self.query(logits) # (B, T, C)
      k = self.key(logits) # (B, T, C)
      w = self.weight(logits) # (B, T, C)
      # k = k.permute(0, 2, 1)

      matrix = (q @ k.transpose(-2, -1)) * (w.shape[2] ** -0.5) # (B, T, T)
      # matrix = torch.tril(matrix)
      # tril = torch.tril(torch.ones((T, T)))
      matrix = matrix.masked_fill(self.tril[:T, :T] == 0, -math.inf)
      # print(matrix)
      matrix = F.softmax(matrix, dim=-1)
      matrix = self.dropout(matrix)
      res = matrix @ w # (B, T, C)
      return res



class MultiHead(nn.Module):
  def __init__(self, cnt_head, sz_head):
    super().__init__()
    self.heads = nn.ModuleList([Head(sz_head) for x in range(cnt_head)])
    self.sz_head = sz_head
    self.W0 = nn.Linear(sz_token, sz_token)
    self.dropout = nn.Dropout(dropout)

  def forward(self, logits):
    # print(logits[:, :, :self.sz_head].shape)
    ret = self.W0(torch.cat([h(logits) for h in self.heads], dim=-1))
    ret = self.dropout(ret)
    # print(ret.shape)
    
    return ret


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(sz_token, 4 * sz_token),
        nn.ReLU(),
        nn.Linear(4 * sz_token, sz_token),
        nn.Dropout(dropout),
    )
  
  def forward(self, x):
    return self.net(x)


class block(nn.Module):
  def __init__(self, cnt_heads):
    super().__init__()
    self.attention = MultiHead(cnt_heads, sz_token // cnt_heads)
    self.mlp = MLP()
    self.ln1 = nn.LayerNorm(sz_token)
    self.ln2 = nn.LayerNorm(sz_token)

  def forward(self, x):
    x = x + self.attention(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class Bigram(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, sz_token)
    self.position_embedding = nn.Embedding(block_size, sz_token)
    self.unembedding = nn.Linear(sz_token, vocab_size, bias=False)
    self.blocks = nn.Sequential(*[block(CNT_HEADS) for _ in range(n_layers)])
    self.ln = nn.LayerNorm(sz_token)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    # idx = torch.cat((idx, torch.randint(high=vocab_size, size=(B, 1))), dim=-1)
    # print(idx.shape)
    logits = self.embedding(idx) # (B, T, sz_token)
    logits_pos = self.position_embedding(torch.arange(end=T)) # (B, T, sz_token)
    logits = logits + logits_pos
    # targets = self.em

    # q = self.query(logits) # (B, T, sz_token)
    # k = self.key(logits) # (B, T, sz_token)
    # w = self.weight(logits) # (B, T, sz_token)
    # # k = k.permute(0, 2, 1)

    # matrix = (q @ k.transpose(-2, -1)) * (sz_token ** -1/2) # (B, T, T)
    # # matrix = torch.tril(matrix)
    # tril = torch.tril(torch.ones((T, T)))
    # matrix = matrix.masked_fill(tril == 0, -math.inf)
    # # print(matrix)
    # matrix = F.softmax(matrix, dim=-1)
    # res = matrix @ w # (B, T, sz_token)
    # print(res.shape)
    # res = self.W0(self.heads.forward(logits, self.Q, self.K, self.V))
    # res = self.mlp.forward(res)
    res = self.blocks(logits)
    res = self.ln(res)
    # print(res.shape)
    logits = self.unembedding(res) # (B, T, C)
    # print(logits.shape)
    # logits = logits + res
    # logits = res[:, -1:, :].squeeze()
    if targets == None:
      loss = None
      return logits, loss
    else:
      # logits = logits.permute(0, 2, 1) # (B, C, T)
      logits = logits.view(B * T, vocab_size)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
      return logits, loss
    # print(logits)


    # print(q.shape)

  # def forward(self, idx, targets=None):
  #   logits = self.embedding(idx) # B, T, sz_token
  #   # print(logits.shape)
  #   logits = self.matrix(logits) # B, T, C
  #   # print(logits.shape)
  #   if targets == None:
  #     loss = None
  #     return logits, loss
  #   else:
  #     logits = logits.permute(0, 2, 1) # B, C, T
  #     loss = F.cross_entropy(logits, targets)
  #     return logits, loss


  def generate(self, idx, cnt_tokens):
    for _ in range(cnt_tokens):
      logits, loss = self(idx[:, -block_size:])
      logits = F.softmax(logits, -1)
      # print(logits)
      # print(logits.shape)
      logits = logits[:, -1:, :] # B, C
      logits = logits.squeeze()
      # print(logits)
      # print(logits.shape)
      nxt = torch.multinomial(logits, num_samples=1)
      # print(nxt)
      # idx = torch.stack([idx, [el for el in nxt]])
      idx = torch.cat((idx, nxt), dim=1)
    return idx



vocab_size = len(all)
m1 = Bigram().to(device)
optimizer = torch.optim.AdamW(m1.parameters(), lr=1e-3)
a, b = get_random('')

for i in range(max_iters):
  if i % SZ == 0:
    res = estimate_loss()
    print(f"{i / SZ}, train: {res['train']}, val: {res['validate']}")
  a, b = get_random('')
  logits, loss = m1.forward(a, b)
  m1.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


# print(m1.forward(a, b))
# print('------------------')
# print(m1.generate(a, 100).tolist())
print(decode(m1.generate(a, 300).tolist()[0]))


m1.eval()
with open('log.txt', 'w') as f:
  f.write(decode(m1.generate(a, 5000).tolist()[0]))