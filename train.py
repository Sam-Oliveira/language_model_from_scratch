import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F


# Create dataset and dataloader
class ShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokens) - self.context_length
        
    def __getitem__(self, idx):
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.context_length + 1]
        
        # Split into input and target
        # basically each idx into dataset outputs a pair of x and y, each of length context_length.
        # then we predict y[0] from x[0], y[1] from x[0:2], etc. until we predict y[context_length-1] from x[0:context_length]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, context_length,n_blocks):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_length, n_embed)
        
        # for self attention
        #self.attention_head= SelfAttention(n_embed,head_size=n_embed)

        # for multi head attention
        self.attention_head= MultiHeadAttention(num_heads=4,n_embed=n_embed,head_size=n_embed//4)

        self.blocks=nn.Sequential(*[Block(n_embed,num_heads=4) for _ in range(n_blocks)],
            nn.LayerNorm(n_embed))

        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.context_length = context_length
    
    def forward(self, idx, targets=None):
        #idx and targets are both of shape (batch_size, context_length)
        token_embedding = self.token_embedding_table(idx) # (batch_size, context_length, n_embed)
        position_embedding = self.position_embedding_table(torch.arange(idx.shape[1])) # (context_length, n_embed)
        x = token_embedding + position_embedding
        x = self.attention_head(x)  # (batch_size, context_length, n_embed)

        x=self.blocks(x)
        logits = self.lm_head(x) # (batch_size, context_length, vocab_size)
        return logits

    def generate(self, idx, max_new_tokens):
        # idx is of shape (batch_size, context_length)
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            # Get predictions
            logits = self(idx_cond)  # (batch_size, context_length, vocab_size)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# self attention makes it so that each token will emit two vectors: one for the query, one for the key.
# the query and key are used to compute the attention weights, which are then used to compute the output.
# the output is a weighted sum of the values, where the weights are the attention weights.
# the values are the embeddings of the tokens.
# the query value represents what the token is looking for in the key value pairs. The key value represents what the token contains.
# It' called self attention because the query, key, and value all come from the same source (the same tokens in x).
# in cross attention, the query comes from one source, and the key and value come from another source. (e.g. in encoder-decoder attention)
class SelfAttention(nn.Module):
    def __init__(self, n_embed,head_size):
        super().__init__()
        self.n_embed = n_embed
        self.query = nn.Linear(n_embed, head_size,bias=False)
        self.key = nn.Linear(n_embed, head_size,bias=False)
        self.value = nn.Linear(n_embed, head_size,bias=False)

    def forward(self, x):
        # T is the context length, C is the embed size
        B,T,C = x.shape
        assert C==self.n_embed, "n_embed must be equal to the number of features in the input"
        q = self.query(x) # (B,T,head_size)
        k = self.key(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size) - head size is the size of representation for each token

        # compute the attention weights
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**0.5 # (B,T,T) (scaling by sqrt of head size)
        weights = weights.softmax(dim=-1) # (B,T,T)
        mask=torch.tril(torch.ones(T,T))

        # for decoder attention, we don't want to attend to future tokens. But for encoder attention, we want to attend to all tokens.
        weights = weights.masked_fill(mask==0, float('-inf'))
        weights=F.softmax(weights,dim=-1)
        out = weights @ v # (B,T,head_size)
        
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,n_embed,head_size):
        super().__init__()
        self.heads=nn.ModuleList([SelfAttention(n_embed,head_size) for _ in range(num_heads)])

    def forward(self,x):
        out=torch.cat([head(x) for head in self.heads],dim=-1)
        return out
    

class Block(nn.Module):
    def __init__(self,n_embed,num_heads):
        super().__init__()
        self.ln1=nn.LayerNorm(n_embed)
        self.attention=MultiHeadAttention(num_heads,n_embed=n_embed,head_size=n_embed//num_heads)
        self.ln2=nn.LayerNorm(n_embed)
        self.feed_forward=FeedForward(n_embed)

    def forward(self,x):
        # layer norm before feed forward and self attention
        x=x+self.attention(self.ln1(x))
        x=x+self.feed_forward(self.ln2(x))
        return x
    
# independently for each token! batch and context length dimensions are treated as batch dimensions
class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed))   
    
    def forward(self,x):
        return self.net(x)

@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader):
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}  # Initialize losses dictionary with zeros
    eval_iters = 2
    for split in ['train', 'val']:
        dataloader = train_dataloader if split == 'train' else val_dataloader
        for k in range(eval_iters):
            X, Y = next(iter(dataloader))  # Get a batch from the dataloader
            logits = model(X, Y)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), Y.view(-1))
            losses[split] += loss.item()
        losses[split] /= (eval_iters)
    model.train()
    return losses


def train():
    n_embed = 32
    context_length = 8
    batch_size = 32
    learning_rate = 1e-3
    max_iters = 1000

    with open('tiny_shakespeare.txt', 'r') as f:
        text = f.read()

    tokenizer = tiktoken.encoding_for_model("gpt-2")
    vocab_size = tokenizer.n_vocab  # Use GPT-2's vocabulary size
    dataset = ShakespeareDataset(text, tokenizer, context_length)

    # Split dataset into train and validation sets (90-10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    model = LanguageModel(vocab_size, n_embed, context_length,n_blocks=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for i in range(max_iters):

        for input,target in train_dataloader:
            logits = model(input,target)
            loss = F.cross_entropy(logits.view(-1,vocab_size), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses=estimate_loss(model, train_dataloader, val_dataloader)
        print('Train loss: ', losses['train'],'Val loss: ', losses['val'])

        # example generation
        output = model.generate(torch.zeros((1,1),dtype=torch.long),max_new_tokens=100).tolist()[0]
        print(tokenizer.decode(output))

train()
