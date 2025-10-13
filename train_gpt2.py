from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import tiktoken 
import matplotlib.pyplot as plt
device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")


# -----------------------------------------------------------------------------

@dataclass
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout:float = 0.0


class MLP(nn.Module):
    def __init__(self, config:GPTconfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x:torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self,config:GPTconfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size)) # C,C -> 1,1,C,C (Batched)

    def forward(self,x:torch.Tensor):
        B,T,C = x.shape

        # intialising our k,q,v
        q,k,v = self.c_attn(x).split(self.config.n_embd,dim=-1)
        
        # We're doing some tensor gymnastics to view our last channel as n_head * head_size, and make the n_head as a Batch Dimension, so that all heads are processed in parallel 
        # split C into head_size,n_head
        q = q.view(B,T,self.config.n_head,C//self.config.n_head).transpose(1,2) # B,n_head,T,head_size
        k = k.view(B,T,self.config.n_head,C//self.config.n_head).transpose(1,2) # B,n_head,T,head_size
        v = v.view(B,T,self.config.n_head,C//self.config.n_head).transpose(1,2) # B,n_head,T,head_size
        
    # heart of cauusal-self-attention mechanism 🧐

            # (B,T,n_head,head_size) @ (B,T,head_size,n_head) -> (B,T,n_head,n_head)
        att = (q@k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))  # scaling attention product by 1 / sqroot(Dk) to prevent the variance from getting too large
        # after matmul we get (B,n_heads,num_key_positions,num_query_positions)

        # you can comment out this line to get self-attention ⬇️
        att = att.masked_fill(self.tril[:,:,:T,:T]==0,float('-inf'))  
        att  = F.softmax(att,dim=-1)
        att = self.dropout(att)
            # (B,T,n_head,n_head) @ (B,T,n_head,head_size) ->   (B,T,n_head,head_size)
        y = att @ v 
            # B,T,n_head,head_size -> B,T,C
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # This is our blender, wherein we mix the outputs of all heads before feeding into an MLP
        out = self.c_proj(y) 

        return out

class Block(nn.Module):
    
    def __init__(self,config:GPTconfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        # Since we do a ln and skip connection before an MLP, its essential for the outputs to be blended beforehand, else it'll be hard to backprop through
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, config:GPTconfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self. lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        # weight sharing scheme
        self.lm_head.weight = self.transformer.wte.weight # type: ignore

        #
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *=  (2*self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight,mean=0.0,std=std) #The 0.2 is still consistent with the Normal Xavier Initialization i.e 1/sqroot(fan_in)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of lenght {T} when block size is {self.config.block_size}"
        
        token_embd = self.transformer.wte(idx) # pyright: ignore[reportCallIssue]

        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        position_embd = self.transformer.wpe(pos) # pyright: ignore[reportCallIssue]

        x = token_embd + position_embd
        x = self.transformer.drop(x) # pyright: ignore[reportCallIssue]
        
        for block in self.transformer.h: # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
            x = block(x)
        
        x = self.transformer.ln_f(x) # pyright: ignore[reportCallIssue]

        logits = self.lm_head(x) # (B,T,Vocab_size)

        loss = None
    
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {model_type}")

         # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.tril')]

         # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class DataLoaderLite:

    def __init__(self,B:int,T:int,filename:str) -> None:
        self.B = B
        self.T = T

        with open(filename,'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 Epoch = {len(self.tokens)//(self.B*self.T)} batches")

        self.current_position = 0

    def next_batch(self)-> tuple[torch.Tensor,torch.Tensor]:
        
        B,T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position+(B*T)+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        
        self.current_position += (B*T)

        if self.current_position + (B*T)+1 >len(self.tokens):
            self.current_position = 0

        return (x , y)


# -----------------------------------------------------------------------------

num_return_sequences = 5
max_length = 30
num_iters = 20

model = GPT(GPTconfig())
model.to(device)
torch.compile(model)
train_loader = DataLoaderLite(16,256,'input.txt')

# optimise 
optim = torch.optim.AdamW(model.parameters(),lr=3e-4)

lossi = []

a = time.perf_counter()

for i in range(num_iters):
    t0 = time.perf_counter()

    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optim.zero_grad()


    logits,loss = model(x,y)
    # uncomment below do debug!
    # import code; code.interact(local=locals()) 
    loss.backward()
    optim.step()

    # waiting for the GPU/CPU to finish scheduled work!
    if device == 'cuda':
        torch.cuda.synchronize()
    # elif device == 'cpu':
    #     torch.cpu.synchronize()

    t1 = time.perf_counter()
    dt = (t1-t0) * 1000 # time diff in milliseconds
    tokens_per_sec = (train_loader.B*train_loader.T) / (t1-t0)
    
    lossi.append(loss.item())
    print(f"step: {i}, loss: {loss.item():.4f}, dt: {dt:.2f}ms, tok/sec:{tokens_per_sec:.1f}")


print(f"Final loss: {loss.item():.3f}")

print(f"It took {time.perf_counter()-a} seconds to run {num_iters} iterations")

# plots
plt.title(f"Loss visualised over {num_iters} iterations!")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.plot(range(num_iters),lossi);plt.show()

import sys; sys.exit(0)

# -----------------------------------------------------------------------------


# model = GPT.from_pretrained('gpt2')
model = GPT(GPTconfig())
model.eval()

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long) #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) 

x = tokens.to(device)

torch.manual_seed(42)
torch.mps.manual_seed(42)

t0= time.perf_counter()

while x.size(1) < max_length:
    with torch.no_grad():
        logits,_ = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,-1)
        
        topkprobs, topkindices = torch.topk(probs,50)
        ix = torch.multinomial(topkprobs,1)
        xcol = torch.gather(topkindices,-1,ix)

        x = torch.cat((x,xcol),dim=1)

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)


print('')
print(f"Its taking {(time.perf_counter()-t0):.2f} seconds for inference to run when device is set to {device}")
