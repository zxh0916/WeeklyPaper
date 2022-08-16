import torch.nn as nn
import torch
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)
    
class MultiHeadDotProductSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.):
        super().__init__()
        self.scale = emb_dim ** -0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        # shape of x: [batch_size, sequence_length, embedding_size]
        b, n, _, h = *x.shape, self.num_heads
        # shape of qkv: [batch_size, sequence_length, 3*embedding_size]
        qkv = self.qkv(x)
        # shape of q, k, v: [batch_size, sequence_length, embedding_size]
        q, k, v = qkv.chunk(3, dim=-1)
        # shape of q, k, v: [batch_size, num_heads, sequence_length, embedding_size / num_heads]
        q, k, v = map(lambda x:x.reshape(b, n, h, -1).transpose(2, 1), (q, k, v))
        # shape of attention_score: [batch_size, num_heads, sequence_length, sequence_length]
        attention_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        if mask is not None:
            # 最前面的[class]token永远得是有效的，所以在第一列pad一列True
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == attention_score.shape[-1], 'mask has incorrect dimensions'
            # Mask中同一行里的False只会出现在True后面!!
            mask = mask[:, None, :] * mask[:, :, None]
            attention_score.masked_fill_(~mask, float('-inf'))
            del mask
        
        attention_weight = F.softmax(attention_score, dim=-1)
        # shape of out: [batch_size, num_heads, sequence_length, embedding_size / num_heads]
        out = torch.matmul(attention_weight, v)
        out = out.transpose(2, 1).reshape(b, n, -1)
        # shape of out: [batch_size, sequence_length, embedding_size]
        out = self.out(out)
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm_input = nn.LayerNorm(emb_dim)
        self.layer_norm_out = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadDotProductSelfAttention(emb_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(emb_dim, hidden_dim, dropout)
        
    def forward(self, x):
        y = self.layer_norm_input(x)
        y = self.attention(y)
        y = y + x
        z = self.layer_norm_out(y)
        z = self.ffn(z)
        return z + y
    
class Encoder(nn.Module):
    def __init__(self, N, emb_dim, num_heads, hidden_dim, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderBlock(emb_dim, num_heads, hidden_dim, dropout) for _ in range(N)])
        
    def forward(self, x):
        return self.layers(x)
    
class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.N
        self.patch_size = cfg.patch_size
        self.emb_dim = cfg.emb_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.dropout = cfg.dropout
        self.image_size = cfg.image_size
        self.num_channels = cfg.num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_classes = cfg.num_classes
        assert self.image_size % self.patch_size == 0, 'image dimensions must be divisible by the patch size'
        
        self.class_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        self.embedding = nn.Linear(self.num_channels * self.patch_size ** 2, self.emb_dim)
        self.PE = nn.Parameter(torch.randn(1, self.num_patches+1, self.emb_dim))
        self.emb_dropout = nn.Dropout(self.dropout)
        self.transformer = Encoder(self.N, self.emb_dim, self.num_heads, self.hidden_dim, self.dropout)
        self.MLP_head = nn.Sequential(
            nn.Linear(self.emb_dim, 2*self.emb_dim),
            nn.Linear(2*self.emb_dim, self.num_classes))
        
    def forward(self, x, tokenize=True):
        if tokenize:
            x = self.tokenize(x)
        b = x.shape[0]
        x = self.embedding(x)
        cls_tokens = self.class_token.repeat(b, 1, 1)
        x = torch.concat((cls_tokens, x), dim=1)
        x += self.PE
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = self.MLP_head(x[:, 0, :])
        return x
    
    def tokenize(self, x):
        b, c, h, w = x.shape
        assert h == self.image_size and w == self.image_size, 'the size of the input image is incorrect'
        x = x.chunk(self.image_size // self.patch_size, dim=2)
        patches = []
        for patch in x:
            patches += patch.chunk(self.image_size // self.patch_size, dim=3)
        # shape of x: [batch_size, num_patches, num_channels * patch_size ** 2]
        x = torch.stack(patches, dim=1).reshape(b, self.num_patches, -1)
        return x
    
    def print_num_params(self):
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))
