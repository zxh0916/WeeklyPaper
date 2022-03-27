import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
        
        
class MultiHeadAttention(nn.Module):
    """实现多头点积注意力"""
    def __init__(self, d_model, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.d = d_model // num_head
        self.scale = self.d ** -0.5
        if d_model % num_head != 0:
            print('!!!!Warning: d_model % num_head != 0!!!!')
        self.linear_q = nn.Linear(d_model, self.d * num_head, bias=False)
        self.linear_k = nn.Linear(d_model, self.d * num_head, bias=False)
        self.linear_v = nn.Linear(d_model, self.d * num_head, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)
        self.output_layer = nn.Linear(self.d * num_head, d_model, bias=False)
        initialize_weight(self.output_layer)
    
    def sequence_mask(self, X, valid_len, value):
        """根据有效长度将注意力分数矩阵每个query的末尾元素用掩码覆盖"""
        maxlen = X.shape[3]
        # [1, 1, d] + [batch_size, num_query, 1] -> [batch_size, num_query, d]
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, None, :] >= valid_len[:, :, None]
        # shape of mask: [batch_size, num_heads, num_query, d]
        mask = mask.unsqueeze(1).repeat(1, X.shape[1], 1, 1)
        X[mask] = value
        return X
        
    def masked_softmax(self, X, valid_len):
        """带掩码的softmax，有效长度可以是每个batch一个（适用于编码器），
        也可以是每个query一个（适用于训练时的解码器自注意力）"""
        if valid_len is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            # 有效长度是一维向量:对批量中的每个样本指定一个有效长度
            # 有效长度是二维张量:对批量中每个样本的每个query都指定一个有效长度
            assert(valid_len.dim() in [1, 2])
            if valid_len.dim() == 1:
                valid_len = valid_len.reshape(-1, 1).repeat(1, X.shape[2])
            X = self.sequence_mask(X, valid_len, value=-1e9)
            return nn.functional.softmax(X, dim=-1)
        
    def forward(self, q, k, v, valid_len):
        d = self.d
        batch_size = q.shape[0]
        assert(k.shape[1] == v.shape[1])
        if valid_len is not None:
            assert(valid_len.shape[0] == batch_size)
        
        q = self.linear_q(q).reshape(batch_size, -1, self.num_head, d)
        k = self.linear_k(k).reshape(batch_size, -1, self.num_head, d)
        v = self.linear_v(v).reshape(batch_size, -1, self.num_head, d)
        
        # [batch_size, #q/#k/#v, num_heads, d] -> [batch_size, num_heads, #q/#k/#v, d]
        q, v, k = q.transpose(1, 2), v.transpose(1, 2), k.transpose(1, 2)
        
        # [batch_size, num_heads, #q, #k/#v]
        x = torch.matmul(q, (k.transpose(2, 3))) * self.scale
        x = self.masked_softmax(x, valid_len)
        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, -1, self.num_head * self.d)
        x = self.output_layer(x)
        
        return x
    
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()

        self.layer1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_size, d_model)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    
class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout, inplace=True)
        
    def forward(self, res, x):
        return self.norm(res + self.dropout(x))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden_size, dropout, num_head):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_head)
        self.add_norm1 = Add_Norm(d_model, dropout)
        self.ffn = FeedForwardNetwork(d_model, ffn_hidden_size)
        self.add_norm2 = Add_Norm(d_model, dropout)
    
    def forward(self, x, enc_valid_len):
        y = self.self_attention(x, x, x, enc_valid_len)
        y = self.add_norm1(x, y)
        z = self.ffn(y)
        z = self.add_norm2(y, z)
        return z
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden_size, dropout, num_head, i):
        super().__init__()
        self.i = i
        
        self.self_attention = MultiHeadAttention(d_model, num_head)
        self.add_norm1 = Add_Norm(d_model, dropout)
        
        self.enc_dec_attention = MultiHeadAttention(d_model, num_head)
        self.add_norm2 = Add_Norm(d_model, dropout)
        
        self.ffn = FeedForwardNetwork(d_model, ffn_hidden_size)
        self.add_norm3 = Add_Norm(d_model, dropout)
        
    def forward(self, x, state):
        
        enc_output, enc_valid_len = state[0], state[1]
        if state[2][self.i] is None:
            self_kv = x
        else:
            self_kv = torch.concat([state[2][self.i], x], dim=1)
        state[2][self.i] = self_kv
        
        if self.training:
            batch_size, num_steps, d = x.shape
            # 训练时，一个样本中有效长度应该与query在序列中的位置相等
            # shape of `dec_valid_len`: [batch_size, num_steps]
            dec_valid_len = torch.arange(1, num_steps+1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_len = None

        y = self.self_attention(x, self_kv, self_kv, dec_valid_len)
        y = self.add_norm1(x, y)
        z = self.enc_dec_attention(y, enc_output, enc_output, enc_valid_len)
        z = self.add_norm2(y, z)
        out = self.ffn(z)
        out = self.add_norm3(z, out)
        return out, state
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.P = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        
    def forward(self, X):
        # print(X.shape, self.P.shape)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
    
class Encoder(nn.Module):
    def __init__(self, N, d_model, ffn_hidden_size, dropout, num_head, vocab_size):
        super().__init__()
        self.emb_scale = d_model ** 0.5
        self.embedding = nn.Embedding(vocab_size, d_model)
        # nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden_size, dropout, num_head) for _ in range(N)])
        
    def forward(self, x, enc_valid_len):
        out = self.positional_encoding(self.embedding(x) * self.emb_scale)
        for m in self.layers:
            out = m(out, enc_valid_len)
        return out
    
    
class Decoder(nn.Module):
    def __init__(self, N, d_model, ffn_hidden_size, dropout, num_head, vocab_size):
        super().__init__()
        self.emb_scale = d_model ** 0.5
        self.embedding = nn.Embedding(vocab_size, d_model)
        # nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden_size, dropout, num_head, i) for i in range(N)])
        
    def forward(self, x, state):
        out = self.positional_encoding(self.embedding(x) * self.emb_scale)
        for m in self.layers:
            out, state = m(out, state)
        # shape of `self.embedding.weight`: [vocab_size, d_model]
        out = torch.matmul(out, self.embedding.weight.T)
        return out, state
    
    def init_state(self, enc_output, enc_valid_len):
        N = len(self.layers)
        return [enc_output, enc_valid_len, [None for _ in range(N)]]
    
    
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 N,
                 d_model,
                 ffn_hidden_size,
                 dropout,
                 num_head):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.ffn_hidden_size = ffn_hidden_size
        self.num_head = num_head
        self.dropout = dropout
        
        self.encoder = Encoder(N, d_model, ffn_hidden_size, dropout, num_head, src_vocab_size)
        self.decoder = Decoder(N, d_model, ffn_hidden_size, dropout, num_head, trg_vocab_size)
    
    def forward(self, inputs, enc_valid_len, targets):
        enc_outputs = self.encoder(inputs, enc_valid_len)
        state = self.decoder.init_state(enc_outputs, enc_valid_len)
        out, state = self.decoder(targets, state)
        return out, state
    
    def print_num_params(self):
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} trainable parameters.')
        
        
