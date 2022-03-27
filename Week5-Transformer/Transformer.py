#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from collections import Counter, OrderedDict, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from d2l import torch as d2l
from torchtext.vocab import vocab
from torchtext.data.functional import generate_sp_model
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


import re
def read_data_nmt():    
    with open('cmn-simplified.txt', 'r',
             encoding='utf-8') as f:
        raw_text = f.readlines()
    text = []
    def preprocess_nmt(text):
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '
        # 使用空格替换不间断空格
        # 使用小写字母替换大写字母
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # 在单词和标点符号之间插入空格
        # out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        #        for i, char in enumerate(text)]
        out = re.sub('[,.?:!，。？：！]', '', text)
        return ''.join(out)
    
    for line in raw_text:
        line = line.split('CC-BY 2.0')
        text.append(preprocess_nmt(line[0]))
    return text


# In[3]:


text = read_data_nmt()
print(len(text))
for i in range(5):
    print(text[i])


# In[4]:


def en_zh_split(text):
    en, zh = [], []
    for line in text:
        line = line.split('\t')
        en.append(line[0])
        zh.append(line[1])
    return en, zh

text_en, text_zh = en_zh_split(text)
print(text_en[:5], text_en[:5])


# In[5]:


def tokenize(lines, mode='en'):
    if mode == 'en':
        return [line.split(' ') for line in lines]
    elif mode == 'zh':
        return [list(line) for line in lines]


# In[6]:


tokenized_en = tokenize(text_en, mode='en')
print(tokenized_en[:3], tokenized_en[-2:])
tokenized_zh = tokenize(text_zh, mode='zh')
print(tokenized_zh[:3], tokenized_zh[-2:])


# In[7]:


def get_vocab(tokens, min_freq):
    def count_corpus(tokens):
        # 这里的 `tokens` 是 1D 列表或 2D 列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        return Counter(tokens)
    
    counter = count_corpus(tokens)
    token_freqs = sorted(counter.items(), key=lambda x: x[1],
                               reverse=True)
    ordered_dict = OrderedDict(token_freqs)
    Vocab = vocab(ordered_dict, min_freq, specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True)
    Vocab.set_default_index(Vocab['<unk>'])
    return Vocab


# In[8]:


vocab_en = get_vocab(tokens=tokenized_en, min_freq=2)
vocab_zh = get_vocab(tokens=tokenized_zh, min_freq=2)


# In[9]:


print(len(vocab_en))
print(len(vocab_zh))
print(vocab_en.lookup_tokens([i for i in range (10)]))
print(vocab_zh.lookup_tokens([i for i in range (10)]))
print(vocab_zh(list('愿指引明路的苍蓝星为你闪耀！')+['<unk>', '<pad>', '<bos>', '<eos>']))


# In[10]:


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


# In[11]:


truncate_pad(vocab_en(tokenized_en[5000]), 10, vocab_en['<pad>'])


# In[12]:


def build_array_nmt(lines, vocab, num_steps, label=False):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab(l) for l in lines]
    if not label:
        lines = [l + [vocab['<eos>']] for l in lines]
    else:
        lines = [[vocab['<bos>']] + l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# In[13]:


array, valid_len = build_array_nmt(tokenized_en[:5], vocab_en, 5, True)
print(array, valid_len)
print(tokenized_en[:5])


# In[14]:


class NMTDataset(Dataset):
    def __init__(self, max_len_en, max_len_zh, source='en', min_freq=0):
        super().__init__()
        self.data = self.raw_dataset(max_len_en, max_len_zh, source, min_freq)
        
    def __len__(self):
        return self.data[0].shape[0]
    
    def __getitem__(self, index):
        return [self.data[i][index] for i in range(len(self.data))]
    
    def raw_dataset(self, max_len_en, max_len_zh, source, min_freq):
        text = read_data_nmt()
        text_en, text_zh = en_zh_split(text)
        tokenized_en = tokenize(text_en, mode='en')
        tokenized_zh = tokenize(text_zh, mode='zh')
        self.vocab_en = get_vocab(tokens=tokenized_en, min_freq=min_freq)
        self.vocab_zh = get_vocab(tokens=tokenized_zh, min_freq=min_freq)
        if source == 'en':
            enc_input, enc_valid_len = build_array_nmt(tokenized_en, self.vocab_en, max_len_en, False)
            target, dec_valid_len = build_array_nmt(tokenized_zh, self.vocab_zh, max_len_zh, False)
            dec_input, _ = build_array_nmt(tokenized_zh, self.vocab_zh, max_len_zh, True)
        elif source == 'zh':
            enc_input, enc_valid_len = build_array_nmt(tokenized_zh, self.vocab_zh, max_len_zh, False)
            target, dec_valid_len = build_array_nmt(tokenized_en, self.vocab_en, max_len_en, False)
            dec_input, _ = build_array_nmt(tokenized_en, self.vocab_en, max_len_en, True)
        return enc_input, enc_valid_len, dec_input, target, dec_valid_len


# In[15]:


dataset_test = NMTDataset(7, 7, 'en')
print(dataset_test[28446])
len(dataset_test)


# In[16]:


a = torch.tensor([1,2,3])
print(type(a))
list(a)


# In[17]:


def idx2sentense(indices, vocab):
    
    if isinstance(indices[0], torch.Tensor) and indices[0].dim() >= 1:
        return [idx2sentense(sentense, vocab) for sentense in indices]
    elif isinstance(indices, torch.Tensor):
        
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
        # special_tokens = []
        sentense = []
        for index in indices:
            if vocab.lookup_token(int(index)) not in special_tokens:
                sentense.append(vocab.lookup_token(int(index)))
            if vocab.lookup_token(int(index)) == '<eos>':
                break
        return sentense


# In[18]:


print(idx2sentense([dataset_test[i][0] for i in range(10)], vocab_en))
print(idx2sentense(dataset_test[500][0], vocab_en))
print(idx2sentense([dataset_test[i][2] for i in range(10)], vocab_zh))
print(idx2sentense(dataset_test[500][2], vocab_zh))


# In[19]:


def BLEU(pred_seq, label_seq, k=4):
    if pred_seq == []:
        return 0
    if isinstance(pred_seq[0], list):
        assert(len(pred_seq) == len(label_seq))
        scores = [BLEU(pred, label) for (pred, label) in zip(pred_seq, label_seq)]
        return sum(scores) / len(scores)
    len_pred, len_label = len(pred_seq), len(label_seq)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        if n > len_pred:
            break
        num_matches, label_subs = 0, defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_seq[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_seq[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_seq[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# In[20]:


def idx2BLEU(output, target, vocab):
    output = idx2sentense(output, vocab)
    target = idx2sentense(target, vocab)
    # print(output[0], target[0])
    return BLEU(output, target)


# In[21]:


sentenses = [dataset_test[i][0] for i in range(10)]
idx2BLEU(sentenses, sentenses, vocab_en)


# In[22]:


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


# In[23]:


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


# In[24]:


atten = MultiHeadAttention(10, 2)
x = torch.randn(3, 5, 10)
valid_len = torch.tensor([[2,3,1,2,3],
                          [1,3,2,1,3],
                          [3,2,1,3,2]])
valid_len2 = torch.tensor([2,3,1])
print(atten(x,x,x, valid_len).shape)
print(atten(x,x,x, valid_len2).shape)


# In[25]:


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


# In[26]:


class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout, inplace=True)
        
    def forward(self, res, x):
        return self.norm(res + self.dropout(x))


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[30]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


net = Transformer(3, 3, 3, 10, 20, 0, 2)
# print(net.encoder)
# print(net.decoder)
test_input = torch.zeros(3, 120, dtype=torch.long)
valid_len = torch.tensor([30, 5, 70])
print(net.encoder(test_input, valid_len).shape)
test_target = torch.zeros(3, 50, dtype=torch.long)
print(net(test_input, valid_len, test_target)[0].shape)


# In[34]:


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带掩码的softmax交叉熵损失函数"""
    # `pred` 的形状：(`batch_size`, `num_steps`, `vocab_size`)
    # `label` 的形状：(`batch_size`, `num_steps`)
    # `valid_len` 的形状：(`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = self.sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss / pred.shape[0]
    
    def sequence_mask(self, X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X


# In[35]:


dataloader_test = DataLoader(dataset_test, batch_size=5, shuffle=True)
for batch in dataloader_test:
    for thing in batch:
        print(thing)
    break


# In[36]:


def warmup_LR(d_model, warmup_steps, cur_step):
    if cur_step == 0:
        return 0
    lr = d_model ** -0.5
    lr *= min(cur_step ** -0.5, (cur_step * warmup_steps ** -1.5))
    return lr


# In[37]:


def train_transformer(net, data_iter, base_lr, warmup_steps, num_iters, source_vocab, target_vocab, device):
    # optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
    get_lr = lambda cur_step: warmup_LR(net.d_model, warmup_steps, cur_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    criterion = MaskedSoftmaxCELoss()
    net.train()
    iter_counter = 1
    
    # writer = SummaryWriter(f'runs/Transformer_N={net.N}_head={net.num_head}_d={net.d_model}_ffn={net.ffn_hidden_size}_dropout={net.dropout}')
    
    timer = d2l.Timer()
    metric = d2l.Accumulator(3)  # 训练损失总和，词元数量
    while True:
        for batch in data_iter:
            train = iter_counter < num_iters
            optimizer.zero_grad()
            enc_input, enc_valid_len, dec_input, target, dec_valid_len = [x.to(device) for x in batch]
            output, _ = net(enc_input, enc_valid_len, dec_input)
            l = criterion(output, target, dec_valid_len)
            l.sum().backward() # 损失函数的标量进行“反向传播”
            # d2l.grad_clipping(net, 1)
            num_tokens = dec_valid_len.sum()
            optimizer.step()
            iter_counter += 1
            scheduler.step()
            with torch.no_grad():
                pred = torch.argmax(output, dim=-1)
                # print(pred[0])
                bleu = idx2BLEU(pred, target, target_vocab)
                metric.add(l.sum(), num_tokens, bleu)
            # writer.add_scalar(float(l.sum()), 'train/loss', global_step=iter_counter)
            # writer.add_scalar(bleu, 'train/BLEU', global_step=iter_counter)
            # writer.add_scalar(optimizer.state_dict()['param_groups'][0]['lr'], 
            #                   'learning rate', global_step=iter_counter)
            if iter_counter % 100 == 0:
                print(f'iter {iter_counter:6d}, loss = {l.sum().item():8.4f}, bleu = {bleu:.8f}')
                print(idx2sentense(pred[0], target_vocab), '\n',
                      idx2sentense(dec_input[0], target_vocab), '\n',
                      idx2sentense(enc_input[0], source_vocab))
            if not train:
                break
        if not train:
            break
    print(f'loss {metric[0] / metric[1]:.3f}, BLEU {metric[2] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')


# In[67]:


def predict(net, src_sentense, src_vocab, trg_vocab, num_steps, source='en'):
    if isinstance(src_sentense, list):
        return [predict(net, sentense, src_vocab, trg_vocab, num_steps, source) for sentense in src_sentense]
    net.eval()
    if source == 'en':
        src_tokens = src_vocab(src_sentense.lower().split(' ') + ['<eos>'])
    else:
        src_tokens = src_vocab(list(src_sentense) + ['<eos>'])
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor(
        [trg_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == trg_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ''.join(trg_vocab.lookup_tokens(output_seq))


# In[39]:


dataset_zh2en = NMTDataset(max_len_en=20, max_len_zh=20, min_freq=1, source='zh')
dataloader_zh2en = DataLoader(dataset_zh2en, batch_size=256, shuffle=True, num_workers=0)
# dataset_en2zh = NMTDataset(max_len_en=20, max_len_zh=20, min_freq=1, source='en')
# dataloader_en2zh = DataLoader(dataset_en2zh, batch_size=256, shuffle=True, num_workers=0)


# In[40]:


net_zh2en = Transformer(src_vocab_size=len(dataset_zh2en.vocab_zh),
                        trg_vocab_size=len(dataset_zh2en.vocab_en),
                        N=4,
                        d_model=128,
                        ffn_hidden_size=256,
                        dropout=0.1,
                        num_head=4).to(device)
net_zh2en.print_num_params()
# net_en2zh = Transformer(src_vocab_size=len(dataset_en2zh.vocab_en),
#                         trg_vocab_size=len(dataset_en2zh.vocab_zh),
#                         N=4,
#                         d_model=128,
#                         ffn_hidden_size=256,
#                         dropout=0,
#                         num_head=4).to(device)
# net_en2zh.print_num_params()


# In[ ]:


train_transformer(net_zh2en,
                  data_iter=dataloader_zh2en,
                  base_lr=1e1,
                  warmup_steps=3000,
                  num_iters=100000,
                  source_vocab=dataset_zh2en.vocab_zh,
                  target_vocab=dataset_zh2en.vocab_en,
                  device=device)
# train_transformer(net_en2zh,
#                   data_iter=dataloader_en2zh,
#                   base_lr=1e1,
#                   warmup_steps=1000,
#                   num_iters=10000,
#                   source_vocab=dataset_en2zh.vocab_en,
#                   target_vocab=dataset_en2zh.vocab_zh,
#                   device=device)


# In[ ]:


torch.save(net_zh2en, f'Transformer_N={net_zh2en.N}_head={net_zh2en.num_head}_d={net_zh2en.d_model}_ffn={net_zh2en.ffn_hidden_size}_dropout={net_zh2en.dropout}_zh2en.pth')
# torch.save(net_en2zh.state_dict(), f'Transformer_N={net.N}_head={net.num_head}_d={net.d_model}_ffn={net.ffn_hidden_size}_dropout={net.dropout}_en2zh.pth')


# In[ ]:


net_zh2en = torch.load(f'Transformer_N={net_zh2en.N}_head={net_zh2en.num_head}_d={net_zh2en.d_model}_ffn={net_zh2en.ffn_hidden_size}_dropout={net_zh2en.dropout}_zh2en.pth')


# In[ ]:


test_input = ['我可以去', 
              '早上好', 
              '你今天过得怎么样', 
              '什么时候去吃晚饭',
              '不错',
              '学英语',
              '汤姆很喜欢吃东西',
              '现在几点了',
              '今天下雨',
              '明天下雨',
              '你打算怎么去学校']
predict(net_zh2en,
        src_sentense=test_input,
        src_vocab=dataset_zh2en.vocab_zh,
        trg_vocab=dataset_zh2en.vocab_en,
        num_steps=20,
        source='zh')


# In[ ]:




