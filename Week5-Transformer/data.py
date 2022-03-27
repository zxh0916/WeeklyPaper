from collections import Counter, OrderedDict, defaultdict
import torchtext
from d2l import torch as d2l
from torchtext.vocab import vocab
import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def en_zh_split(text):
    en, zh = [], []
    for line in text:
        line = line.split('\t')
        en.append(line[0])
        zh.append(line[1])
    return en, zh


def tokenize(lines, mode='en'):
    if mode == 'en':
        return [line.split(' ') for line in lines]
    elif mode == 'zh':
        return [list(line) for line in lines]
    
    
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


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


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


class NMTDataset(Dataset):
    def __init__(self, max_len_en, max_len_zh, source='en', min_freq=1):
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
    
