import math
from collections import Counter, OrderedDict, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from d2l import torch as d2l
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data import NMTDataset, truncate_pad
from module import Transformer
from utils import idx2sentense, BLEU, idx2BLEU

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
    
def warmup_LR(d_model, warmup_steps, cur_step):
    if cur_step == 0:
        return 0
    lr = d_model ** -0.5
    lr *= min(cur_step ** -0.5, (cur_step * warmup_steps ** -1.5))
    return lr

def evaluate(net, valid_data_iter, target_vocab, label_smoothing):
    net.eval()
    valid_loss, valid_bleu = [], []
    criterion = MaskedSoftmaxCELoss(label_smoothing=label_smoothing)
    with torch.no_grad():
        for batch in valid_data_iter:
            enc_input, enc_valid_len, dec_input, target, dec_valid_len = [x.to(device) for x in batch]
            output, _ = net(enc_input, enc_valid_len, dec_input)
            loss = criterion(output, target, dec_valid_len)
            pred = torch.argmax(output, dim=-1)
            bleu = idx2BLEU(pred, target, target_vocab)
            valid_loss.append(loss.sum().item())
            valid_bleu.append(bleu)
    avg = lambda lst: sum(lst) / len(lst)
    return avg(valid_loss), avg(valid_bleu)

def train_transformer(net,
                      train_data_iter,
                      valid_data_iter,
                      base_lr,
                      label_smoothing,
                      warmup_steps,
                      num_iters,
                      source_vocab,
                      target_vocab,
                      device,
                      mode):

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
    get_lr = lambda cur_step: warmup_LR(net.d_model, warmup_steps, cur_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    criterion = MaskedSoftmaxCELoss(label_smoothing=label_smoothing)
    net.train()
    iter_counter = 1
    
    writer = SummaryWriter(f'runs/Transformer_N={net.N}_head={net.num_head}_d={net.d_model}_ffn={net.ffn_hidden_size}_dropout={net.dropout}_num_iters={num_iters}_warmup_steps={warmup_steps}_lr={base_lr}_label_smoothing={label_smoothing}_{mode}')
    while True:
        net.train()
        for batch in train_data_iter:
            train = iter_counter < num_iters
            optimizer.zero_grad()
            enc_input, enc_valid_len, dec_input, target, dec_valid_len = [x.to(device) for x in batch]
            output, _ = net(enc_input, enc_valid_len, dec_input)
            l = criterion(output, target, dec_valid_len)
            l.sum().backward() # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = dec_valid_len.sum()
            optimizer.step()
            iter_counter += 1
            scheduler.step()
            with torch.no_grad():
                pred = torch.argmax(output, dim=-1)
                # print(pred[0])
                bleu = idx2BLEU(pred, target, target_vocab)
            writer.add_scalar('train/loss', float(l.sum().item()), global_step=iter_counter)
            writer.add_scalar('train/BLEU', bleu, global_step=iter_counter)
            writer.add_scalar('learning rate', 
                              optimizer.state_dict()['param_groups'][0]['lr'], 
                              global_step=iter_counter)
            if iter_counter % 100 == 0:
                print(f'TRAIN iter {iter_counter:6d}, loss = {l.sum().item():8.4f}, bleu = {bleu:.8f}')
                print(idx2sentense(pred[0], target_vocab), '\n',
                      idx2sentense(dec_input[0], target_vocab), '\n',
                      idx2sentense(enc_input[0], source_vocab))
            if not train:
                break
        # 每个batch结束之后验证一次
        valid_loss, valid_BLEU = evaluate(net, valid_data_iter, target_vocab, label_smoothing)
        writer.add_scalar('valid/loss', valid_loss, global_step=iter_counter)
        writer.add_scalar('valid/BLEU', valid_BLEU, global_step=iter_counter)
        print(f'VALID iter {iter_counter:6d}, loss = {valid_loss:8.4f}, bleu = {valid_BLEU:.8f}')
        torch.save(net, f'Transformer_N={net.N}_head={net.num_head}_d={net.d_model}_ffn={net.ffn_hidden_size}_dropout={net.dropout}_{mode}.pth')
        if not train:
            break
    
    
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
    if source == 'en':
        return ''.join(trg_vocab.lookup_tokens(output_seq))
    elif source == 'zh':
        return ' '.join(trg_vocab.lookup_tokens(output_seq))


if __name__ == '__main__':
    max_len_en=20
    max_len_zh=20
    min_freq=1
    batch_size=256
    valid_ratio = 0.08
    N=2
    d_model=32
    ffn_hidden_size=64
    dropout=0.1
    num_head=4
    base_lr=1e1
    label_smoothing=0.0
    warmup_steps=1000
    num_iters=30000
    num_workers=4
    
    mode = 'zh2en'
    # mode = 'en2zh'
    
    if mode == 'zh2en':
        dataset_zh2en = NMTDataset(max_len_en=max_len_en, max_len_zh=max_len_zh, min_freq=min_freq, source='zh')
        train_dataset_zh2en, valid_dataset_zh2en = \
            random_split(dataset_zh2en, [len(dataset_zh2en) - int(len(dataset_zh2en) * valid_ratio), 
                                         int(len(dataset_zh2en) * valid_ratio)])
        train_data_iter = DataLoader(train_dataset_zh2en,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        valid_data_iter = DataLoader(valid_dataset_zh2en,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        net_zh2en = Transformer(src_vocab_size=len(dataset_zh2en.vocab_zh),
                                trg_vocab_size=len(dataset_zh2en.vocab_en),
                                N=N,
                                d_model=d_model,
                                ffn_hidden_size=ffn_hidden_size,
                                dropout=dropout,
                                num_head=num_head).to(device)
        net_zh2en.print_num_params()
        train_transformer(net_zh2en,
                          train_data_iter=train_data_iter,
                          valid_data_iter=valid_data_iter,
                          base_lr=base_lr,
                          label_smoothing=label_smoothing,
                          warmup_steps=warmup_steps,
                          num_iters=num_iters,
                          source_vocab=dataset_zh2en.vocab_zh,
                          target_vocab=dataset_zh2en.vocab_en,
                          device=device,
                          mode=mode)
        torch.save(net_zh2en, f'Transformer_N={net_zh2en.N}_head={net_zh2en.num_head}_d={net_zh2en.d_model}_ffn={net_zh2en.ffn_hidden_size}_dropout={net_zh2en.dropout}_zh2en.pth')
        
    else:
        dataset_en2zh = NMTDataset(max_len_en=max_len_en, max_len_zh=max_len_zh, min_freq=min_freq, source='en')
        train_dataset_en2zh, valid_dataset_en2zh = \
            random_split(dataset_en2zh, [len(dataset_en2zh) - int(len(dataset_en2zh) * valid_ratio), 
                                         int(len(dataset_en2zh) * valid_ratio)])
        train_data_iter = DataLoader(train_dataset_en2zh,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        valid_data_iter = DataLoader(valid_dataset_en2zh,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        net_en2zh = Transformer(src_vocab_size=len(dataset_en2zh.vocab_en),
                                trg_vocab_size=len(dataset_en2zh.vocab_zh),
                                N=N,
                                d_model=d_model,
                                ffn_hidden_size=ffn_hidden_size,
                                dropout=dropout,
                                num_head=num_head).to(device)
        net_en2zh.print_num_params()
        train_transformer(net_en2zh,
                          train_data_iter=train_data_iter,
                          valid_data_iter=valid_data_iter,
                          base_lr=base_lr,
                          label_smoothing=label_smoothing,
                          warmup_steps=warmup_steps,
                          num_iters=num_iters,
                          source_vocab=dataset_en2zh.vocab_en,
                          target_vocab=dataset_en2zh.vocab_zh,
                          device=device,
                          mode=mode)
        torch.save(net_en2zh, f'Transformer_N={net_en2zh.N}_head={net_en2zh.num_head}_d={net_en2zh.d_model}_ffn={net_en2zh.ffn_hidden_size}_dropout={net_en2zh.dropout}_en2zh.pth')
        