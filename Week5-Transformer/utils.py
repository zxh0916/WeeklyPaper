import math
import torch
from collections import defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def idx2BLEU(output, target, vocab):
    output = idx2sentense(output, vocab)
    target = idx2sentense(target, vocab)
    # print(output[0], target[0])
    return BLEU(output, target)