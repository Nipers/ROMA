import torch
import torch.nn as nn
import unicodedata
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

attribute2id = {"category":1, "title": 2, "brand":3}
prefix_type_id = 4
suffix_type_id = 5
img_type_id = 6
img_token_id = 50265
item_encode_id = 49
prefix_item_id = 50
suffix_item_id = 51
special_tokens_dict = {'additional_special_tokens': ['[img]']}
PRETRAIN = "pretrain"
FINETUNE = "finetune"
def split_seqs(batch_item_ids, reverse=False):
    batch_item_seq = []
    labels = []
    batch_adp_idxes = []
    for item_ids in batch_item_ids: 
        if "adp_idx" in item_ids:
            batch_adp_idxes.append(item_ids["adp_idx"])
        item_ids = item_ids['items']
        if reverse:
            batch_item_seq.append(item_ids[:-1][::-1])
        else:
            batch_item_seq.append(item_ids[:-1])
        labels.append(item_ids[-1])
    if len(batch_adp_idxes) > 0:
        return batch_item_seq, labels, batch_adp_idxes
    return batch_item_seq, labels

def split_pretrain_seqs(batch_item_ids):
    batch_item_seq = []
    labels = []
    for item_ids in batch_item_ids:
        batch_item_seq.append(item_ids[:-1])
        labels.append([item_ids[-1]])
    return batch_item_seq, labels

def create_position_ids_from_input_ids(input_ids, padding_idx, item_pos_offset = None):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    if item_pos_offset != None:
        incremental_indices[incremental_indices != 0] += item_pos_offset
    return incremental_indices.long() + padding_idx
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, (1 - float(current_step) / float(max(1, num_training_steps)))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def create_optimizer_and_scheduler(model: nn.Module, num_train_optimization_steps, args):
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config):
        super().__init__()
        self.temp = config.temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    

def is_subword(tokenizer, token: str):
    if (
        not tokenizer.convert_tokens_to_string(token).startswith(" ")
        and not is_punctuation(token[0])
    ):
        return True
    
    return False

def is_punctuation(char: str):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False



def encode_batch(tokenized_items, batch_item_seq, tokenizer):

    encoded_features = []
    for item_seq in batch_item_seq:
        feature_seq = []
        for item in item_seq:
            input_ids, token_type_ids = tokenized_items[item]
            feature_seq.append([input_ids, token_type_ids])
        encoded_features.append(tokenizer.encode(feature_seq, encode_item=False))
    return encoded_features

import json
import torch
import torch.nn as nn

MAX_VAL = 1e4

def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        if as_int:
            data = dict((int(key), value) for (key, value) in raw.items())
        else:
            data = dict((key, value) for (key, value) in raw.items())
        del raw
        return data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels):
        labels = labels.squeeze()
        
        try:
            loss = self.ce(scores, labels).item()
        except:
            print(scores.size())
            print(labels.size())
            loss = 0.0
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC

        return res + [loss]

