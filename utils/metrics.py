import torch
import torch.nn as nn
from torchmetrics import Metric
from .utils import MAX_VAL
class PretrainEvalMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("hit", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="mean")

    def update(self, correct_num, total_num, loss):
        self.hit += correct_num
        self.total += total_num
        self.loss += loss

    def compute(self):
        return self.hit.float() / self.total.float(), self.hit.int(), self.total.int(), self.loss.float()

class LocalMetric(Metric):
    # used to calculate ndcg@k and hit@k
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.add_state("ndcg", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("hit", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        
    def update(self, scores, labels):
        # B X item num, B
        labels = labels.squeeze()   
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        rank = (predicts < scores).sum(-1).float()
        indicator = (rank < self.k).float()
        self.ndcg += ((1 / torch.log2(rank + 2)) * indicator).sum()
        self.hit += indicator.sum()
        self.total += scores.size(0)

    def compute(self):
        return self.ndcg.float() / self.total.float(), self.hit.float() / self.total.float()
    

class GlobalMetric(Metric):
    # used to calculate mrr and auc
    def __init__(self):
        super().__init__()
        self.add_state("mrr", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("auc", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        
    def update(self, scores, labels, loss):
        # B X item num, B
        labels = labels.squeeze()    
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float() + 1
        self.mrr += (1 / rank).sum().item()
        self.auc += (1 - (rank / valid_length)).sum().item()
        self.total += scores.size(0)
        self.loss += loss

    def compute(self):
        return self.mrr.float() / self.total.float(), self.auc.float() / self.total.float(), self.loss.float() / self.total.float()
