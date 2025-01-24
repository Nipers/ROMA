from torch.utils.data import Dataset
from .RecRobertaCollator import (
    RecRobertaPretrainCollator,
    RecRobertaFinetuneCollator,
    RecRobertaEvalCollator
)
from typing import List


class RecRobertaPretrainDataset(Dataset):
    def __init__(self, dataset: List, collator: RecRobertaPretrainCollator):
        super().__init__()

        self.collator = collator
        self.seqs = []
        for data in dataset:
            if len(data) > 1:
                self.seqs.append(data[-10:])
    
    def preprocess(self, dataset):
        self.seqs = []
        self.unbroken = []
        for seq in dataset:
            seq = seq[-100:]
            target_pos = len(seq)
            if target_pos < 2:
                continue
            if target_pos < 6:
                self.seqs.append(seq)
                self.unbroken.append(True)
                continue
            for end in range(target_pos, 4, -1):
                self.seqs.append(seq[:end])
                self.unbroken.append(end == target_pos)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):     
        if self.collator.mode == "train":  
            return self.seqs[index] 
        else:
            return self.seqs[index]

    def collate_fn(self, data):
        return self.collator([{'items': line} for line in data])


class RecRobertaTrainDataset(Dataset):
    def __init__(self, user2train, collator: RecRobertaFinetuneCollator):

        '''
        user2train: dict of sequence data, user--> item sequence
        '''
        
        self.user2train = user2train
        self.collator = collator
        self.users = sorted(user2train.keys())
        self.preprocess()
    
    def preprocess(self):
        self.seqs = []
        for user in self.users:
            seq = self.user2train[user]
            target_pos = len(seq)
            if target_pos < 2:
                continue
            for end in range(target_pos, 1, -1):
                self.seqs.append(seq[:end])
            

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):

        return self.seqs[index][-20:]

    def collate_fn(self, data):

        return self.collator([{'items': line} for line in data])


class RecRobertaEvalDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, mode, collator: RecRobertaEvalCollator, align = False):
        self.user2train = user2train
        self.user2val = user2val
        self.user2test = user2test
        self.collator = collator

        if mode == "val":
            self.users = list(self.user2val.keys())
        else:
            self.users = list(self.user2test.keys())

        self.mode = mode
        self.align = align

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.user2train[user] if self.mode == "val" else self.user2train[user] + self.user2val[user]
        label = self.user2val[user] if self.mode == "val" else self.user2test[user]
        if self.align:
            if len(seq) >= 3:
                if self.mode == "val":
                    seq = seq[1:]
                if self.mode == "test":
                    seq = seq[2:]
                
        return seq[-49:], label[0]

    def collate_fn(self, data):
        return self.collator([{'items': line[0], 'label': line[1]} for line in data])



class MOARecRobertaPretrainDataset(Dataset):
    def __init__(self, seqs: list, adp_idxes:list, collator: RecRobertaPretrainCollator, last1item:list=None, max_seq_len = 50):
        super().__init__()

        self.collator = collator
        if self.collator.mode == "dev" and last1item != None:
            for i in range(len(seqs)):
                seqs[i] += last1item[i]
        self.seqs = seqs
        self.adp_idxes = adp_idxes
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):       
            return self.seqs[index][-self.max_seq_len:], self.adp_idxes[index]

    def collate_fn(self, data):
        return self.collator([{'items': line[0], "adp_idx": line[1]} for line in data])

class MOARecRobertaTrainDataset(Dataset):
    def __init__(self, user2train, collator: RecRobertaFinetuneCollator):

        '''
        user2train: dict of sequence data, user--> item sequence
        '''
        
        self.user2train = user2train
        self.collator = collator
        self.users = sorted(user2train.keys())
        self.preprocess()
    
    def preprocess(self):
        self.seqs = []
        for user in self.users:
            seq = self.user2train[user]
            target_pos = len(seq)
            if target_pos < 2:
                continue
            for end in range(target_pos, 1, -1):
                self.seqs.append(seq[:end])
            
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):

        return self.seqs[index][-10:]

    def collate_fn(self, data):

        return self.collator([{'items': line, "adp_idx": 0} for line in data])

class MOARecRobertaEvalDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, mode, collator: RecRobertaEvalCollator, align = False):
        self.user2train = user2train
        self.user2val = user2val
        self.user2test = user2test
        self.collator = collator

        if mode == "val":
            self.users = list(self.user2val.keys())
        else:
            self.users = list(self.user2test.keys())

        self.mode = mode
        self.align = align

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.user2train[user] if self.mode == "val" else self.user2train[user] + self.user2val[user]
        label = self.user2val[user] if self.mode == "val" else self.user2test[user]
        if self.align:
            if len(seq) >= 3:
                if self.mode == "val":
                    seq = seq[1:]
                if self.mode == "test":
                    seq = seq[2:]
                
        return seq[-9:], label[0]

    def collate_fn(self, data):
        return self.collator([{'items': line[0], 'label': line[1], "adp_idx": 0} for line in data])
