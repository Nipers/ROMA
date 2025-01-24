from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from model import RecRobertaTokenizer
import torch
import random
from utils import is_subword, split_seqs, split_pretrain_seqs
@dataclass
class RecRobertaPretrainCollator:
    tokenizer: RecRobertaTokenizer
    tokenized_items: Dict
    mlm_probability: float
    mode: str
    use_img: bool
    max_seq_limit: int
    

    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        seq_cont_input = None
        item_pred_input = None
        batch_item_seqs = []
        batch_adp_idxes = []
        for item_ids in batch_item_ids:    
            if "adp_idx" in item_ids:
                batch_adp_idxes.append(item_ids["adp_idx"])
            item_ids = item_ids["items"]  
            if self.mode == "train":
                item_seq_len = len(item_ids)
                start = (item_seq_len-1) // 2
                if start < 2:
                    start = 2
                target_pos = random.randint(start, item_seq_len)
                batch_item_seqs.append(item_ids[:target_pos])
            else:
                batch_item_seqs.append(item_ids)
        pred_batch_item_seq, labels = split_pretrain_seqs(batch_item_seqs)
        pred_batch = self.tokenizer.encode_batch(self.tokenized_items, pred_batch_item_seq, False, add_prefx_and_suffix = False)
        labels = self.tokenizer.encode_batch(self.tokenized_items, labels, False, add_prefx_and_suffix = False)
        if self.mode == "train":
            mlm_input_ids_s, mlm_labels_s = self.mask_mlm(pred_batch)
            mlm_input_ids_t, mlm_labels_t = self.mask_mlm(labels)
            pred_batch = self.tokenizer.padding(pred_batch)
            labels = self.tokenizer.padding(labels)
            
            item_pred_input = dict()
            for k, v in pred_batch.items():
                item_pred_input[k + "_s"] = torch.LongTensor(v)
            for k, v in labels.items():
                item_pred_input[k + "_t"] = torch.LongTensor(v)
            item_pred_input["mlm_input_ids_s"] = torch.LongTensor(mlm_input_ids_s)
            item_pred_input["mlm_input_ids_t"] = torch.LongTensor(mlm_input_ids_t)
            item_pred_input["mlm_labels_s"] = torch.LongTensor(mlm_labels_s)
            item_pred_input["mlm_labels_t"] = torch.LongTensor(mlm_labels_t)
        
        else:
            pred_batch = self.tokenizer.padding(pred_batch)
            labels = self.tokenizer.padding(labels)
            item_pred_input = dict()
            for k, v in pred_batch.items():
                item_pred_input[k + "_s"] = torch.LongTensor(v)
            for k, v in labels.items():
                item_pred_input[k + "_t"] = torch.LongTensor(v)
        if len(batch_adp_idxes) != 0:
            item_pred_input["adp_idx"] = torch.LongTensor(batch_adp_idxes)

        return item_pred_input, seq_cont_input

    def sample_pairs(self, batch_item_ids):
        batch_item_seq_a = []
        batch_item_seq_b = []

        for item_ids in batch_item_ids:
            item_seq_len = len(item_ids)
            start = (item_seq_len-1) // 2
            target_pos = random.randint(start, item_seq_len-1)
            batch_item_seq_a.append(item_ids[:target_pos])
            batch_item_seq_b.append([item_ids[target_pos]])

    def sample_seq_pairs(self, batch_item_ids):
        batch_item_seq_a = []
        batch_item_seq_b = []

        for item_ids in batch_item_ids:
            item_seq_len = len(item_ids)
            if item_seq_len == 2:
                target_pos = 1
            else:
                target_pos = random.randint(1, item_seq_len-1)
            batch_item_seq_a.append(item_ids[:target_pos])
            batch_item_seq_b.append(item_ids[target_pos:])

        return batch_item_seq_a, batch_item_seq_b


    def mask_mlm(self, flat_features):

        input_ids = [e["input_ids"] for e in flat_features]

        batch_input = self._collate_batch(input_ids)

        mask_labels = []
        for e in flat_features:
            ref_tokens = []
            for id in e["input_ids"]:
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = self._collate_batch(mask_labels)
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        return inputs, labels

    

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):

        cand_indexes = []

        for (i, token) in enumerate(input_tokens):

            if token == self.tokenizer.bos_token or token == self.tokenizer.eos_token:
                continue

            if is_subword(self.tokenizer, token) and len(cand_indexes) > 0:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _collate_batch(self, examples, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if self.tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

@dataclass
class VEMLMPretrainCollator(RecRobertaPretrainCollator):
    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        seq_cont_input = None
        item_pred_input = None
        batch_item_seqs = []
        batch_adp_idxes = []
        for item_ids in batch_item_ids:    
            if "adp_idx" in item_ids:
                batch_adp_idxes.append(item_ids["adp_idx"])
            item_ids = item_ids["items"]  
            batch_item_seqs.append(item_ids[-self.max_seq_limit:])
        pred_batch_item_seq, label_items = split_pretrain_seqs(batch_item_seqs)
        pred_batch = self.tokenizer.encode_batch(self.tokenized_items, pred_batch_item_seq, False, add_prefx_and_suffix = False, use_img=self.use_img)
        labels = self.tokenizer.encode_batch(self.tokenized_items, label_items, True, add_prefx_and_suffix = False, use_img=self.use_img)
        
        if self.mode == "train":
            mlm_input_ids_s, mlm_labels_s = self.mask_mlm(pred_batch)
            pred_batch = self.tokenizer.padding(pred_batch)
            input_ids = pred_batch.pop("input_ids")
            input_ids = torch.LongTensor(input_ids)
            assert mlm_input_ids_s.shape == input_ids.shape, f'mlm: {mlm_input_ids_s.shape} / input_ids: {input_ids.shape}'
            
            item_pred_input = dict()
            for k, v in pred_batch.items():
                if k != "batch_seq_len":
                    item_pred_input[k + "_s"] = torch.LongTensor(v)
            for k, v in labels.items():
                if k != "batch_seq_len":
                    item_pred_input[k + "_t"] = torch.LongTensor(v)
            item_pred_input["mlm_input_ids_s"] = torch.LongTensor(mlm_input_ids_s)
            item_pred_input["mlm_labels_s"] = torch.LongTensor(mlm_labels_s)
        
        else:
            pred_batch = self.tokenizer.padding(pred_batch)
            item_pred_input = dict()
            for k, v in pred_batch.items():
                if k != "batch_seq_len":
                    item_pred_input[k + "_s"] = torch.LongTensor(v)
            for k, v in labels.items():
                if k != "batch_seq_len":
                    item_pred_input[k + "_t"] = torch.LongTensor(v)
        if len(batch_adp_idxes) != 0:
            item_pred_input["adp_idx"] = torch.LongTensor(batch_adp_idxes)
        batch_seq_len_s = pred_batch.pop("batch_seq_len")
        batch_seq_len_t = labels.pop("batch_seq_len")
        batch_item_seq_ids_s = []
        batch_item_seq_ids_t = []
        for idx, seq in enumerate(pred_batch_item_seq):
            seq_len = batch_seq_len_s[idx]
            seq = seq[-seq_len:]
            seq = seq[::-1]
            batch_item_seq_ids_s += seq
        for idx, seq in enumerate(label_items):
            seq_len = batch_seq_len_t[idx]
            seq = seq[-seq_len:]
            seq = seq[::-1]
            batch_item_seq_ids_t += seq
        item_pred_input["batch_item_seq_ids_s"] = torch.LongTensor(batch_item_seq_ids_s)
        item_pred_input["batch_item_seq_ids_t"] = torch.LongTensor(batch_item_seq_ids_t)
        return item_pred_input, seq_cont_input

@dataclass
class RecRobertaFinetuneCollator:

    tokenizer: RecRobertaTokenizer
    tokenized_items: Dict
    mlm_probability:float
    use_mlm: bool
    use_img: bool
    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        batch_item_seq, labels = split_seqs(batch_item_ids)
        batch = self.tokenizer.encode_batch(self.tokenized_items, batch_item_seq, padding=True, add_prefx_and_suffix=False, use_img=self.use_img)
        batch["labels"] = labels
        batch_seq_len = batch.pop("batch_seq_len")
        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        batch_item_seq_ids = []
        for idx, seq in enumerate(batch_item_seq):
            seq_len = batch_seq_len[idx]
            seq = seq[-seq_len:]
            seq = seq[::-1]
            batch_item_seq_ids += seq
        batch["batch_item_seq_ids"] = torch.LongTensor(batch_item_seq_ids)
        if self.use_mlm:
            for i in range(len(batch_item_ids)):
                batch_item_ids[i] = batch_item_ids[i]['items']
                
            whole_seq = self.tokenizer.encode_batch(self.tokenized_items, batch_item_ids, padding=False, add_prefx_and_suffix=False, use_img=False)

            mlm_input_ids, mlm_labels = self.mask_mlm(whole_seq)
            whole_seq = self.tokenizer.padding(whole_seq)
            batch["mlm_input_ids"] = torch.LongTensor(mlm_input_ids)
            batch["mlm_labels"] = torch.LongTensor(mlm_labels)
            batch["token_type_ids_mlm"] = torch.LongTensor(whole_seq["token_type_ids"])
            batch["item_position_ids_mlm"] = torch.LongTensor(whole_seq["item_position_ids"])
            batch["attention_mask_mlm"] = torch.LongTensor(whole_seq["attention_mask"])

        return batch

    def sample_train_data(self, batch_item_ids):

        batch_item_seq = []
        labels = []

        for item_ids in batch_item_ids:

            item_ids = item_ids['items']
            item_seq_len = len(item_ids)
            start = min(item_seq_len, 1)
            target_pos = random.randint(start, item_seq_len-1)
            batch_item_seq.append(item_ids[:target_pos])
            labels.append(item_ids[target_pos])

        return batch_item_seq, labels

    def mask_mlm(self, flat_features):

        input_ids = [e["input_ids"] for e in flat_features]

        batch_input = self._collate_batch(input_ids)

        mask_labels = []
        for e in flat_features:
            ref_tokens = []
            for id in e["input_ids"]:
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = self._collate_batch(mask_labels)
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        return inputs, labels

    

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):

        cand_indexes = []

        for (i, token) in enumerate(input_tokens):

            if token == self.tokenizer.bos_token or token == self.tokenizer.eos_token:
                continue

            if is_subword(self.tokenizer, token) and len(cand_indexes) > 0:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels



    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


    def _collate_batch(self, examples, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if self.tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

@dataclass
class RecRobertaEvalCollator:

    tokenizer: RecRobertaTokenizer
    tokenized_items: Dict
    use_img: bool

    def __call__(self, batch_data: List[Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        batch_item_seq, labels = self.prepare_eval_data(batch_data)

        batch = self.tokenizer.encode_batch(self.tokenized_items, batch_item_seq, padding=True, add_prefx_and_suffix=False, use_img=self.use_img)

        batch_seq_len = batch.pop("batch_seq_len")
        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        batch_item_seq_ids = []
        for idx, seq in enumerate(batch_item_seq):
            seq_len = batch_seq_len[idx]
            seq = seq[-seq_len:]
            seq = seq[::-1]
            batch_item_seq_ids += seq
        batch["batch_item_seq_ids"] = torch.LongTensor(batch_item_seq_ids)
        labels = torch.LongTensor(labels)
        
        return batch, labels

    def prepare_eval_data(self, batch_data):

        batch_item_seq = []
        labels = []

        for data_line in batch_data:
            item_ids = data_line['items']
            label = data_line['label']
            
            batch_item_seq.append(item_ids)
            labels.append(label)

        return batch_item_seq, labels, 

@dataclass
class MOARecRobertaFinetuneCollator:

    tokenizer: RecRobertaTokenizer
    tokenized_items: Dict
    mlm_probability:float
    use_mlm: bool
    use_img: bool
    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        batch_item_seq, labels, batch_adp_idxes = split_seqs(batch_item_ids)
        batch = self.tokenizer.encode_batch(self.tokenized_items, batch_item_seq, padding=False, add_prefx_and_suffix=False, use_img=self.use_img)

        if self.use_mlm:
        
            mlm_input_ids, mlm_labels = self.mask_mlm(batch)
            batch = self.tokenizer.padding(batch)
            batch.pop("input_ids")
            batch["mlm_input_ids"] = mlm_input_ids
            batch["mlm_labels"] = mlm_labels
        else:
            batch = self.tokenizer.padding(batch)

        batch["labels"] = labels
        batch_seq_len = batch.pop("batch_seq_len")
        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        
        batch_item_seq_ids = []
        for idx, seq in enumerate(batch_item_seq):
            seq_len = batch_seq_len[idx]
            seq = seq[-seq_len:]
            seq = seq[::-1]
            batch_item_seq_ids += seq
        batch["batch_item_seq_ids"] = torch.LongTensor(batch_item_seq_ids)
        if len(batch_adp_idxes) > 0:
            batch["adp_idx"] = torch.LongTensor(batch_adp_idxes)

        return batch

    def sample_train_data(self, batch_item_ids):

        batch_item_seq = []
        labels = []

        for item_ids in batch_item_ids:

            item_ids = item_ids['items']
            item_seq_len = len(item_ids)
            start = min(item_seq_len, 1)
            target_pos = random.randint(start, item_seq_len-1)
            batch_item_seq.append(item_ids[:target_pos])
            labels.append(item_ids[target_pos])

        return batch_item_seq, labels

    def mask_mlm(self, flat_features):

        input_ids = [e["input_ids"] for e in flat_features]

        batch_input = self._collate_batch(input_ids)

        mask_labels = []
        for e in flat_features:
            ref_tokens = []
            for id in e["input_ids"]:
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = self._collate_batch(mask_labels)
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        return inputs, labels

    

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):

        cand_indexes = []

        for (i, token) in enumerate(input_tokens):

            if token == self.tokenizer.bos_token or token == self.tokenizer.eos_token:
                continue

            if is_subword(self.tokenizer, token) and len(cand_indexes) > 0:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100 

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _collate_batch(self, examples, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if self.tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

@dataclass
class MOARecRobertaEvalCollator:

    tokenizer: RecRobertaTokenizer
    tokenized_items: Dict
    use_img: bool

    def __call__(self, batch_data: List[Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        batch_item_seq, labels, batch_adp_idxes = self.prepare_eval_data(batch_data)

        batch = self.tokenizer.encode_batch(self.tokenized_items, batch_item_seq, padding=True, add_prefx_and_suffix=False, use_img=self.use_img)

        batch_seq_len = batch.pop("batch_seq_len")
        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        batch_item_seq_ids = []
        for idx, seq in enumerate(batch_item_seq):
            seq_len = batch_seq_len[idx]
            seq = seq[-seq_len:]
            seq = seq[::-1]
            batch_item_seq_ids += seq
        batch["batch_item_seq_ids"] = torch.LongTensor(batch_item_seq_ids)
        labels = torch.LongTensor(labels)
        if len(batch_adp_idxes) > 0:
            batch["adp_idx"] = torch.LongTensor(batch_adp_idxes)
        
        return batch, labels

    def prepare_eval_data(self, batch_data):

        batch_item_seq = []
        labels = []
        batch_adp_idxes = []

        for data_line in batch_data:
            if "adp_idx" in data_line:
                batch_adp_idxes.append(data_line["adp_idx"])

            item_ids = data_line['items']
            label = data_line['label']
            
            batch_item_seq.append(item_ids)
            labels.append(label)

        return batch_item_seq, labels, batch_adp_idxes

