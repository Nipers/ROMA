import torch
from transformers import RobertaTokenizer
from utils import attribute2id, prefix_type_id, suffix_type_id, img_type_id, img_token_id, item_encode_id, prefix_item_id, suffix_item_id, special_tokens_dict


class RecRobertaTokenizer(RobertaTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None):
        cls.config = config
        tokenizer = super().from_pretrained(pretrained_model_name_or_path)
        tokenizer.add_special_tokens(special_tokens_dict)
        assert img_token_id == tokenizer.get_added_vocab()['[img]'], f"utils: {img_token_id} / real:{tokenizer.get_added_vocab()['[img]']}"
        tokenizer.decoder[img_token_id] = "[img]"
        return tokenizer
        
    def __call__(self, items, pad_to_max=False, return_tensor=False):
        '''
        items: item sequence or a batch of item sequence, item sequence is a list of dict

        return:
        input_ids: token ids
        item_position_ids: the position of items
        token_type_ids: id for key or value
        attention_mask: local attention masks
        '''
        if len(items)>0 and isinstance(items[0], list): # batched items
            inputs = self.batch_encode(items, pad_to_max=pad_to_max)
        else:
            inputs = self.encode(items)

        if return_tensor:

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v)

        return inputs

    def item_tokenize(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def encode_item(self, item):
        
        input_ids = []
        token_type_ids = []

        for attr_name in attribute2id:
            attr_value = item[attr_name]
            if attr_value == "":
                attr_value = "void"
            name_tokens = self.item_tokenize(attr_name)
            value_tokens = self.item_tokenize(attr_value)

            attr_tokens = name_tokens + value_tokens
            attr_type_ids = [attribute2id[attr_name]] * len(name_tokens)
            attr_type_ids += [attribute2id[attr_name]] * len(value_tokens)

            attr_tokens = attr_tokens[:self.config.max_attr_length]
            attr_type_ids = attr_type_ids[:self.config.max_attr_length]
            input_ids += attr_tokens
            token_type_ids += attr_type_ids

        return input_ids, token_type_ids

    def encode(self, items, encode_item=True, add_prefx_and_suffix = False, pos = None, use_img=False, adp_idx=None):
        '''
        Encode a sequence of items.
        the order of items:  [past...present]
        return: [present...past]
        '''
        items = items[::-1]  # reverse items order
        items = items[:self.config.max_item_embeddings - 5] # truncate the number of items, -1 for <s>

        input_ids = [self.bos_token_id]
        item_position_ids = [0]
        token_type_ids = [0]
        prefix = f"Here is a sequence of items interacted by a customer in the mixed domain: "
        suffix = f", next item interacted by this customer should be: "
        if add_prefx_and_suffix:
            prefix_input_ids = self.item_tokenize(prefix)
            item_position_ids += [prefix_item_id] * len(prefix_input_ids)
            token_type_ids += [prefix_type_id] * len(prefix_input_ids)
            input_ids += prefix_input_ids
        seq_len = len(items)
        for item_idx, item in enumerate(items):
            if encode_item:            
                item_input_ids, item_token_type_ids = self.encode_item(item)
            else:
                item_input_ids, item_token_type_ids = item
            if len(input_ids) + len(item_input_ids) + int(use_img) <= self.config.max_token_num:
                input_ids += item_input_ids
                token_type_ids += item_token_type_ids
                input_ids += [img_token_id] * int(use_img)
                token_type_ids += [img_type_id] * int(use_img)
                if pos == None:
                    item_position_ids += [item_idx + 1] * (len(item_input_ids) + int(use_img)) 
                else:
                    item_position_ids += [pos] * (len(item_input_ids) + int(use_img)) 
            else:
                seq_len = item_idx
                break
        input_ids = input_ids[:self.config.max_token_num]
        item_position_ids = item_position_ids[:self.config.max_token_num]
        token_type_ids = token_type_ids[:self.config.max_token_num]
        
        if add_prefx_and_suffix:
            suffix_input_ids = self.item_tokenize(suffix)
            item_position_ids += [suffix_item_id] * len(suffix_input_ids)
            token_type_ids += [suffix_type_id] * len(suffix_input_ids)
            input_ids += suffix_input_ids
        attention_mask = [1] * len(input_ids)

        res = {
            "input_ids": input_ids,
            "item_position_ids": item_position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "seq_len": seq_len
        }
        if adp_idx != None:
            res["adp_idx"] = adp_idx
        return res

    def padding(self, item_batch, pad_to_max=False):

        if pad_to_max:
            max_length = self.config.max_token_num
        else:
            max_length = max([len(items["input_ids"]) for items in item_batch])      

        batch_input_ids = []
        batch_item_position_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []
        batch_seq_len = []
        batch_adp_idx = []

        for items in item_batch:

            input_ids = items["input_ids"]
            item_position_ids = items["item_position_ids"]
            token_type_ids = items["token_type_ids"]
            attention_mask = items["attention_mask"]
            if "adp_idx" in items:
                batch_adp_idx.append(items["adp_idx"])
            seq_len = items["seq_len"]
            length_to_pad = max_length - len(input_ids)

            input_ids += [self.pad_token_id] * length_to_pad
            item_position_ids += [self.config.max_item_embeddings - 1] * length_to_pad
            token_type_ids += [self.config.token_type_size - 1] * length_to_pad
            attention_mask += [0] * length_to_pad

            batch_input_ids.append(input_ids)
            batch_item_position_ids.append(item_position_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_seq_len.append(seq_len)
        res = {
            "input_ids": batch_input_ids,
            "item_position_ids": batch_item_position_ids,
            "token_type_ids": batch_token_type_ids,
            "attention_mask": batch_attention_mask,
            "batch_seq_len": batch_seq_len,
        }
        if len(batch_adp_idx) != 0:
            res["adp_idx"] = batch_adp_idx
        return res

    def encode_batch(self, tokenized_items, batch_item_seq, padding=False, add_prefx_and_suffix = False, pos = None, use_img=False, batch_adp_idx = None):        
        encoded_features = []
        adp_idx = None
        for seq_idx, item_seq in enumerate(batch_item_seq):
            if batch_adp_idx != None:
                adp_idx = batch_adp_idx[seq_idx]
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids = tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids])
            encoded_features.append(self.encode(feature_seq, encode_item=False, add_prefx_and_suffix = add_prefx_and_suffix, pos=pos, use_img=use_img, adp_idx=adp_idx))
        if padding:
            return self.padding(encoded_features, False)
        return encoded_features

    def batch_encode(self, item_batch, encode_item=True, pad_to_max=False, add_prefx_and_suffix = False, pos = None, use_img=False, batch_adp_idx = None):
        adp_idx = None
        encoded_batch = []
        for seq_idx, items in enumerate(item_batch):
            if batch_adp_idx != None:
                adp_idx = batch_adp_idx[seq_idx]
            encoded_batch.append(self.encode(items, encode_item=encode_item, add_prefx_and_suffix = add_prefx_and_suffix, pos=pos, use_img=use_img, adp_idx=adp_idx))
        return self.padding(encoded_batch, pad_to_max)
