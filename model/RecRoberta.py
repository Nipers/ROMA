import torch
import torch.nn as nn 
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
from transformers.utils import ModelOutput
import os
import numpy as np
from transformers.models.roberta.modeling_roberta import *
from config import RecRobertaConfig
from utils import Similarity, create_position_ids_from_input_ids, img_type_id
from .RecRobertaTokenizer import RecRobertaTokenizer
from faiss import IndexFlatIP, normalize_L2, StandardGpuResources, index_cpu_to_gpu
@dataclass
class RecRobertaMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
from tqdm import tqdm

class RecRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: RecRobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.token_type_size, config.hidden_size, padding_idx=config.token_type_size - 1)
        self.item_position_embeddings = nn.Embedding(config.max_item_embeddings, config.hidden_size, padding_idx=config.max_item_embeddings - 1)
        self.item_num = config.item_num
        self.use_img = config.use_img
        if self.use_img:
            img_emb_table = np.load(config.img_emb_path)
            img_marks = np.load(config.img_marks_path)
            init_dim = img_emb_table.shape[1]
            self.img_proj_layer = nn.Sequential(
                nn.Linear(init_dim, init_dim * 2),
                nn.GELU(),
                nn.Linear(init_dim * 2, config.hidden_size)
            )
            self.img_emb_table = nn.Embedding.from_pretrained(torch.FloatTensor(img_emb_table), freeze=True)
            self.img_marks = torch.BoolTensor(img_marks)

            self.void_img_emb = nn.Parameter(torch.randn((1, config.hidden_size), dtype=torch.float32))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, item_position_ids=None, inputs_embeds=None, batch_item_seq_ids = None, use_img=False):
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        dtype = inputs_embeds.dtype
        if use_img:
            img_indice = torch.arange(0, self.item_num, 1, device=device, dtype=torch.long)
            img_emb_table = self.img_proj_layer(self.img_emb_table(img_indice)).to(device=device, dtype=dtype)
            img_emb_table[~self.img_marks] = self.void_img_emb.to(device=device, dtype=dtype)
            img_embs = img_emb_table[batch_item_seq_ids]
            img_token_location = (token_type_ids == img_type_id)

            assert img_token_location.sum().item() == img_embs.size(0), f"{img_token_location.sum().item()}-{img_embs.size(0)}"

            inputs_embeds[img_token_location] = img_embs
        position_embeddings = self.position_embeddings(position_ids)
        item_position_embeddings = self.item_position_embeddings(item_position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + item_position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor inputs_embeds:
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
    

class RecRobertaPooler(nn.Module):
    def __init__(self, config: RecRobertaConfig):
        super().__init__()
        self.pooler_type = config.pooler_type

    def forward(self, attention_mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        output = None
        if self.pooler_type == 'cls':
            output = hidden_states[:, 0]
        elif self.pooler_type == "avg":
            output = ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError
        return output
    

class RecRobertaModel(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = RecRobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RecRobertaPooler(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        batch_item_seq_ids: Optional[bool] = None,
        use_img: bool = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            batch_item_seq_ids = batch_item_seq_ids,
            use_img = use_img,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(attention_mask, sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

@dataclass
class RecRobertaPretrainingOutput:
    
    cl_correct_num: float = 0.0
    cl_total_num: float = 1e-5
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RecRobertaForPretrain(RobertaPreTrainedModel):
    def __init__(self, config: RecRobertaConfig):
        super().__init__(config)

        self.roberta = RecRobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.sim = Similarity(config)
        self.loss_fct = CrossEntropyLoss()
        # Initialize weights and apply final processing
        self.padding_idx = config.pad_token_id
        self.item_pos_offset = config.item_pos_offset
        self.mlm_weight = config.mlm_weight
        self.seq_cont_weight = config.seq_cont_weight
        self.post_init()

    def item_seq_contrast(self,        
        input_ids_s: Optional[torch.Tensor] = None,
        attention_mask_s: Optional[torch.Tensor] = None,
        token_type_ids_s: Optional[torch.Tensor] = None,
        item_position_ids_s: Optional[torch.Tensor] = None,
        position_ids_s: Optional[torch.Tensor] = None,
        input_ids_t: Optional[torch.Tensor] = None,
        attention_mask_t: Optional[torch.Tensor] = None,
        token_type_ids_t: Optional[torch.Tensor] = None,
        item_position_ids_t: Optional[torch.Tensor] = None,
        position_ids_t: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        mlm_input_ids_s: Optional[torch.Tensor] = None,
        mlm_input_ids_t: Optional[torch.Tensor] = None,
        mlm_labels_s: Optional[torch.Tensor] = None,
        mlm_labels_t: Optional[torch.Tensor] = None,
        return_correct = False,
    ):
        
        batch_size = input_ids_s.size(0)
        outputs_s = self.roberta(
            input_ids_s,
            attention_mask=attention_mask_s,
            token_type_ids=token_type_ids_s,
            position_ids=position_ids_s,
            item_position_ids=item_position_ids_s,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if return_correct:
            # item pos ids starts from item_pos_offset
            position_ids_t = create_position_ids_from_input_ids(input_ids_t, self.padding_idx, self.item_pos_offset)
        outputs_t = self.roberta(
            input_ids_t,
            attention_mask=attention_mask_t,
            token_type_ids=token_type_ids_t,
            position_ids=position_ids_t,
            item_position_ids=item_position_ids_t,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        z1 = outputs_s.pooler_output  # (bs*num_sent, hidden_size)
        z2 = outputs_t.pooler_output  # (bs*num_sent, hidden_size)

        if dist.is_initialized() and self.training:
            
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss = self.loss_fct(cos_sim, labels)

        if mlm_labels_s != None:
            mlm_labels_s = mlm_labels_s.view(-1, mlm_labels_s.size(-1))
            mlm_labels_t = mlm_labels_t.view(-1, mlm_labels_t.size(-1))
            mlm_outputs_s = self.roberta(
                mlm_input_ids_s,
                attention_mask=attention_mask_s,
                token_type_ids=token_type_ids_s,
                position_ids=position_ids_s,
                item_position_ids=item_position_ids_s,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            mlm_outputs_t = self.roberta(
                mlm_input_ids_t,
                attention_mask=attention_mask_t,
                token_type_ids=token_type_ids_t,
                position_ids=position_ids_t,
                item_position_ids=item_position_ids_t,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            prediction_scores_s = self.lm_head(mlm_outputs_s.last_hidden_state)
            prediction_scores_t = self.lm_head(mlm_outputs_t.last_hidden_state)
            mlm_loss_s = self.loss_fct(prediction_scores_s.view(-1, self.config.vocab_size), mlm_labels_s.view(-1))
            mlm_loss_t = self.loss_fct(prediction_scores_t.view(-1, self.config.vocab_size), mlm_labels_t.view(-1))
            mlm_loss_s += mlm_loss_t

            loss += mlm_loss_s * self.mlm_weight
        if return_correct:
            correct_num = (torch.argmax(cos_sim, 1) == labels).sum()
            return RecRobertaPretrainingOutput(
                loss=loss,
                logits=cos_sim,
                cl_correct_num=correct_num,
                cl_total_num=batch_size
            )
        return RecRobertaPretrainingOutput(
            loss=loss,
        )

    def mask_token_prediction(
        self,
        mlm_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        is_item = False
    ):  
        if is_item:
            position_ids = create_position_ids_from_input_ids(mlm_input_ids, self.padding_idx, self.item_pos_offset)

        mlm_outputs = self.roberta(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
        loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        return RecRobertaPretrainingOutput(
            loss=loss
        )
    
    
    def forward(self, batch):
        item_pred_input, seq_cont_input= batch
        if item_pred_input != None:
            output1 = self.item_seq_contrast(**item_pred_input, return_correct=True)
        else:
            output1 = None
        if seq_cont_input != None:
            output2 = self.item_seq_contrast(**seq_cont_input, return_correct=False)
            if output1 != None:
                output1.loss += output2.loss * self.seq_cont_weight
            else:
                output1 = output2
        return output1



class RecRobertaForSeqRec(RobertaPreTrainedModel):
    def __init__(self, config: RecRobertaConfig):
        super().__init__(config)

        self.roberta = RecRobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.sim = Similarity(config)
        self.use_mlm = config.use_mlm
        self.use_img = config.use_img
        self.mlm_weight = config.mlm_weight
        self.loss_fct = CrossEntropyLoss()
        self.item_embedding = None
        self.post_init()

    def update_item_embedding(self, tokenizer: RecRobertaTokenizer, tokenized_items, batch_size, cache_file=None):
        if cache_file is not None:
            if os.path.exists(cache_file) == True:
                print(f"Use cached item embedding from {cache_file}")
                self.item_embedding = nn.Embedding.from_pretrained(torch.load(cache_file).to(self.device), freeze=True).to(self.device)
                return
            else:
                print(f"File not exist: {cache_file}")
        
        self.eval()
        items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
        item_ids = [ele[0] for ele in items]
        items = [ele[1] for ele in items]
        item_embedding = []
        with torch.no_grad():
            for i in tqdm(range(0, len(items), batch_size), ncols=100, desc='Encode all items'):
                item_batch = [[item] for item in items[i:i+batch_size]]
                batch_item_seq_ids = item_ids[i:i+batch_size]
                inputs = tokenizer.batch_encode(item_batch, encode_item=False, pad_to_max=False)
                inputs.pop("batch_seq_len")
                for k, v in inputs.items():
                    inputs[k] = torch.LongTensor(v).to(self.device)
                inputs["batch_item_seq_ids"] = torch.LongTensor(batch_item_seq_ids).to(self.device)
                outputs = self.roberta(**inputs)
                item_embedding.append(outputs.pooler_output.detach())

        item_embedding = torch.cat(item_embedding, dim=0)#.cpu()
        if cache_file != None:
            print(f"Save item embedding into cached {cache_file}")
            torch.save(item_embedding.cpu(), cache_file)

        self.item_embedding = nn.Embedding.from_pretrained(item_embedding, freeze=True)

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is None:
            candidate_embeddings = self.item_embedding.weight.unsqueeze(0) # (1, num_items, hidden_size)
        else:
            candidate_embeddings = self.item_embedding(candidates) # (batch_size, candidates, hidden_size)
        pooler_output = pooler_output.unsqueeze(1) # (batch_size, 1, hidden_size)
        return self.sim(pooler_output, candidate_embeddings)


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                item_position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                candidates: Optional[torch.Tensor] = None, # candidate item ids
                labels: Optional[torch.Tensor] = None, # target item ids
                batch_item_seq_ids: Optional[bool] = None,
                mlm_input_ids: Optional[bool] = None,
                mlm_labels: Optional[torch.Tensor] = None,
                token_type_ids_mlm: Optional[torch.Tensor] = None,
                item_position_ids_mlm: Optional[torch.Tensor] = None,
                attention_mask_mlm: Optional[torch.Tensor] = None,
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            batch_item_seq_ids=batch_item_seq_ids,
            use_img = self.use_img,
        )
        
        pooler_output = outputs.pooler_output # (bs, hidden_size)

        if labels is None:
            return self.similarity_score(pooler_output, candidates)


        if self.config.finetune_negative_sample_size<=0: 
            logits = self.similarity_score(pooler_output)
            loss = self.loss_fct(logits, labels)

        else:  
            candidates = torch.cat((labels.unsqueeze(-1), torch.randint(0, self.config.item_num, size=(batch_size, self.config.finetune_negative_sample_size)).to(labels.device)), dim=-1)
            logits = self.similarity_score(pooler_output, candidates)
            target = torch.zeros_like(labels, device=labels.device)
            loss = self.loss_fct(logits, target)

        if self.use_mlm:
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            mlm_outputs = self.roberta(
                mlm_input_ids,
                attention_mask=attention_mask_mlm,
                token_type_ids=token_type_ids_mlm,
                position_ids=position_ids,
                item_position_ids=item_position_ids_mlm,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                use_img=False,
            )
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            mlm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            loss += self.mlm_weight * mlm_loss
        return loss
    
class RERecRoberta4Rec(RobertaPreTrainedModel):
    def __init__(self, config: RecRobertaConfig):
        super().__init__(config)

        self.roberta = RecRobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.sim = Similarity(config)
        self.use_mlm = config.use_mlm
        self.use_img = config.use_img
        self.mlm_weight = config.mlm_weight
        self.loss_fct = CrossEntropyLoss()
        self.max_item_embeddings = config.max_item_embeddings
        self.padding_idx = config.pad_token_id
        self.item_pos_offset = config.item_pos_offset
        self.item_embedding = None
        self.post_init()
        self.index = None
        self.diff_pos = config.diff_pos
           
    def update_item_embedding(self, tokenizer: RecRobertaTokenizer, tokenized_items, batch_size, cache_file=None, build_index = False):
        if cache_file is not None:
            if os.path.exists(cache_file) == True:
                print(f"Use cached item embedding from {cache_file}")
                self.item_embedding = nn.Embedding.from_pretrained(torch.load(cache_file).to(self.device), freeze=True).to(self.device)
                return
            else:
                print(f"File not exist: {cache_file}")
        
        self.eval()
        items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
        item_ids = [ele[0] for ele in items]
        items = [ele[1] for ele in items]
        item_embedding = []
        with torch.no_grad():
            for i in tqdm(range(0, len(items), batch_size), ncols=100, desc='Encode all items'):
                item_batch = [[item] for item in items[i:i+batch_size]]
                batch_item_seq_ids = item_ids[i:i+batch_size]
                if self.diff_pos:
                    pos = self.max_item_embeddings - 2
                    inputs = tokenizer.batch_encode(item_batch, encode_item=False, pad_to_max=False, pos=pos)
                    if self.item_pos_offset != 0:
                        inputs["position_ids"] = create_position_ids_from_input_ids(inputs["input_ids"], self.padding_idx, self.item_pos_offset)
                else:
                    inputs = tokenizer.batch_encode(item_batch, encode_item=False, pad_to_max=False)
                inputs.pop("batch_seq_len")
                for k, v in inputs.items():
                    inputs[k] = torch.LongTensor(v).to(self.device)
                inputs["batch_item_seq_ids"] = torch.LongTensor(batch_item_seq_ids).to(self.device)
                outputs = self.roberta(**inputs)
                item_embedding.append(outputs.pooler_output.detach())

        item_embedding = torch.cat(item_embedding, dim=0)#.cpu()
        if cache_file != None:
            print(f"Save item embedding into cached {cache_file}")
            torch.save(item_embedding.cpu(), cache_file)
        if build_index:            
            item_emb = item_embedding.cpu().detach().numpy()
            item_emb = np.array(item_emb)
            normalize_L2(item_emb)
            self.index = IndexFlatIP(self.config.hidden_size)
            if dist.is_initialized():
                res = StandardGpuResources()
                res.setTempMemory(256 * 1024 * 1024)
                local_rank = dist.get_rank()
                self.index = index_cpu_to_gpu(res, local_rank, self.index)
            self.index.add(item_emb)

        self.item_embedding = nn.Embedding.from_pretrained(item_embedding, freeze=True)

    def similarity_score(self, pooler_output, candidates = None, tokenizer: RecRobertaTokenizer=None, tokenized_items=None):
        # candidiates: batch_size * (1 + finetune_negative_sample_size)
        if tokenizer != None:
            batch_candidates = candidates.tolist()
            batch_candidate_emb = []
            items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
            items = [ele[1] for ele in items]
            for candidate in batch_candidates:
                cand_items = [[items[item_idx]] for item_idx in candidate]
                # inputs = tokenizer.batch_encode(cand_items, encode_item=False, pad_to_max=False)
                if self.diff_pos:
                    pos = self.max_item_embeddings - 2
                    inputs = tokenizer.batch_encode(cand_items, encode_item=False, pad_to_max=False, pos=pos)
                    if self.item_pos_offset != 0:
                        inputs["position_ids"] = create_position_ids_from_input_ids(inputs["input_ids"], self.padding_idx, self.item_pos_offset)
                else:
                    inputs = tokenizer.batch_encode(cand_items, encode_item=False, pad_to_max=False)
                inputs.pop("batch_seq_len")

                for k, v in inputs.items():
                    inputs[k] = torch.LongTensor(v).to(self.device)
                inputs["batch_item_seq_ids"] = torch.LongTensor(candidate).to(self.device)
                outputs = self.roberta(**inputs)
                batch_candidate_emb.append(outputs.pooler_output)
            batch_candidate_emb = torch.stack(batch_candidate_emb, dim=0) # (batch_size, candidates, hidden_size)     
        else:
            batch_candidate_emb = self.item_embedding.weight.unsqueeze(0) # (1, num_items, hidden_size)
        pooler_output = pooler_output.unsqueeze(1) # (batch_size, 1, hidden_size)
        return self.sim(pooler_output, batch_candidate_emb)

    def get_candidate(self, seq_rep:np.ndarray, labels:np.ndarray):
        batch_size = labels.shape[0]
        if seq_rep != None:
            normalize_L2(seq_rep)
            _, candidates = self.index.search(seq_rep, self.config.finetune_negative_sample_size + 1)
        else:
            candidates = []
            for _ in range(batch_size):
                candidates.append(np.random.choice(self.config.item_num, size=(1, self.config.finetune_negative_sample_size + 1), replace=False))
            candidates = np.concatenate(candidates, axis=0)
        labels = labels.reshape(-1, 1)
        hit = (candidates == labels)
        hit_row = hit.any(axis = -1)
        candidates[hit] = candidates[hit_row][:,0]
        candidates[:,0] = labels.reshape(-1)
        return candidates


    def forward(self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            item_position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None, # target item ids
            batch_item_seq_ids: Optional[bool] = None,
            tokenizer: RecRobertaTokenizer = None, 
            tokenized_items = None,
        ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)

        seq_rep = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            batch_item_seq_ids=batch_item_seq_ids,
            use_img = self.use_img,
        )
        
        pooler_output = seq_rep.pooler_output # (bs, hidden_size)
        if tokenizer is not None and tokenized_items is not None:
            candidates = []
            for _ in range(batch_size):
                candidates.append(np.random.choice(self.config.item_num, size=(1, self.config.finetune_negative_sample_size + 1), replace=False))
            candidates = np.concatenate(candidates, axis=0)
            candidates[:,0] = labels.cpu().detach().numpy().reshape(-1)

            logits = self.similarity_score(pooler_output, candidates, tokenizer, tokenized_items)
            labels = torch.zeros_like(labels, device=labels.device)
        else:
            logits = self.similarity_score(pooler_output, None, None, None)

        if labels is None: 
            return logits
        else:
            return self.loss_fct(logits, labels)