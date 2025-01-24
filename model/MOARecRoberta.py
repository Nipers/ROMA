import torch
import torch.nn as nn 
import torch.distributed as dist
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
import os
from transformers.models.roberta.modeling_roberta import *
from transformers.activations import ACT2FN
from config import RecRobertaConfig
from utils import Similarity, create_position_ids_from_input_ids, FINETUNE
from .RecRobertaTokenizer import RecRobertaTokenizer
from tqdm import tqdm
from .RecRoberta import RecRobertaEmbeddings, RecRobertaPooler, RecRobertaPretrainingOutput
import torch.nn.functional as F
import numpy as np
from faiss import normalize_L2

class BaseGate(nn.Module):
    def __init__(self, num_adapter):
        super().__init__()
        self.num_adapter = num_adapter
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None
    
class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which adapters the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_adapter, top_k=2):
        super().__init__(num_adapter)
        self.gate = nn.Linear(d_model, self.num_adapter)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LoRAAdapter(nn.Module):
    def __init__(self, config, use_gate):
        super().__init__()
        self.num_adapter = config.num_adapter
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.adp_intermediate_size
        self.dense1 = nn.Linear(config.hidden_size, self.intermediate_size * self.num_adapter)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.ModuleList([nn.Linear(self.intermediate_size, config.hidden_size) for _ in range(self.num_adapter)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter_top_k = config.adapter_top_k
        self.scaling = 2
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = NaiveGate(config.hidden_size, config.num_adapter, config.adapter_top_k)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dense1.bias)
        for adp_idx in range(self.num_adapter):
            nn.init.zeros_(self.dense2[adp_idx].weight)
            nn.init.zeros_(self.dense2[adp_idx].bias)
        if self.use_gate:
            nn.init.kaiming_uniform_(self.gate.gate.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate.gate.bias)

    def forward(self, input_tensor: torch.Tensor, domain_idx:torch.Tensor=None) -> torch.Tensor:

        B, L = input_tensor.shape[:2]

        input_tensor = self.dropout(input_tensor)
        hidden_states = self.dense1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        # batch_size x seq_len x num_adapter x self_inter
        new_shape = hidden_states.size()[:-1] + (self.num_adapter, self.intermediate_size)
        # num_adapter x batch_size x seq_len x self_inter
        hidden_states = hidden_states.view(new_shape).transpose(2, 1).transpose(1, 0)
        adp_output = []
        for adp_idx in range(self.num_adapter):
            adp_output.append(self.dense2[adp_idx](hidden_states[adp_idx]))
        # batch_size x num_adapter x seq_len x hidden_size
        adp_output = torch.stack(adp_output, dim=0).transpose(1, 0)
        if self.use_gate:
            # (batch_size x seq_len) x num_adapter x hidden_size
            adp_output = adp_output.transpose(2, 1).view(-1, self.num_adapter, self.hidden_size)
            gate_top_k_idx, _ = self.gate(input_tensor)
            gate_top_k_idx = gate_top_k_idx.view(-1, self.adapter_top_k).transpose(0,1)
            batch_idx = torch.arange(adp_output.shape[0], dtype = torch.long).to(adp_output.device)
            adp_output = adp_output[batch_idx, gate_top_k_idx].transpose(0, 1)
            adp_output = adp_output.sum(dim=-2) / self.adapter_top_k
            # batch_size x seq_len x hidden_size
            adp_output = adp_output.view(B, L, -1)
        else:
            assert domain_idx != None and domain_idx.shape[0] == B, f"domain_idx: {domain_idx.shape}"
            domain_idx = domain_idx.reshape(1, B)
            batch_idx = torch.arange(B, dtype = torch.long).to(adp_output.device)
            adp_output = adp_output[batch_idx, domain_idx].squeeze(0)

        adp_output *= self.scaling
        return adp_output


class LoRA_MD_Adapter(nn.Module):
    def __init__(self, config, use_gate):
        super().__init__()
        self.num_adapter = config.num_adapter
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.adp_intermediate_size
        self.dense1 = nn.Linear(config.hidden_size, self.intermediate_size * self.num_adapter)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.ModuleList([nn.Linear(self.intermediate_size, config.hidden_size) for _ in range(self.num_adapter)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter_top_k = config.adapter_top_k
        self.scaling = 2
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = NaiveGate(config.hidden_size, config.num_adapter, config.adapter_top_k)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dense1.bias)
        for adp_idx in range(self.num_adapter):
            nn.init.zeros_(self.dense2[adp_idx].weight)
            nn.init.zeros_(self.dense2[adp_idx].bias)
        if self.use_gate:
            nn.init.kaiming_uniform_(self.gate.gate.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate.gate.bias)

    def forward(self, input_tensor: torch.Tensor, domain_idx:torch.Tensor=None) -> torch.Tensor:

        B, L = input_tensor.shape[:2]
        input_tensor = self.dropout(input_tensor)
        hidden_states = self.dense1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        # batch_size x seq_len x num_adapter x self_inter
        new_shape = hidden_states.size()[:-1] + (self.num_adapter, self.intermediate_size)
        # num_adapter x batch_size x seq_len x self_inter
        hidden_states = hidden_states.view(new_shape).transpose(2, 1).transpose(1, 0)
        adp_output = []
        for adp_idx in range(self.num_adapter):
            adp_output.append(self.dense2[adp_idx](hidden_states[adp_idx]))
        # batch_size x num_adapter x seq_len x hidden_size
        adp_output = torch.stack(adp_output, dim=0).transpose(1, 0)
        if self.use_gate:
            # (batch_size x seq_len) x num_adapter x hidden_size
            adp_output = adp_output.transpose(2, 1).view(-1, self.num_adapter, self.hidden_size)
            
            
            gate_top_k_idx, gate_top_k_score = self.gate(input_tensor)
            
            gate_top_k_score = gate_top_k_score.unsqueeze(-1)
            
            adp_output *= gate_top_k_score
            
            adp_output = adp_output.sum(dim=-2) 
           
            adp_output = adp_output.view(B, L, -1)
        else:
            assert domain_idx != None and domain_idx.shape[0] == B, f"domain_idx: {domain_idx.shape}"
            domain_idx = domain_idx.reshape(1, B)
            batch_idx = torch.arange(B, dtype = torch.long).to(adp_output.device)
            adp_output = adp_output[batch_idx, domain_idx].squeeze(0)

        adp_output *= self.scaling
        return adp_output


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class MOARobertaLayer(nn.Module):
    def __init__(self, config, use_gate, avg=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        self.adapter_type = config.adapter_type
        if self.adapter_type == "Naive":
            self.adapters = nn.ModuleList([Adapter(config) for _ in range(config.num_adapter)])
        elif self.adapter_type == "LoRA":
            if avg:
                self.adapters = LoRAAdapter(config, use_gate)
            else:
                self.adapters = LoRA_MD_Adapter(config, use_gate)
        
    
    def init_adapters_with_LM_pm(self):
        with torch.no_grad():
            inter_state_dict = self.intermediate.state_dict()
            output_state_dict = self.output.state_dict()
            adapter_state_dict = self.adapters[0].state_dict()
            adapter_state_dict["dense1.weight"] = inter_state_dict["dense.weight"]
            adapter_state_dict["dense1.bias"] = inter_state_dict["dense.bias"]
            adapter_state_dict["dense2.weight"] = output_state_dict["dense.weight"]
            adapter_state_dict["dense2.bias"] = output_state_dict["dense.bias"]
            adapter_state_dict["LayerNorm.weight"] = output_state_dict["LayerNorm.weight"]
            adapter_state_dict["LayerNorm.bias"] = output_state_dict["LayerNorm.bias"]
            for adapter in self.adapters:
                adapter.load_state_dict(adapter_state_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adp_idx: Optional[torch.Tensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, adp_idx
        )
        outputs = (layer_output,) + outputs


        return outputs

    def feed_forward_chunk(self, attention_output, adp_idx = None):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        adapter_output = []
        if self.adapter_type == "Naive":
            if adp_idx != None or self.use_gate:
                for adapter in self.adapters:
                    adapter_output.append(adapter(attention_output))
                adapter_output = torch.stack(adapter_output, dim = 0).transpose(0, 1)

                if not self.use_gate:
                    adapter_output = adapter_output[torch.arange(layer_output.shape[0], dtype = torch.long).to(layer_output.device), adp_idx]
                else:
                    gate_top_k_idx, gate_score = self.gate(attention_output)
                    adapter_output = adapter_output[gate_top_k_idx]
                    adapter_output *= gate_score
                    adapter_output = adapter_output.sum(dim = 1)
                assert layer_output.shape == adapter_output.shape, f"{layer_output.shape}/{adapter_output.shape}"
                layer_output += adapter_output
                layer_output /= 2
        elif self.adapter_type == "LoRA":
            adapter_output = self.adapters(attention_output, adp_idx)
            layer_output += adapter_output

        return layer_output


class MOARecRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_adapter_layer = config.num_adapter_layer
        assert self.num_adapter_layer == config.num_hidden_layers
        self.early_late_num_layer = int(self.num_adapter_layer / 2)
        self.layer = nn.ModuleList()
        self.use_gate = (config.mode == FINETUNE)
        print(f"If use gate for the late layers? {self.use_gate}")
        # Bottom layer for multi modal
        for _ in range(self.early_late_num_layer):
            self.layer.append(MOARobertaLayer(config, use_gate=True,avg=True))
        # Upper layer for multi domain

        for _ in range(self.early_late_num_layer):
            self.layer.append(MOARobertaLayer(config, use_gate=self.use_gate))
        self.gradient_checkpointing = False

    def init_adp_layers_with_LM_pm(self):
        for idx in range(self.num_adapter_layer):
            self.layer[-idx-1].init_adapters_with_LM_pm()
        return
    
    def desiable_exp(self):
        for idx in range(self.num_adapter_layer):
            self.layer[-idx-1].use_gate = False
        return

    def enable_exp(self):
        for idx in range(self.num_adapter_layer):
            self.layer[-idx-1].use_gate = True
        return

    def fix_gate(self):
        for idx in range(self.num_adapter_layer):
            for param in self.layer[-idx-1].adapters.gate.parameters():
                param.requires_grad = False
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        adp_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                if i < self.config.num_hidden_layers - self.config.num_adapter_layer:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        adp_idx,
                        attention_mask,
                        layer_head_mask
                    )

            else:
                if i < self.config.num_hidden_layers - self.config.num_adapter_layer:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        adp_idx,
                        attention_mask,
                        layer_head_mask,
                        past_key_value,
                        output_attentions,
                    )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
class MOARecRobertaModel(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = RecRobertaEmbeddings(config)
        self.encoder = MOARecRobertaEncoder(config)

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
        adp_idx: Optional[torch.Tensor] = None,
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

        # past_key_values_length
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
            adp_idx=adp_idx,
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

class MOARecRobertaForPretrain(RobertaPreTrainedModel):
    def __init__(self, config: RecRobertaConfig):
        super().__init__(config)

        self.roberta = MOARecRobertaModel(config)
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
        adp_idx: Optional[torch.Tensor] = None,
        attention_mask_s: Optional[torch.Tensor] = None,
        token_type_ids_s: Optional[torch.Tensor] = None,
        item_position_ids_s: Optional[torch.Tensor] = None,
        position_ids_s: Optional[torch.Tensor] = None,
        batch_item_seq_ids_s: Optional[bool] = None,
        input_ids_t: Optional[torch.Tensor] = None,
        attention_mask_t: Optional[torch.Tensor] = None,
        token_type_ids_t: Optional[torch.Tensor] = None,
        item_position_ids_t: Optional[torch.Tensor] = None,
        position_ids_t: Optional[torch.Tensor] = None,
        batch_item_seq_ids_t: Optional[bool] = None,
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
            adp_idx = adp_idx,
            attention_mask=attention_mask_s,
            token_type_ids=token_type_ids_s,
            position_ids=position_ids_s,
            item_position_ids=item_position_ids_s,
            batch_item_seq_ids = batch_item_seq_ids_s,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if return_correct:
            position_ids_t = create_position_ids_from_input_ids(input_ids_t, self.padding_idx, self.item_pos_offset)
        outputs_t = self.roberta(
            input_ids_t,
            adp_idx = adp_idx,
            attention_mask=attention_mask_t,
            token_type_ids=token_type_ids_t,
            position_ids=position_ids_t,
            item_position_ids=item_position_ids_t,
            batch_item_seq_ids = batch_item_seq_ids_t,
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
                adp_idx = adp_idx,
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
                adp_idx = adp_idx,
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
    
    def splited_VE_mlm_seq_item_contrast(self,       
        mlm_input_ids_s: Optional[torch.Tensor] = None, 
        mlm_labels_s: Optional[torch.Tensor] = None,
        attention_mask_s: Optional[torch.Tensor] = None,
        token_type_ids_s: Optional[torch.Tensor] = None,
        item_position_ids_s: Optional[torch.Tensor] = None,
        position_ids_s: Optional[torch.Tensor] = None,
        batch_item_seq_ids_s: Optional[bool] = None,
        input_ids_t: Optional[torch.Tensor] = None,
        attention_mask_t: Optional[torch.Tensor] = None,
        token_type_ids_t: Optional[torch.Tensor] = None,
        item_position_ids_t: Optional[torch.Tensor] = None,
        position_ids_t: Optional[torch.Tensor] = None,
        batch_item_seq_ids_t: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adp_idx: Optional[torch.Tensor] = None,
        return_correct = False,
    ):
        mid_layer_num = self.roberta.encoder.early_late_num_layer - 1
        batch_size = mlm_input_ids_s.size(0)

    
        mlm_labels_s = mlm_labels_s.view(-1, mlm_labels_s.size(-1))
        mlm_outputs_s = self.roberta(
            mlm_input_ids_s,
            adp_idx = adp_idx,
            attention_mask=attention_mask_s,
            token_type_ids=token_type_ids_s,
            position_ids=position_ids_s,
            item_position_ids=item_position_ids_s,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            batch_item_seq_ids = batch_item_seq_ids_s,
            return_dict=True,
        )
        if return_correct:
            position_ids_t = create_position_ids_from_input_ids(input_ids_t, self.padding_idx, self.item_pos_offset)
        outputs_t = self.roberta(
            input_ids_t,
            adp_idx = adp_idx,
            attention_mask=attention_mask_t,
            token_type_ids=token_type_ids_t,
            position_ids=position_ids_t,
            item_position_ids=item_position_ids_t,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            batch_item_seq_ids = batch_item_seq_ids_t,
            return_dict=True,
        )
        z1 = mlm_outputs_s.pooler_output  # (bs*num_sent, hidden_size)
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

        prediction_scores_s = self.lm_head(mlm_outputs_s.hidden_states[mid_layer_num])
        mlm_loss_s = self.loss_fct(prediction_scores_s.view(-1, self.config.vocab_size - 1), mlm_labels_s.view(-1))
        
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

    def vision_enhanced_mlm(self,       
        mlm_input_ids_s: Optional[torch.Tensor] = None, 
        mlm_labels_s: Optional[torch.Tensor] = None,
        attention_mask_s: Optional[torch.Tensor] = None,
        token_type_ids_s: Optional[torch.Tensor] = None,
        item_position_ids_s: Optional[torch.Tensor] = None,
        position_ids_s: Optional[torch.Tensor] = None,
        batch_item_seq_ids_s: Optional[bool] = None,
        input_ids_t: Optional[torch.Tensor] = None,
        attention_mask_t: Optional[torch.Tensor] = None,
        token_type_ids_t: Optional[torch.Tensor] = None,
        item_position_ids_t: Optional[torch.Tensor] = None,
        position_ids_t: Optional[torch.Tensor] = None,
        batch_item_seq_ids_t: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adp_idx: Optional[torch.Tensor] = None,
        return_correct = False,
    ):
        
        batch_size = mlm_input_ids_s.size(0)

    
        mlm_labels_s = mlm_labels_s.view(-1, mlm_labels_s.size(-1))
        mlm_outputs_s = self.roberta(
            mlm_input_ids_s,
            adp_idx = adp_idx,
            attention_mask=attention_mask_s,
            token_type_ids=token_type_ids_s,
            position_ids=position_ids_s,
            item_position_ids=item_position_ids_s,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            batch_item_seq_ids = batch_item_seq_ids_s,
            return_dict=True,
        )
        if return_correct:
            position_ids_t = create_position_ids_from_input_ids(input_ids_t, self.padding_idx, self.item_pos_offset)
        outputs_t = self.roberta(
            input_ids_t,
            adp_idx = adp_idx,
            attention_mask=attention_mask_t,
            token_type_ids=token_type_ids_t,
            position_ids=position_ids_t,
            item_position_ids=item_position_ids_t,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            batch_item_seq_ids = batch_item_seq_ids_t,
            return_dict=True,
        )
        z1 = mlm_outputs_s.pooler_output  # (bs*num_sent, hidden_size)
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

        prediction_scores_s = self.lm_head(mlm_outputs_s.last_hidden_state)
        mlm_loss_s = self.loss_fct(prediction_scores_s.view(-1, self.config.vocab_size - 1), mlm_labels_s.view(-1))
        
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
        item_pred_input, seq_cont_input = batch
        if item_pred_input != None:
            if "input_ids_s" in item_pred_input:
                output1 = self.item_seq_contrast(**item_pred_input, return_correct=True)
            else:
                output1 = self.splited_VE_mlm_seq_item_contrast(**item_pred_input, return_correct=True)

        else:
            output1 = None
        if seq_cont_input != None:
            output2 = self.item_seq_contrast(**seq_cont_input, return_correct=False)
            if output1 != None:
                output1.loss += output2.loss * self.seq_cont_weight
            else:
                output1 = output2

        return output1



class MOARecRobertaForSeqRec(RobertaPreTrainedModel):
    def __init__(self, config: RecRobertaConfig):
        super().__init__(config)

        self.roberta = MOARecRobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.sim = Similarity(config)
        self.use_mlm = config.use_mlm
        self.use_img = config.use_img
        self.mlm_weight = config.mlm_weight
        self.loss_fct = CrossEntropyLoss()
        self.max_item_embeddings = config.max_item_embeddings
        self.item_pos_offset = config.item_pos_offset
        self.item_embedding = None
        self.post_init()
        self.diff_pos = config.diff_pos

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

    def similarity_score(self, pooler_output, candidates = None, tokenizer: RecRobertaTokenizer=None, tokenized_items=None):
        if tokenizer != None:
            batch_candidates = candidates.tolist()
            batch_candidate_emb = []
            items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
            items = [ele[1] for ele in items]
            for candidate in batch_candidates:
                cand_items = [[items[item_idx]] for item_idx in candidate]
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
                adp_idx: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                item_position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                candidates: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, 
                batch_item_seq_ids: Optional[bool] = None,
                mlm_input_ids: Optional[bool] = None,
                mlm_labels: Optional[torch.Tensor] = None,
                tokenizer: RecRobertaTokenizer = None, 
                tokenized_items = None,
        ):

        mid_layer_num = self.roberta.encoder.early_late_num_layer - 1

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids != None: 
            batch_size = input_ids.size(0)
            outputs = self.roberta(
                input_ids,
                adp_idx = adp_idx,
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
        else: 
            assert self.use_mlm
            batch_size = mlm_input_ids.size(0)
            outputs = self.roberta(
                mlm_input_ids,
                adp_idx = adp_idx,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                item_position_ids=item_position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                batch_item_seq_ids=batch_item_seq_ids,
                use_img = self.use_img,
            )

        pooler_output = outputs.pooler_output # (bs, hidden_size)



        if labels is None:
            return self.similarity_score(pooler_output, candidates)


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
            loss = self.loss_fct(logits, labels)
        if mlm_labels != None:
            assert self.use_mlm
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(outputs.hidden_states[mid_layer_num])
            mlm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size - 1), mlm_labels.view(-1))
            loss += self.mlm_weight * mlm_loss
            
        return loss
