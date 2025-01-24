from transformers.models.roberta.modeling_roberta import RobertaConfig


class RecRobertaConfig(RobertaConfig):

    def __init__(self, 
                sep_token_id: int = 2,
                token_type_size: int = 4, # <s>, key, value, <pad>
                max_token_num: int = 2048,
                max_item_embeddings: int = 32, # 1 for <s>, 50 for items
                max_attr_num: int = 12,
                max_attr_length: int = 8,
                pooler_type: str = 'cls',
                temp: float = 0.05,
                mlm_weight: float = 0.1,
                seq_cont_weight: float = 0.1,
                item_num: int = 0,
                finetune_negative_sample_size: int = 0,
                item_pos_offset: int = 0,
                num_adapter_layer: int = 0,
                num_adapter: int = 0,
                use_gate: bool = False,
                adp_intermediate_size=192,
                **kwargs):
        super().__init__(**kwargs)

        self.sep_token_id = sep_token_id
        self.token_type_size = token_type_size
        self.max_token_num = max_token_num
        self.item_pos_offset = item_pos_offset
        self.max_item_embeddings = max_item_embeddings
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length
        self.pooler_type = pooler_type
        self.temp = temp
        self.mlm_weight = mlm_weight
        self.seq_cont_weight = seq_cont_weight

        # finetune config
        self.adp_intermediate_size = adp_intermediate_size
        self.item_num = item_num
        self.finetune_negative_sample_size = finetune_negative_sample_size
        self.num_adapter_layer = num_adapter_layer
        self.num_adapter = num_adapter
        self.use_gate = use_gate
