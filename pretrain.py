from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from model import ( 
    RecRobertaTokenizer, 
    RecRobertaConfig, 
    PretrainWrapper,)
from data import (
    VEMLMPretrainCollator,
    MOARecRobertaPretrainDataset
)
from utils import read_json

PRETRAIN_CONFIG = {
    "MOARecRoberta": (
        RecRobertaConfig, 
        RecRobertaTokenizer, 
        VEMLMPretrainCollator,
        MOARecRobertaPretrainDataset
    )
}



tokenizer_glb = None
def _par_tokenize_doc(doc):    
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids

def load_data(args):
    train_path = os.path.join(args.data_path, args.train_file)
    dev_path = os.path.join(args.data_path, args.dev_file)
    meta_path = os.path.join(args.data_path, args.item_attr_file)
    smap_path = os.path.join(args.data_path, args.item2id_file)

    train_data = json.load(open(train_path, "r"))
    val_data = json.load(open(dev_path, "r"))
    item_meta_dict = json.load(open(meta_path, "r"))
    item2id = json.load(open(smap_path,"r"))


    if args.use_moa and not args.use_gate:
        domain_idx_file = os.path.join(args.data_path, args.domain_idx_file)
        domain_idx = json.load(open(domain_idx_file, "r"))
        return train_data, val_data, item_meta_dict, item2id, domain_idx

    return train_data, val_data, item_meta_dict, item2id

def main(args):
    torch.set_float32_matmul_precision('high')

    print(args)
    seed_everything(42)
    logger = TensorBoardLogger(save_dir=args.output_dir, name=f"logs")
    version = logger.version

    ConfigModule, ModelModule, TokenizerModule, CollatorModule, DatasetModule = PRETRAIN_CONFIG[args.model_name]
    config = ConfigModule.from_pretrained(args.model_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 53
    config.token_type_size = 8 
    config.use_img = args.use_img
    if config.use_img:
        config.img_emb_path = args.img_emb_path
        config.img_marks_path = args.img_marks_path
    if args.model_name == "Recformer":
        config.attention_window = [64] * 12
    if args.use_moa:
        config.num_adapter = args.num_adapter
        config.num_adapter_layer = args.num_adapter_layer
        config.use_gate = args.use_gate
        config.adapter_type = args.adapter_type
        config.adapter_top_k = args.adapter_top_k
        config.adp_intermediate_size = args.adp_intermediate_size
    config.max_token_num = 1024
    config.max_position_embeddings = 1100
    config.mlm_weight = args.mlm_weight
    config.pooler_type = "avg"
    tokenizer = TokenizerModule.from_pretrained(args.model_path, config)

    global tokenizer_glb
    tokenizer_glb = tokenizer

    # preprocess corpus
    path_corpus = Path(os.path.join(args.data_path, args.item_attr_file))
    dir_corpus = path_corpus.parent
    dir_preprocess = dir_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_tokenized_items = dir_preprocess / f'MOARecRoberta_tokenized_items.json'
    if args.use_moa and not args.use_gate:
        train_data, dev_data, item_meta_dict, item2id, domain_idx = load_data(args)
    else:
        train_data, dev_data, item_meta_dict, item2id = load_data(args)

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        if args.use_moa:
            tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        else:
            tokenized_items = {item_id: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        json.dump(tokenized_items, open(path_tokenized_items, 'w'))
    if args.use_moa:
        tokenized_items = read_json(path_tokenized_items, True)
    else:
        tokenized_items = json.load(open(path_tokenized_items))
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    train_collator = CollatorModule(tokenizer, tokenized_items, mlm_probability=args.mlm_probability, mode = "train", use_img=args.use_img)
    dev_collator = CollatorModule(tokenizer, tokenized_items, mlm_probability=args.mlm_probability, mode = "dev", use_img=args.use_img)
    if args.use_moa:
        train_set = DatasetModule(train_data, domain_idx, train_collator)
        dev_set = DatasetModule(train_data, domain_idx, dev_collator, dev_data)
    else:
        train_set = DatasetModule(train_data, train_collator)
        dev_set = DatasetModule(dev_data, dev_collator)

    train_loader = DataLoader(train_set, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            collate_fn=train_set.collate_fn,
                            num_workers=args.dataloader_num_workers)
    dev_loader = DataLoader(dev_set, 
                            batch_size=args.batch_size, 
                            collate_fn=dev_set.collate_fn,
                            num_workers=args.dataloader_num_workers)
    
    pytorch_model = ModelModule(config)
    state_dict = torch.load(args.init_ckpt, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    with torch.no_grad():
        token_type_embeddings = state_dict.pop("roberta.embeddings.token_type_embeddings.weight")
        pytorch_model.roberta.embeddings.token_type_embeddings.weight[0:3] = token_type_embeddings[0:3]
        pytorch_model.roberta.embeddings.token_type_embeddings.weight[3:5] = token_type_embeddings[1:3]
        pytorch_model.roberta.embeddings.token_type_embeddings.weight[5:7] = token_type_embeddings[1:3]

        item_position_embeddings = state_dict.pop("roberta.embeddings.item_position_embeddings.weight")

    pytorch_model.load_state_dict(state_dict, strict = False)
    pytorch_model.resize_token_embeddings(len(tokenizer))
    if args.model_name == "MOARecRoberta":
        if args.adapter_type == "Naive":
            pytorch_model.roberta.encoder.init_adp_layers_with_LM_pm()

    fix_ls = ["roberta.encoder.layer.{}.attention", "roberta.encoder.layer.{}.intermediate", "roberta.encoder.layer.{}.output"]
    for i in range(args.n_freeze_layer):
        for name, param in pytorch_model.named_parameters():
            for fix_para in fix_ls:
                if fix_para.format(i) in name:
                    param.requires_grad = False
    for name, param in pytorch_model.named_parameters():
        if args.fix_word_embedding:
            if "embeddings.word_embeddings" in name:
                param.requires_grad = False
            

    model = PretrainWrapper(pytorch_model, learning_rate=args.learning_rate)
    if args.pretrained_ckpt != None:
        print(f"Resume from ckpt {args.pretrained_ckpt}")
        state_dict = torch.load(args.pretrained_ckpt, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=3, dirpath=os.path.join(args.output_dir, f"logs/version_{version}/ckpt"), monitor="val_acc", mode="max", filename="{epoch}-{step}-{val_acc:.4f}")
    
    trainer = Trainer(accelerator="gpu",
                    logger=logger,
                    max_epochs=args.num_train_epochs,
                    devices=args.devices,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    check_val_every_n_epoch=args.check_val_every_n_epoch,
                    default_root_dir=args.output_dir,
                    gradient_clip_val=1.0,
                    log_every_n_steps=args.log_step,
                    strategy='ddp_find_unused_parameters_true',
                    precision="bf16-mixed",
                    callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Recformer")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--item_attr_file', type=str, required=True)
    parser.add_argument('--domain_idx_file', type=str)
    parser.add_argument('--item2id_file', type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', type=int, default=2)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--mlm_weight', type=float, default=0.2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_step', type=int, default=5000)
    parser.add_argument('--adapter_top_k', type=int, default=2)
    parser.add_argument('--adapter_type', type=str, default="Naive", help="Choose from [`Naive`, `LoRA`]")
    parser.add_argument('--devices', default=[0], type=int, nargs="+")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--init_ckpt', type=str, default='longformer_ckpt/longformer-base-4096.bin')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--n_freeze_layer', type=int, default=0)

    parser.add_argument('--use_moa', action='store_true')
    parser.add_argument('--use_gate', action='store_true')
    parser.add_argument('--num_adapter', type=int, default=0)
    parser.add_argument('--num_adapter_layer', type=int, default=0)
    parser.add_argument('--adp_intermediate_size', type=int, default=192)

    parser.add_argument('--use_img', action='store_true')
    parser.add_argument('--img_emb_path', type=str, default='dataset/recroberta/pretrain_data/img_emb_beit3.npy')
    parser.add_argument('--img_marks_path', type=str, default='dataset/recroberta/pretrain_data/has_img_beit3.npy')

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)