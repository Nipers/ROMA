import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from utils import read_json, FINETUNE
from faiss import IndexFlatIP, normalize_L2, StandardGpuResources, index_cpu_to_gpu_multiple_py
from model import (
    RecRobertaTokenizer,
    REFinetuneWrapper,
    MOARecRobertaForSeqRec
)
from config import (
    RecRobertaConfig,
)
from data import (
    MOARecRobertaTrainDataset,
    MOARecRobertaEvalDataset,
    MOARecRobertaFinetuneCollator,
    MOARecRobertaEvalCollator,
)

FINETUNE_CONFIG = {
    "MOARERecRoberta": (
        RecRobertaConfig, 
        MOARecRobertaForSeqRec, 
        RecRobertaTokenizer,
        MOARecRobertaTrainDataset, 
        MOARecRobertaEvalDataset, 
        MOARecRobertaFinetuneCollator, 
        MOARecRobertaEvalCollator,
        REFinetuneWrapper,
    )
}



def load_data(args):

    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


tokenizer_glb: RecRobertaTokenizer = None
item_index: IndexFlatIP = None
def _par_tokenize_doc(doc):    
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids

def test_stage(args, fine_tune_model, train_loader, dev_loader, test_loader, ckpt):
    logger = TensorBoardLogger(save_dir=args.output_dir, name=f"logs")
    version = logger.version
    checkpoint_callback = ModelCheckpoint(save_top_k=5, dirpath=os.path.join(args.output_dir, f"logs/version_{version}/ckpt"), monitor="val_ndcg_10", mode="max", filename="{epoch}-{step}-{val_ndcg_10:.6f}")        
    early_stop_callback = EarlyStopping(monitor="val_ndcg_10", min_delta=0, patience=10, verbose=False, mode="max")
    
    trainer = Trainer(accelerator="gpu",
                    logger=logger,
                    max_epochs=args.num_train_epochs,
                    devices=args.devices,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    default_root_dir=args.output_dir,
                    gradient_clip_val=1.0,
                    strategy='ddp_find_unused_parameters_true',
                    precision="bf16-mixed",
                    callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    fine_tune_model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"))["state_dict"])
    trainer.test(fine_tune_model, dataloaders=test_loader)


version = 0

def main(args):
    torch.set_float32_matmul_precision('high')
    
    ConfigModule, ModelModule, TokenizerModule, TrainDataset, \
    EvalDataset, TrainCollator, EvalCollator, Wrapper = FINETUNE_CONFIG[args.model_name]
    print(args)
    seed_everything(42)


    config = ConfigModule.from_pretrained(args.model_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.token_type_size = 8
    config.max_item_embeddings = 53
    config.max_token_num = 1024
    config.max_position_embeddings = 2050
    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    config.item_num = len(item2id)
    config.domain = args.domain
    config.pooler_type = args.pooler_type
    config.use_img = args.use_img
    if config.use_img:
        config.img_emb_path = args.img_emb_path
        config.img_marks_path = args.img_marks_path
    if args.use_moa:
        config.num_adapter = args.num_adapter
        config.num_adapter_layer = args.num_adapter_layer
        config.use_gate = args.use_gate
        config.adapter_type = args.adapter_type
        config.adapter_top_k = args.adapter_top_k
        config.adp_intermediate_size = args.adp_intermediate_size
    config.use_mlm = args.use_mlm
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = TokenizerModule.from_pretrained(args.model_path, config)
    config.diff_pos = args.diff_pos
    config.mode = FINETUNE
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / f'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    args.output_dir = os.path.join(args.output_dir, path_corpus.name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()
        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')


    model = ModelModule(config)
    
    
    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.roberta.embeddings.word_embeddings.parameters():
            param.requires_grad = False
    
    fine_tune_model = Wrapper(model, tokenizer_glb, tokenized_items, args.batch_size, args.learning_rate, args.warmup_steps, args.weight_decay, True, n_freeze_layer=args.n_freeze_layer)
    
    fine_tune_model.model.resize_token_embeddings(len(tokenizer))

    sub_dirs = ["logs", "cached_item_emb"]
    output_dir = args.output_dir
    for sub_dir in sub_dirs:
        output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Encoding items.')
    fine_tune_model.model.update_item_embedding(tokenizer_glb, tokenized_items, 32, cache_file=os.path.join(output_dir, f"item_emb_init.pt"))
    print(f'Encoding over.')
    # return
    finetune_data_collator = TrainCollator(tokenizer, tokenized_items, mlm_probability = args.mlm_probability, use_mlm = config.use_mlm, use_img = config.use_img)
    eval_data_collator = EvalCollator(tokenizer, tokenized_items, use_img = config.use_img)

    train_data = TrainDataset(train, collator=finetune_data_collator)
    val_data = EvalDataset(train, val, test, mode='val', collator=eval_data_collator, align=args.align_eval)
    test_data = EvalDataset(train, val, test, mode='test', collator=eval_data_collator, align=args.align_eval)

    
    train_loader = DataLoader(train_data, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            collate_fn=train_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
    dev_loader = DataLoader(val_data, 
                            batch_size=32, 
                            collate_fn=val_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
    test_loader = DataLoader(test_data, 
                            batch_size=32, 
                            collate_fn=test_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
    test_stage(args, fine_tune_model, train_loader, dev_loader, test_loader, args.best_ckpt)


    
               
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default='Scientific')
    parser.add_argument('--model_name', type=str, default='MOARERecRoberta')
    parser.add_argument('--model_path', type=str, default='nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large')
    parser.add_argument('--best_ckpt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')
    parser.add_argument('--pooler_type', type=str, default='cls')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=32, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
    # train
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_freeze_layer', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--devices', type=int, nargs="+", default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--align_eval', action='store_true')
    parser.add_argument('--diff_pos', action='store_true')
    parser.add_argument('--stage_1_check_interval', type=float, default=0.5)
    parser.add_argument('--stage_2_check_interval', type=float, default=0.25)
    parser.add_argument('--update_emb_interval', type=int, default=40)
    parser.add_argument('--mlm_probability', type=float, default=0.15)

    parser.add_argument('--use_moa', action='store_true')
    parser.add_argument('--use_gate', action='store_true')
    parser.add_argument('--num_adapter', type=int, default=0)
    parser.add_argument('--num_adapter_layer', type=int, default=0)
    parser.add_argument('--adapter_top_k', type=int, default=0)
    parser.add_argument('--adapter_type', type=str, default="Lora")
    parser.add_argument('--adp_intermediate_size', type=int, default=192)
    

    parser.add_argument('--use_img', action='store_true')
    parser.add_argument('--img_emb_path', type=str, default='./Scientific/img_emb.npy')
    parser.add_argument('--img_marks_path', type=str, default='./Scientific/has_img.npy')
    
    
    parser.add_argument('--use_mlm', action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    args.img_emb_path = os.path.join(args.data_path, args.img_emb_path)
    args.img_marks_path = os.path.join(args.data_path, args.img_marks_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
