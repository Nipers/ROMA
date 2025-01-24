import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils import PretrainEvalMetric, GlobalMetric, LocalMetric

class PretrainWrapper(pl.LightningModule):
    def __init__(self, 
                model: nn.Module,
                learning_rate: float = 5e-5,
                warmup_steps: int = 0,
                weight_decay: float = 0.0
                ):
        super().__init__()
        
        self.hparams.learning_rate = learning_rate
        self.hparams.warmup_steps = warmup_steps
        self.hparams.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.metric = PretrainEvalMetric()        

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = outputs.loss
        correct_num = outputs.cl_correct_num
        total_num = outputs.cl_total_num
        self.metric.update(correct_num, total_num, loss)
    
    def on_validation_epoch_end(self):
        acc, hit, total, loss = self.metric.compute()
        self.log(f"val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_hit", hit, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_total", total, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.metric.reset()
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        print(batch[0])
        assert 1 == 2
        outputs = self.model(batch)
        loss = outputs.loss
        correct_num = outputs.cl_correct_num
        total_num = outputs.cl_total_num
        self.metric.update(correct_num, total_num, loss)
    
    def on_test_epoch_end(self):
        acc, hit, total, loss = self.metric.compute()
        self.log(f"val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_hit", hit, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_total", total, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.metric.reset()
        return super().on_test_epoch_end()


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]



class FinetuneWrapper(pl.LightningModule):
    def __init__(self, 
            model: nn.Module,
            tokenizer,
            tokenized_items,
            batch_size,
            learning_rate: float = 5e-5,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            update_item_emb: bool = True,
            metric_ks: list = [5, 10, 20, 50],
            n_freeze_layer: int = 0,
        ):
        super().__init__()
        self.loss_fct = CrossEntropyLoss()
        self.hparams.learning_rate = learning_rate
        self.hparams.warmup_steps = warmup_steps
        self.hparams.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.metric_ks = metric_ks
        self.local_metric_0 = LocalMetric(metric_ks[0])
        self.local_metric_1 = LocalMetric(metric_ks[1])
        self.local_metric_2 = LocalMetric(metric_ks[2])
        self.local_metric_3 = LocalMetric(metric_ks[3])
        self.global_metric = GlobalMetric()
        self.update_item_emb = update_item_emb
        self.tokenizer = tokenizer
        self.tokenized_items = tokenized_items
        self.batch_size = batch_size
        self.n_freeze_layer = n_freeze_layer
        
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs

    def validation_step(self, batch, batch_idx):
        batch, labels = batch
        with torch.no_grad():
            scores = self.model(**batch)
        loss = self.loss_fct(scores, labels).item()
        self.local_metric_0.update(scores, labels)
        self.local_metric_1.update(scores, labels)
        self.local_metric_2.update(scores, labels)
        self.local_metric_3.update(scores, labels)

        self.global_metric.update(scores, labels, loss)

    def test_step(self, batch, batch_idx):
        batch, labels = batch
        with torch.no_grad():
            scores = self.model(**batch)
        loss = self.loss_fct(scores, labels).item()
        self.local_metric_0.update(scores, labels)
        self.local_metric_1.update(scores, labels)
        self.local_metric_2.update(scores, labels)
        self.local_metric_3.update(scores, labels)

        self.global_metric.update(scores, labels, loss)
        
    def on_train_epoch_start(self) -> None:
        if self.update_item_emb:
            print(f'Encoding items.')
            self.model.update_item_embedding(self.tokenizer, self.tokenized_items, self.batch_size)
            print(f'Encoding over.')
        else:
            print(f'Fix item embeddings.')

        return super().on_train_epoch_start()
    
    def on_validation_epoch_end(self):
        self.log_metric("val")
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self) -> None:
        self.log_metric("test")
        return super().on_test_epoch_end()

    def log_metric(self, mode):
        log_dict = {}
        ndcg, hit = self.local_metric_0.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[0]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[0]}"] = hit
        ndcg, hit = self.local_metric_1.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[1]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[1]}"] = hit
        ndcg, hit = self.local_metric_2.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[2]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[2]}"] = hit
        ndcg, hit = self.local_metric_3.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[3]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[3]}"] = hit
        
        self.local_metric_0.reset()
        self.local_metric_1.reset()
        self.local_metric_2.reset()
        self.local_metric_3.reset()
        mrr, auc, loss = self.global_metric.compute()
        self.global_metric.reset()
        log_dict[f"{mode}_mrr"] = mrr
        log_dict[f"{mode}_auc"] = auc
        log_dict[f"{mode}_loss"] = loss
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
    def configure_optimizers(self):
        model = self.model
        no_tune = [f"encoder.layer.{i}" for i in range(self.n_freeze_layer)]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and not any(nt in n for nt in no_tune))],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and not any(nt in n for nt in no_tune))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    

class REFinetuneWrapper(pl.LightningModule):
    def __init__(self, 
            model: nn.Module,
            tokenizer,
            tokenized_items,
            batch_size,
            learning_rate: float = 5e-5,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            update_item_emb: bool = True,
            metric_ks: list = [5, 10, 20, 50],
            n_freeze_layer: int = 0,
            update_emb_interval: int = 40,
        ):
        super().__init__()
        self.loss_fct = CrossEntropyLoss()
        self.hparams.learning_rate = learning_rate
        self.hparams.warmup_steps = warmup_steps
        self.hparams.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.metric_ks = metric_ks
        self.local_metric_0 = LocalMetric(metric_ks[0])
        self.local_metric_1 = LocalMetric(metric_ks[1])
        self.local_metric_2 = LocalMetric(metric_ks[2])
        self.local_metric_3 = LocalMetric(metric_ks[3])
        self.global_metric = GlobalMetric()
        self.update_item_emb = update_item_emb
        self.tokenizer = tokenizer
        self.tokenized_items = tokenized_items
        self.batch_size = batch_size
        self.n_freeze_layer = n_freeze_layer
        self.update_emb_interval = update_emb_interval

    def training_step(self, batch, batch_idx):
        
        if self.update_item_emb:
            # stage_1, retrieve similar embedding
            outputs = self.model(**batch, tokenizer=self.tokenizer, tokenized_items = self.tokenized_items)
        else:
            # stage_2, fix_item_embedding, no tokenizer
            outputs = self.model(**batch)
        return outputs

    def validation_step(self, batch, batch_idx):
        batch, labels = batch
        with torch.no_grad():
            scores = self.model(**batch)
            
        loss = self.loss_fct(scores, labels).item()
        self.local_metric_0.update(scores, labels)
        self.local_metric_1.update(scores, labels)
        self.local_metric_2.update(scores, labels)
        self.local_metric_3.update(scores, labels)

        self.global_metric.update(scores, labels, loss)

    def test_step(self, batch, batch_idx):
        batch, labels = batch
        with torch.no_grad():
            scores = self.model(**batch)
        loss = self.loss_fct(scores, labels).item()
        self.local_metric_0.update(scores, labels)
        self.local_metric_1.update(scores, labels)
        self.local_metric_2.update(scores, labels)
        self.local_metric_3.update(scores, labels)

        self.global_metric.update(scores, labels, loss)
    
    def on_validation_epoch_end(self):
        self.log_metric("val")
        return super().on_validation_epoch_end()
    
    def on_validation_epoch_start(self):
        if self.update_item_emb:
            print(f'Encoding items before Valid Epoch.')
            self.model.update_item_embedding(self.tokenizer, self.tokenized_items, 32)
            print(f'Encoding over.')
        return super().on_validation_epoch_start()
  

    def on_test_epoch_end(self) -> None:
        self.log_metric("test")
        return super().on_test_epoch_end()

    def log_metric(self, mode):
        log_dict = {}
        ndcg, hit = self.local_metric_0.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[0]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[0]}"] = hit
        ndcg, hit = self.local_metric_1.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[1]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[1]}"] = hit
        ndcg, hit = self.local_metric_2.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[2]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[2]}"] = hit
        ndcg, hit = self.local_metric_3.compute()
        log_dict[f"{mode}_ndcg_{self.metric_ks[3]}"] = ndcg
        log_dict[f"{mode}_hit_{self.metric_ks[3]}"] = hit


        self.local_metric_0.reset()
        self.local_metric_1.reset()
        self.local_metric_2.reset()
        self.local_metric_3.reset()
        mrr, auc, loss = self.global_metric.compute()
        self.global_metric.reset()
        log_dict[f"{mode}_mrr"] = mrr
        log_dict[f"{mode}_auc"] = auc
        log_dict[f"{mode}_loss"] = loss
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
    def configure_optimizers(self):
        model = self.model
        no_tune = [f"encoder.layer.{i}" for i in range(self.n_freeze_layer)]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and not any(nt in n for nt in no_tune))],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and not any(nt in n for nt in no_tune))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


