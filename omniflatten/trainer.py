"""Token-normalized training, validation selection and resumable checkpoints."""
from pathlib import Path
from contextlib import nullcontext
from itertools import islice
import json
import math
import torch
from torch.nn import functional as F

class OmniFlattenTrainer:
    def __init__(self, model, tokenizer, accelerator, args):
        self.model, self.tokenizer, self.accelerator, self.args = model, tokenizer, accelerator, args
        self.step, self.epoch, self.next_batch, self.best = 0, 0, 0, math.inf
        self.metrics = {"train_steps":[], "validation":[]}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        totals = torch.zeros(2, device=self.accelerator.device, dtype=torch.float64)
        for batch in loader:
            labels = batch['labels'][:,1:].clone()
            labels[batch['attention_mask'][:,1:]==0] = -100
            if 'loss_mask' in batch:
                labels[batch['loss_mask'][:,1:]==0] = -100
            logits = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits[:,:-1].float()
            losses = F.cross_entropy(logits.transpose(1,2), labels, ignore_index=-100, reduction='none')
            stats = torch.stack([losses.sum(1), (labels!=-100).sum(1)],dim=1)
            totals += self.accelerator.gather_for_metrics(stats).double().sum(0)
        self.model.train()
        if totals[1]==0:
            raise ValueError('Validation set has no supervised tokens')
        return (totals[0]/totals[1]).item()

    def save_checkpoint(self, path, optimizer, scheduler):
        path = Path(path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(path/'training_state'))
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save_pretrained(path, self.tokenizer)
            metadata = {'stage':self.args.stage, 'step':self.step, 'epoch':self.epoch,
                        'next_batch':self.next_batch, 'best_val_loss':self.best,
                        'epochs':self.args.epochs, 'batch_size':self.args.batch_size,
                        'gradient_accumulation_steps':self.args.gradient_accumulation_steps,
                        'world_size':self.accelerator.num_processes,
                        'seed':self.args.seed, 'lr':self.args.lr,
                        'warmup_ratio':self.args.warmup_ratio,
                        'weight_decay':self.args.weight_decay,
                        'model_dtype':self.args.model_dtype,
                        'max_seq_len':self.args.max_seq_len,
                        'gradient_checkpointing':self.args.gradient_checkpointing}
            (path/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path):
        metadata = json.loads((Path(path)/'training_metadata.json').read_text())
        for key in ('stage','epochs','batch_size','gradient_accumulation_steps','seed','lr','warmup_ratio','weight_decay','model_dtype','max_seq_len','gradient_checkpointing'):
            if metadata[key] != getattr(self.args,key):
                raise ValueError(f'Resume requires the same {key}; use --checkpoint for a new stage')
        if metadata['world_size'] != self.accelerator.num_processes:
            raise ValueError('Resume requires the same process count')
        self.accelerator.load_state(str(Path(path)/'training_state'))
        self.step, self.epoch, self.next_batch, self.best = [metadata[k] for k in ('step','epoch','next_batch','best_val_loss')]

    def train(self, train_loader, val_loader, optimizer, scheduler, output_dir):
        acc = self.accelerator
        accumulation = self.args.gradient_accumulation_steps
        if self.args.eval_before_training:
            initial = self.evaluate(val_loader)
            self.metrics['initial_validation_loss'] = initial
            acc.print(f'initial_validation_loss={initial:.6f}')
        self.model.train()
        for epoch in range(self.epoch,self.args.epochs):
            self.epoch = epoch
            if hasattr(train_loader,'set_epoch'):
                train_loader.set_epoch(epoch)
            loader = acc.skip_first_batches(train_loader,self.next_batch) if self.next_batch else train_loader
            iterator = iter(loader)
            while True:
                batches = list(islice(iterator,accumulation))
                if not batches:
                    break
                counts=[]
                for batch in batches:
                    valid = (batch['labels'][:,1:]!=-100) & batch['attention_mask'][:,1:].bool()
                    if 'loss_mask' in batch:
                        valid &= batch['loss_mask'][:,1:].bool()
                    counts.append(valid.sum())
                total = acc.reduce(torch.stack(counts).sum(), reduction='sum')
                if total==0:
                    raise ValueError('Training window contains no supervised tokens')
                optimizer.zero_grad(set_to_none=True)
                running = torch.zeros((),device=acc.device)
                for i,(batch,count) in enumerate(zip(batches,counts)):
                    sync = nullcontext() if i==len(batches)-1 else acc.no_sync(self.model)
                    with sync:
                        loss = self.model(**batch).loss
                        # DDP averages gradients across ranks; normalize across
                        # all supervised tokens in this entire accumulation window.
                        acc.backward(loss*count*acc.num_processes/total)
                        running += loss.detach()*count
                norm = acc.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite gradient norm')
                optimizer.step()
                if not acc.optimizer_step_was_skipped:
                    scheduler.step()
                self.step += 1
                self.next_batch += len(batches)
                value = (acc.reduce(running,reduction='sum')/total).item()
                self.metrics['train_steps'].append({'step':self.step,'loss':value,
                    'supervised_tokens':int(total.item()),'gradient_norm':float(norm.item()),
                    'optimizer_step_skipped':bool(acc.optimizer_step_was_skipped)})
                if self.step%self.args.log_interval==0:
                    acc.print(f'epoch={epoch+1} step={self.step} loss={value:.5f}')
                if self.args.save_interval and self.step%self.args.save_interval==0:
                    self.save_checkpoint(Path(output_dir)/f'step-{self.step}',optimizer,scheduler)
            val_loss = self.evaluate(val_loader)
            acc.print(f'epoch={epoch+1} validation_loss={val_loss:.6f}')
            self.metrics['validation'].append({'epoch':epoch+1,'loss':val_loss})
            self.epoch, self.next_batch = epoch+1,0
            if val_loss < self.best:
                self.best = val_loss
                self.save_checkpoint(Path(output_dir)/'best_model',optimizer,scheduler)
            self.save_checkpoint(Path(output_dir)/'last_model',optimizer,scheduler)

        if torch.cuda.is_available():
            peaks = torch.tensor([torch.cuda.max_memory_allocated(acc.device)],device=acc.device)
            self.metrics['peak_allocated_bytes_by_rank'] = acc.gather(peaks).cpu().tolist()
        if acc.is_main_process:
            Path(output_dir).mkdir(parents=True,exist_ok=True)
            (Path(output_dir)/'metrics.json').write_text(json.dumps(self.metrics,indent=2))
