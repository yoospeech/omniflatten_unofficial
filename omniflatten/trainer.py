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

    @staticmethod
    def _valid_tokens(batch):
        valid = (batch['labels'][:, 1:] != -100) & batch['attention_mask'][:, 1:].bool()
        if 'loss_mask' in batch:
            valid &= batch['loss_mask'][:, 1:].bool()
        return valid

    def _loss_weights(self, batch, valid, global_task_counts):
        """Give ASR and TTS equal total weight, regardless of token count."""
        if 'task_id' not in batch:
            return None, valid.sum()
        factors = torch.zeros(2, device=valid.device, dtype=torch.float32)
        present = global_task_counts > 0
        factors[present] = 1.0 / global_task_counts[present].to(torch.float32)
        per_row = factors[batch['task_id']]
        weights = torch.zeros_like(batch['labels'], dtype=torch.float32)
        weights[:, 1:] = per_row[:, None]
        return weights, (valid * per_row[:, None]).sum()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        multitask = None
        totals = torch.zeros(4, device=self.accelerator.device, dtype=torch.float64)
        for batch in loader:
            # Evaluate examples separately so gather_for_metrics can remove
            # distributed sampler padding while preserving token weighting.
            stats = []
            for i in range(batch['input_ids'].shape[0]):
                item = {key: value[i:i+1] for key, value in batch.items() if key != 'task_id'}
                valid = self._valid_tokens(item)
                count = valid.sum()
                loss = self.model(**item, use_cache=False).loss
                task = int(batch['task_id'][i]) if 'task_id' in batch else 0
                row = torch.zeros(4, device=count.device, dtype=torch.float64)
                row[2*task:2*task+2] = torch.stack((loss * count, count)).double()
                stats.append(row)
            stats = torch.stack(stats)
            totals += self.accelerator.gather_for_metrics(stats).double().sum(0)
        self.model.train()
        if totals[1] == 0 and totals[3] == 0:
            raise ValueError('Validation set has no supervised tokens')
        values = [totals[0]/totals[1] if totals[1] else None,
                  totals[2]/totals[3] if totals[3] else None]
        if 'task_id' not in batch or values[0] is None or values[1] is None:
            return next(value for value in values if value is not None).item()
        active = [value for value in values if value is not None]
        return {'loss': torch.stack(active).mean().item(),
                'asr_loss': None if values[0] is None else values[0].item(),
                'tts_loss': None if values[1] is None else values[1].item(),
                'asr_tokens': int(totals[1].item()), 'tts_tokens': int(totals[3].item())}

    def save_checkpoint(self, path, optimizer, scheduler):
        path = Path(path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(path/'training_state'))
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save_pretrained(path, self.tokenizer)
            reduction = 'balanced_tasks' if (self.args.stage == '1' and getattr(self.args, 'stage1_task', 'both') == 'both') else 'token'
            metadata = {'stage':self.args.stage, 'stage1_task':getattr(self.args, 'stage1_task', 'both'),
                        'stage1_loss_reduction':reduction, 'step':self.step, 'epoch':self.epoch,
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
        if metadata.get('stage1_task', 'both') != getattr(self.args, 'stage1_task', 'both'):
            raise ValueError('Resume requires the same stage1_task')
        expected_reduction = 'balanced_tasks' if (self.args.stage == '1' and getattr(self.args, 'stage1_task', 'both') == 'both') else 'token'
        if metadata.get('stage1_loss_reduction', 'token') != expected_reduction:
            raise ValueError('Resume requires the same stage1 loss reduction; use --checkpoint to restart optimizer state')
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
            initial_loss = initial['loss'] if isinstance(initial, dict) else initial
            self.metrics['initial_validation_loss'] = initial_loss
            acc.print(f'initial_validation_loss={initial_loss:.6f}')
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
                task_counts = torch.zeros(2, device=acc.device, dtype=torch.long)
                for batch in batches:
                    valid = self._valid_tokens(batch)
                    counts.append(valid.sum())
                    if 'task_id' in batch:
                        for task in range(2):
                            task_counts[task] += valid[batch['task_id'] == task].sum()
                total = acc.reduce(torch.stack(counts).sum(), reduction='sum')
                global_task_counts = acc.reduce(task_counts, reduction='sum')
                if total==0:
                    raise ValueError('Training window contains no supervised tokens')
                optimizer.zero_grad(set_to_none=True)
                running = torch.zeros((),device=acc.device)
                weighted_total = (global_task_counts > 0).sum().to(torch.float32) if task_counts.any() else total
                for i,(batch,count) in enumerate(zip(batches,counts)):
                    sync = nullcontext() if i==len(batches)-1 else acc.no_sync(self.model)
                    with sync:
                        valid = self._valid_tokens(batch)
                        weights, local_total = self._loss_weights(batch, valid, global_task_counts)
                        model_batch = {key: value for key, value in batch.items() if key != 'task_id'}
                        if weights is not None:
                            model_batch['loss_weight'] = weights
                        loss = self.model(**model_batch).loss
                        # DDP averages gradients across ranks; normalize across
                        # all supervised tokens in this entire accumulation window.
                        acc.backward(loss*local_total*acc.num_processes/weighted_total)
                        running += loss.detach()*local_total
                norm = acc.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite gradient norm')
                optimizer.step()
                if not acc.optimizer_step_was_skipped:
                    scheduler.step()
                self.step += 1
                self.next_batch += len(batches)
                value = (acc.reduce(running,reduction='sum')/weighted_total).item()
                self.metrics['train_steps'].append({'step':self.step,'loss':value,
                    'supervised_tokens':int(total.item()),'gradient_norm':float(norm.item()),
                    'optimizer_step_skipped':bool(acc.optimizer_step_was_skipped)})
                if self.step%self.args.log_interval==0:
                    acc.print(f'epoch={epoch+1} step={self.step} loss={value:.5f}')
                if self.args.save_interval and self.step%self.args.save_interval==0:
                    self.save_checkpoint(Path(output_dir)/f'step-{self.step}',optimizer,scheduler)
            validation = self.evaluate(val_loader)
            val_loss = validation['loss'] if isinstance(validation, dict) else validation
            if isinstance(validation, dict):
                acc.print(f"epoch={epoch+1} validation_loss={val_loss:.6f} "
                          f"asr_loss={validation['asr_loss']:.6f} tts_loss={validation['tts_loss']:.6f}")
                self.metrics['validation'].append({'epoch':epoch+1, **validation})
            else:
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
