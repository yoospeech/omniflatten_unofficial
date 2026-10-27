#!/usr/bin/env python3
"""Train stages 1 -> 2 -> 3a -> 3b using a single model/tokenizer contract."""
import argparse
from pathlib import Path
import math
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from omniflatten.model import OmniFlattenModel, OmniFlattenConfig
from omniflatten.trainer import OmniFlattenTrainer
from omniflatten.data.modality_alignment_dataset import ModalityAlignmentDataset
from omniflatten.data.dialogue_dataset import HalfDuplexDialogueDataset, FullDuplexThreeStreamDataset, FullDuplexTwoStreamDataset
from omniflatten.data.formatting import collate

STAGE_DEFAULTS = {'1':1024,'2':8192,'3a':8192,'3b':8192}

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=list(STAGE_DEFAULTS))
    parser.add_argument('--data_dir',required=True,help='Directory containing train.jsonl and val.jsonl')
    parser.add_argument('--backbone',default='Qwen/Qwen2-0.5B')
    parser.add_argument('--checkpoint',help='Model/tokenizer checkpoint from preceding stage')
    parser.add_argument('--resume_from_checkpoint',help='Full training checkpoint from this stage')
    parser.add_argument('--output_dir',default='outputs')
    parser.add_argument('--codec_dir',help='Local CosyVoice v1 checkpoint (omit with precomputed speech_tokens)')
    parser.add_argument('--max_seq_len',type=int)
    parser.add_argument('--gradient_checkpointing',action='store_true')
    parser.add_argument('--eval_before_training',action='store_true')
    parser.add_argument('--model_dtype',choices=['float32','bfloat16'],default='float32',
                        help='bfloat16 reduces parameter/gradient/Adam state memory; no FP32 master weights')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--eval_batch_size',type=int,default=1)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--weight_decay',type=float,default=0.1)
    parser.add_argument('--warmup_ratio',type=float,default=0.05)
    parser.add_argument('--max_grad_norm',type=float,default=1.0)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--log_interval',type=int,default=10)
    parser.add_argument('--save_interval',type=int,default=0,help='0 saves only at epoch boundaries')
    precision=parser.add_mutually_exclusive_group()
    precision.add_argument('--bf16',action='store_true')
    precision.add_argument('--fp16',action='store_true')
    args=parser.parse_args()
    if args.checkpoint and args.resume_from_checkpoint:
        parser.error('Choose --checkpoint or --resume_from_checkpoint')
    if args.stage!='1' and not (args.checkpoint or args.resume_from_checkpoint):
        parser.error('Dialogue stages require a checkpoint from the curriculum')
    if args.codec_dir and args.num_workers:
        parser.error('Online codec extraction requires --num_workers 0; precompute for parallel loading')
    if min(args.batch_size,args.eval_batch_size,args.gradient_accumulation_steps,args.epochs,args.log_interval)<=0:
        parser.error('Batch sizes, accumulation, epochs and log_interval must be positive')
    if not 0<=args.warmup_ratio<1 or args.lr<=0 or args.save_interval<0:
        parser.error('Invalid learning-rate schedule or save interval')
    args.max_seq_len = args.max_seq_len or STAGE_DEFAULTS[args.stage]
    return args


def build_model(args):
    path = args.resume_from_checkpoint or args.checkpoint
    if path:
        if args.checkpoint:
            meta_path=Path(path)/'training_metadata.json'
            if not meta_path.exists():
                raise ValueError('Previous-stage checkpoint lacks training_metadata.json')
            previous={'2':'1','3a':'2','3b':'3a'}.get(args.stage)
            if previous is not None and json.loads(meta_path.read_text())['stage']!=previous:
                raise ValueError(f'Stage {args.stage} requires a stage {previous} checkpoint')
        return OmniFlattenModel.from_pretrained(path)
    model=OmniFlattenModel(OmniFlattenConfig(args.backbone))
    tokenizer=AutoTokenizer.from_pretrained(args.backbone)
    model.prepare_tokenizer(tokenizer)
    return model,tokenizer


def build_datasets(args, tokenizer):
    codec=None
    if args.codec_dir:
        from omniflatten.speech_tokenizer import CosyVoiceSpeechTokenizer
        codec=CosyVoiceSpeechTokenizer(args.codec_dir)
    cls={'1':ModalityAlignmentDataset,'2':HalfDuplexDialogueDataset,
         '3a':FullDuplexThreeStreamDataset,'3b':FullDuplexTwoStreamDataset}[args.stage]
    datasets=[cls(args.data_dir,tokenizer,codec,max_seq_len=args.max_seq_len,
                  manifest_filename=f'{split}.jsonl') for split in ('train','val')]
    if any(len(ds)==0 for ds in datasets):
        raise ValueError('Train and validation splits must both be nonempty')
    return datasets


def main():
    args=parse_args()
    set_seed(args.seed)
    # Accumulation is explicitly token-normalized in the trainer.
    accelerator=Accelerator(mixed_precision='bf16' if args.bf16 else 'fp16' if args.fp16 else 'no',
                            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True,data_seed=args.seed),
                            kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True)])
    model,tokenizer=build_model(args)
    if args.model_dtype=='bfloat16':
        model.to(dtype=torch.bfloat16)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    train_ds,val_ds=build_datasets(args,tokenizer)
    loaders=[DataLoader(ds,batch_size=bs,shuffle=shuffle,num_workers=args.num_workers,collate_fn=collate)
             for ds,bs,shuffle in [(train_ds,args.batch_size,True),(val_ds,args.eval_batch_size,False)]]
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95),weight_decay=args.weight_decay)
    model,optimizer,train_loader,val_loader=accelerator.prepare(model,optimizer,*loaders)
    steps=math.ceil(len(train_loader)/args.gradient_accumulation_steps)*args.epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(steps*args.warmup_ratio),num_training_steps=steps)
    accelerator.register_for_checkpointing(scheduler)
    trainer=OmniFlattenTrainer(model,tokenizer,accelerator,args)
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    trainer.train(train_loader,val_loader,optimizer,scheduler,Path(args.output_dir)/f'stage{args.stage}')
    accelerator.end_training()

if __name__=='__main__':
    main()
