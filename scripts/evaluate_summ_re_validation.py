"""Evaluate saved smoke checkpoints on a third, held-out meeting."""
import gc
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from omniflatten.model import OmniFlattenModel
from omniflatten.data.modality_alignment_dataset import ModalityAlignmentDataset
from omniflatten.data.dialogue_dataset import HalfDuplexDialogueDataset, FullDuplexThreeStreamDataset, FullDuplexTwoStreamDataset
from omniflatten.data.formatting import collate
from omniflatten.inference import OmniFlattenInference, InferenceConfig

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    root=Path('outputs/summ_re_validation')
    data=Path('prepared_data/summ_re_validation')
    report={'python':sys.executable,'torch':torch.__version__,
            'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'training_world_size':2,'model_dtype':'bfloat16 (parameters and Adam states; no FP32 masters)',
            'scope':'Small real-audio execution check, not convergence/quality validation',
            'preparation':json.loads((data/'preparation.json').read_text()),'stages':{}}
    for stage,cls in [('1',ModalityAlignmentDataset),('2',HalfDuplexDialogueDataset),
                      ('3a',FullDuplexThreeStreamDataset),('3b',FullDuplexTwoStreamDataset)]:
        model,tok=OmniFlattenModel.from_pretrained(root/f'stage{stage}/best_model')
        model.to(device='cuda:0',dtype=torch.bfloat16).eval()
        directory=data/('stage3' if stage in ('3a','3b') else f'stage{stage}')
        ds=cls(directory,tok,None,manifest_filename='test.jsonl')
        total=0.;count=0
        for batch in DataLoader(ds,batch_size=1,collate_fn=collate):
            batch={k:v.cuda(0) for k,v in batch.items()}
            n=int((batch['labels'][:,1:]!=-100).sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**batch).loss
            if not torch.isfinite(loss):raise FloatingPointError(stage)
            total+=loss.item()*n;count+=n
        result=json.loads((root/f'stage{stage}/metrics.json').read_text())
        result.update(test_loss=total/count,test_supervised_tokens=count,test_examples=len(ds))
        if stage in ('3a','3b'):
            entry=json.loads((directory/'test.jsonl').read_text().splitlines()[0])
            codes=entry['user_speech_tokens']
            engine=OmniFlattenInference(model,None,None,tok,InferenceConfig(mode=stage,device='cuda:0',
                max_response_chunks=2,max_new_tokens=256))
            packets=list(engine.stream_generate((codes[i:i+10] for i in range(0,len(codes),10))))
            expected_text=2 if stage=='3a' else 0
            assert all(len(p['text_tokens'])==expected_text and len(p['speech_tokens'])==10 for p in packets)
            valid=set(range(model.layout.speech_offset,model.layout.speech_offset+4096))|{engine.special_ids['SILENT_SPEECH']}
            assert all(t in valid for p in packets for t in p['speech_tokens'])
            result['generation']={'packets':len(packets),'text_tokens_per_packet':expected_text,
                                  'speech_tokens_per_packet':10,'valid_modality_ids':True,
                                  'audio_decoding_tested':False}
            del engine,packets
        report['stages'][stage]=result
        print(stage,'test_loss',result['test_loss'],flush=True)
        del model,tok,ds
        gc.collect();torch.cuda.empty_cache()
    (root/'validation_report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
