"""Prepare a reproducible, real-audio SUMM-RE smoke curriculum.

Meeting speakers are mapped to user/assistant only for execution testing.
This is not the paper's assistant-response synthesis dataset or a quality eval.
"""
import argparse
import io
import json
import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio


def prepare(cache_dir, codec_dir, backbone, output_dir, examples=2):
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('HF_DATASETS_OFFLINE','1')
    import datasets
    from transformers import AutoTokenizer
    from omniflatten.speech_tokenizer import CosyVoiceSpeechTokenizer
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    torch.set_num_threads(2)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    # The supplied example configuration isn't present in this download.
    # Keep the requested API and explicitly record the cached config used.
    # HF load_dataset writes a lock even when reusing cached data. Link the
    # large Arrow shards into a writable cache instead of changing originals.
    source=Path(datasets.config.HF_DATASETS_CACHE)/'linagora___summ-re'
    destination=Path(cache_dir)/'linagora___summ-re'
    if source.resolve()!=destination.resolve() and source.exists():
        for path in source.rglob('*'):
            if path.suffix not in ('.json','.arrow'):continue
            target=destination/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                if path.suffix=='.arrow':target.symlink_to(path.resolve())
                else:shutil.copyfile(path,target)
    actual_config='example' if (destination/'example').is_dir() else 'default'
    ds=datasets.load_dataset('linagora/SUMM-RE',actual_config,cache_dir=cache_dir)
    ds=ds.cast_column('audio',datasets.Audio(decode=False))
    tokenizer=AutoTokenizer.from_pretrained(backbone,local_files_only=True)
    codec=CosyVoiceSpeechTokenizer(codec_dir)
    report={'dataset':'linagora/SUMM-RE','requested_config':'example','actual_config':actual_config,
            'rows':{s:len(ds[s]) for s in ds},'codec_dir':str(Path(codec_dir).resolve()),
            'backbone':str(Path(backbone).resolve()),
            'scope':'Real meeting audio; speaker-role mapping for smoke tests, not assistant dialogue quality',
            'splits':{}}
    for source_split,split in [('train','train'),('dev','val'),('test','test')]:
        # One whole meeting per split; official split boundaries are preserved.
        meeting=ds[source_split][0]['meeting_id']
        rows=[]
        for i in range(len(ds[source_split])):
            row=ds[source_split][i]
            if row['meeting_id']!=meeting: break
            rows.append(row)
        segments=sorted([dict(t,speaker=row['speaker_id'],row=i) for i,row in enumerate(rows)
                         for t in row['segments']],key=lambda t:t['start'])
        # Prefer genuine overlapping two-speaker exchanges where the fixed
        # text/speech chunk ratio can accommodate the real transcript.
        candidates=[]
        for i,u in enumerate(segments):
            for a in segments[i+1:i+8]:
                duration=max(u['end'],a['end'])-u['start']
                if a['speaker']==u['speaker'] or duration>5 or duration<0.5: continue
                if min(u['end']-u['start'],a['end']-a['start'])<0.3: continue
                if not u['transcript'].strip() or not a['transcript'].strip(): continue
                first=int(np.floor((a['start']-u['start'])*5))*2
                last=int(np.ceil((a['end']-u['start'])*5))*2
                text=tokenizer.encode(a['transcript'],add_special_tokens=False)
                if len(text)>last-first: continue
                overlap=max(0,min(u['end'],a['end'])-a['start'])
                candidates.append((overlap<=0,duration,u,a,overlap))
        candidates.sort(key=lambda p:(p[0],p[1]))
        chosen=[];used=set()
        for candidate in candidates:
            _,_,u,a,_=candidate
            keys=[(t['speaker'],t['start']) for t in (u,a)]
            if any(k in used for k in keys): continue
            chosen.append(candidate);used.update(keys)
            if len(chosen)==examples:break
        if len(chosen)!=examples:
            raise ValueError(f'Insufficient short aligned exchanges in {meeting}')
        # Decode each recording once; source channels remain speaker-specific.
        recordings={}
        for row_index in {t['row'] for c in chosen for t in c[2:4]}:
            value=rows[row_index]['audio']
            source=io.BytesIO(value['bytes']) if value.get('bytes') else value['path']
            audio,sr=sf.read(source,dtype='float32',always_2d=True)
            recordings[row_index]=(audio.mean(1),sr)
        stage1=[];stage2=[];stage3=[];details=[]
        for index,(_,duration,u,a,overlap) in enumerate(chosen):
            start=u['start'];length=round(duration*16000)
            turns=[];channels=[]
            for role,segment in [('user',u),('assistant',a)]:
                audio,sr=recordings[segment['row']]
                clip=audio[round(segment['start']*sr):round(segment['end']*sr)].copy()
                waveform=torchaudio.functional.resample(torch.from_numpy(clip),sr,16000)
                codes=codec.encode(waveform).tolist()
                stage1.append({'speech_tokens':codes,'text':segment['transcript'], 'meeting_id':meeting,
                               'speaker_id':segment['speaker']})
                turns.append({'role':role,'speech_tokens':codes,'text':segment['transcript']})
                channel=torch.zeros(length)
                offset=round((segment['start']-start)*16000)
                n=min(len(waveform),length-offset)
                channel[offset:offset+n]=waveform[:n]
                # Preserve actual segment placement, including genuine overlap.
                # Other turns in this recording are excluded from this tiny fixture.
                channels.append(codec.encode(channel).tolist())
            identifier=f'{meeting}-smoke-{index}'
            stage2.append({'dialogue_id':identifier,'turns':turns})
            timed=[{'role':role,'text':t['transcript'],'start_time':t['start']-start,
                    'end_time':t['end']-start} for role,t in [('user',u),('assistant',a)]]
            stage3.append({'dialogue_id':identifier,'user_speech_tokens':channels[0],
                           'assistant_speech_tokens':channels[1],'turns':timed})
            details.append({'dialogue_id':identifier,'source_start':start,'duration':duration,
                            'overlap_seconds':overlap,'speakers':[u['speaker'],a['speaker']],
                            'channel_tokens':[len(c) for c in channels]})
        for name,records in [('stage1',stage1),('stage2',stage2),('stage3',stage3)]:
            (output/name).mkdir(exist_ok=True)
            (output/name/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        report['splits'][split]={'meeting':meeting,'examples':details}
        print(split,meeting,details,flush=True)
    if len({s['meeting'] for s in report['splits'].values()})!=3:
        raise ValueError('Meetings must not cross dataset splits')
    (output/'preparation.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir',default='.validation/hf_cache')
    parser.add_argument('--codec_dir',default='pretrained_models/CosyVoice-300M')
    parser.add_argument('--backbone',default='pretrained_models/Qwen2-0.5B')
    parser.add_argument('--output_dir',default='prepared_data/summ_re_validation')
    args=parser.parse_args()
    prepare(**vars(args))

if __name__=='__main__':main()
