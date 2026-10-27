"""Export cached SUMM-RE segments to stage-1 manifests; preserve official splits.

No role inference: meeting participants are not automatically user/assistant.
Uses Arrow directly so existing HF caches stay read-only and need no download.
"""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import soundfile as sf


def iter_rows(directory, split):
    files = sorted(Path(directory).glob(f'*-{split}-*.arrow')) + sorted(Path(directory).glob(f'*-{split}.arrow'))
    if not files:
        raise FileNotFoundError(f'No {split} Arrow files in {directory}')
    for path in files:
        with pa.memory_map(str(path),'r') as source:
            for batch in pa.ipc.open_stream(source):
                # One row can contain an entire meeting recording.
                for i in range(batch.num_rows):
                    yield batch.slice(i,1).to_pylist()[0]


def export(directory, output, max_segments=0, max_duration=12.0):
    output=Path(output)
    if max_segments<0 or max_duration<=0:
        raise ValueError('Invalid segment limits')
    counts={}
    meeting_sets={}
    for source_split,target_split in [('train','train'),('dev','val'),('test','test')]:
        (output/target_split).mkdir(parents=True,exist_ok=True)
        count=0; meeting_sets[target_split]=set()
        with (output/f'{target_split}.jsonl').open('w',encoding='utf-8') as manifest:
            for row in iter_rows(directory,source_split):
                audio=row['audio']
                src=io.BytesIO(audio['bytes']) if audio.get('bytes') else audio['path']
                with sf.SoundFile(src) as recording:
                    sr=recording.samplerate
                    for segment in row['segments']:
                        start,end=segment['start'],segment['end']
                        text=segment['transcript'].strip()
                        if not text or not 0<=start<end or end-start>max_duration:
                            continue
                        a,b=round(start*sr),round(end*sr)
                        if b>len(recording):
                            raise ValueError('Segment timestamp exceeds audio duration')
                        recording.seek(a)
                        waveform=recording.read(b-a,dtype='float32',always_2d=True).mean(axis=1)
                        if not len(waveform) or not np.isfinite(waveform).all():
                            continue
                        path=output/target_split/f'{count:08d}.wav'
                        sf.write(path,waveform,sr)
                        manifest.write(json.dumps({'audio_path':str(path.relative_to(output)), 'text':text,
                            'meeting_id':row['meeting_id'],'speaker_id':row['speaker_id'],
                            'source_start':start,'source_end':end},ensure_ascii=False)+'\n')
                        meeting_sets[target_split].add(row['meeting_id'])
                        count+=1
                        if max_segments and count>=max_segments:
                            break
                if max_segments and count>=max_segments:
                    break
        counts[target_split]=count
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if meeting_sets[a]&meeting_sets[b]:
            raise ValueError(f'Meetings overlap between {a} and {b}; do not train until split leakage is resolved')
    (output/'source.json').write_text(json.dumps({'dataset':'linagora/SUMM-RE','counts':counts,
        'purpose':'stage-1 paired audio/transcript; not synthetic assistant dialogues'},indent=2))
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrow_dir',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--max_segments',type=int,default=0,help='Per split; 0 exports all eligible segments')
    parser.add_argument('--max_duration',type=float,default=12.0)
    args=parser.parse_args()
    print(export(args.arrow_dir,args.output_dir,args.max_segments,args.max_duration))

if __name__=='__main__':
    main()
