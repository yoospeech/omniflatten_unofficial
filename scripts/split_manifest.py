"""Assign whole dialogues to deterministic 98/1/1 splits shared by all stages."""
import argparse
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack


def split_for_id(dialogue_id, seed=42):
    digest=hashlib.sha256(f'{seed}:{dialogue_id}'.encode()).digest()
    bucket=int.from_bytes(digest[:8],'big')%10000
    return 'val' if bucket<100 else 'test' if bucket<200 else 'train'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    source=Path(args.input)
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    def resolve(record):
        for key,value in list(record.items()):
            if key.endswith('audio_path'):
                record[key]=str((source.parent/value).resolve())
            elif key=='turns':
                for turn in value: resolve(turn)
        return record
    counts=dict.fromkeys(['train','val','test'],0)
    if source.resolve() in {(root/f'{s}.jsonl').resolve() for s in counts}:
        raise ValueError('Split output must not overwrite input')
    with ExitStack() as stack:
        outputs={s:stack.enter_context((root/f'{s}.jsonl').open('w')) for s in counts}
        for line in source.read_text().splitlines():
            if not line.strip(): continue
            record=json.loads(line)
            if 'dialogue_id' not in record:
                raise ValueError('Each record needs dialogue_id for group-safe splits')
            split=split_for_id(record['dialogue_id'],args.seed)
            outputs[split].write(json.dumps(resolve(record),ensure_ascii=False)+'\n')
            counts[split]+=1
    print(counts)
    if not all(counts.values()):
        print('Some splits are empty: a tiny corpus is insufficient for 98/1/1 splitting.')

if __name__=='__main__':main()
