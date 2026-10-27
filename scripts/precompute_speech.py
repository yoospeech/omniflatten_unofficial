"""Convert audio manifests to raw CosyVoice IDs without changing text/timing."""
import argparse
import json
from pathlib import Path
from omniflatten.speech_tokenizer import CosyVoiceSpeechTokenizer
from omniflatten.data.data_utils import load_audio

def encode_record(record, root, codec):
    result=dict(record)
    for prefix in ('','user_','assistant_'):
        key=prefix+'audio_path'
        if key in result:
            waveform=load_audio(root/result[key],16000)
            codes=codec.encode(waveform)
            result[prefix+'speech_tokens']=codes.tolist()
            # Keep provenance paths valid even if manifests are relocated.
            result[key]=str((root/result[key]).resolve())
    if 'turns' in result:
        result['turns']=[encode_record(t,root,codec) for t in result['turns']]
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--codec_dir',required=True)
    args=parser.parse_args()
    source,target=Path(args.input),Path(args.output)
    if source.resolve()==target.resolve():
        raise ValueError('Input and output must differ')
    codec=CosyVoiceSpeechTokenizer(args.codec_dir)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    with source.open() as reader,temporary.open('w') as writer:
        for line in reader:
            if line.strip():
                writer.write(json.dumps(encode_record(json.loads(line),source.parent,codec),ensure_ascii=False)+'\n')
    temporary.replace(target)

if __name__=='__main__':
    main()
