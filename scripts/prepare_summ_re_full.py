"""Prepare every SUMM-RE speech/transcript segment for stage-1 training.

Reads cached Arrow files, splits long segments using word timestamps, and
precomputes CosyVoice v1 IDs. Progress is committed per original segment.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import soundfile as sf
import torch
import torchaudio

from omniflatten.speech_tokenizer import CosyVoiceSpeechTokenizer

SPLITS = {'train': 'train', 'dev': 'val', 'test': 'test'}


def shards(root, split):
    paths = sorted(root.glob(f'*-{split}-*.arrow')) + sorted(root.glob(f'*-{split}.arrow'))
    if not paths:
        raise FileNotFoundError(f'No {split} Arrow shards in {root}')
    return paths


def iter_rows(paths, columns=None):
    for path in paths:
        with pa.memory_map(str(path), 'r') as source:
            for batch in pa.ipc.open_stream(source):
                if columns:
                    batch = batch.select(columns)
                for index in range(batch.num_rows):
                    yield path.name, batch.slice(index, 1).to_pylist()[0]


def audit_meetings(root):
    groups = {}
    for source_split, target_split in SPLITS.items():
        groups[target_split] = {row['meeting_id'] for _, row in iter_rows(
            shards(root, source_split), ['meeting_id'])}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f'Meeting overlap between {left} and {right}: {sorted(overlap)[:5]}')
    return {k: len(v) for k, v in groups.items()}


def segment_chunks(segment, max_duration):
    """Return time-aligned fragments without silently dropping transcripts."""
    start, end = float(segment['start']), float(segment['end'])
    text = segment['transcript'].strip()
    if not text or not 0 <= start < end:
        raise ValueError('Empty transcript or invalid segment timing')
    if end - start <= max_duration:
        return [(start, end, text, False)]
    words = segment.get('words') or []
    if text.split() != [word['word'] for word in words]:
        raise ValueError('Cannot safely split: transcript differs from timestamped words')
    chunks, group, group_start, group_end = [], [], None, None
    for word in words:
        a, b = float(word['start']), float(word['end'])
        if not start <= a < b <= end + 1e-5:
            raise ValueError('Invalid word timestamps')
        shortened_silence = False
        if b-a > max_duration:
            # SUMM-RE sometimes labels long pauses as one 'sil' word.
            # Preserve the label but cap only that pause; record this in output.
            if word['word'].lower() != 'sil':
                raise ValueError(f'Non-silence word exceeds {max_duration}s')
            b = a + max_duration
            shortened_silence = True
        if group and (b-group_start > max_duration or shortened_silence):
            chunks.append((group_start, group_end, ' '.join(group), False))
            group = []
        if shortened_silence:
            chunks.append((a, b, word['word'], True))
            continue
        if not group:
            group_start = a
        group.append(word['word'])
        group_end = b
    if group:
        chunks.append((group_start, group_end, ' '.join(group), False))
    if not chunks or ' '.join(c[2] for c in chunks).split() != text.split():
        raise ValueError('Long-segment splitting did not preserve transcript words')
    return chunks


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def fingerprint(root, codec, max_duration):
    h = hashlib.sha256()
    for path in [*sum((shards(root, name) for name in SPLITS), []), codec / 'speech_tokenizer_v1.onnx']:
        stat = path.stat()
        h.update(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    h.update(f'v1:max_duration={max_duration}'.encode())
    return h.hexdigest()


def process_split(source_split, target_split, root, output, codec, signature,
                  max_duration, max_segments):
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / f'.{target_split}.state.json'
    part_path = output / f'{target_split}.jsonl.part'
    final_path = output / f'{target_split}.jsonl'
    review_path = output / f'{target_split}.review.jsonl'
    state = {'signature': signature, 'next_row': 0, 'next_segment': 0, 'bytes': 0,
             'segments': 0, 'records': 0, 'shortened_silences': 0, 'review_segments': 0,
             'review_bytes': 0, 'complete': False}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['signature'] != signature:
            raise ValueError(f'{state_path}: input or options changed; choose a new output directory')
        if state['complete']:
            if not final_path.is_file() or final_path.stat().st_size != state['bytes']:
                raise ValueError(f'{final_path}: completed output is missing or changed')
            if not review_path.exists() or review_path.stat().st_size != state['review_bytes']:
                raise ValueError(f'{review_path}: review output is missing or changed')
            print(f'{target_split}: already complete ({state["records"]} records)', flush=True)
            return state
    elif final_path.exists() or part_path.exists() or review_path.exists():
        raise FileExistsError(f'{target_split}: existing output without state; choose a new directory')
    else:
        atomic_json(state_path, state)
    if max_segments and state['segments'] >= max_segments:
        print(f'{target_split}: already at test limit ({state["segments"]} segments)', flush=True)
        return state
    if final_path.exists() and not part_path.exists():
        os.replace(final_path, part_path)  # Recover interrupted finalization.
    if part_path.exists() and part_path.stat().st_size < state['bytes']:
        raise ValueError(f'{part_path}: output is shorter than its checkpoint')
    if review_path.exists() and review_path.stat().st_size < state['review_bytes']:
        raise ValueError(f'{review_path}: review output is shorter than its checkpoint')
    with part_path.open('a+b') as output_file, review_path.open('a+b') as review_file:
        review_file.truncate(state['review_bytes'])
        review_file.seek(state['review_bytes'])
        output_file.truncate(state['bytes'])
        output_file.seek(state['bytes'])
        for row_index, (shard_name, row) in enumerate(iter_rows(shards(root, source_split))):
            if row_index < state['next_row']:
                continue
            audio = row['audio']
            src = io.BytesIO(audio['bytes']) if audio.get('bytes') else audio['path']
            with sf.SoundFile(src) as recording:
                sr = recording.samplerate
                for seg_index, segment in enumerate(row['segments']):
                    if row_index == state['next_row'] and seg_index < state['next_segment']:
                        continue
                    try:
                        fragments = segment_chunks(segment, max_duration)
                        encoded = []
                        for start, end, text, shortened in fragments:
                            a, b = round(start*sr), round(end*sr)
                            if b > len(recording) or b <= a:
                                raise ValueError('Segment lies outside recording')
                            recording.seek(a)
                            waveform = recording.read(b-a, dtype='float32', always_2d=True).mean(axis=1)
                            if not len(waveform) or not np.isfinite(waveform).all():
                                raise ValueError('Invalid audio waveform')
                            waveform = torch.from_numpy(waveform)
                            if sr != 16000:
                                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                            codes = codec.encode(waveform).tolist()
                            if not codes:
                                raise ValueError('Codec produced no tokens')
                            encoded.append({'speech_tokens': codes, 'text': text,
                                'meeting_id': row['meeting_id'], 'speaker_id': row['speaker_id'],
                                'audio_id': row['audio_id'], 'source_shard': shard_name,
                                'source_segment_start': segment['start'], 'source_segment_end': segment['end'],
                                'source_start': start, 'source_end': end,
                                'shortened_silence': shortened})
                    except ValueError as exc:
                        if not str(exc).startswith('Non-silence word exceeds'):
                            raise RuntimeError(f'{target_split} row {row_index} segment {seg_index} '
                                               f'meeting {row["meeting_id"]}: {exc}') from exc
                        encoded = []
                        review = {'reason':str(exc), 'meeting_id':row['meeting_id'],
                                  'speaker_id':row['speaker_id'], 'audio_id':row['audio_id'],
                                  'source_shard':shard_name, 'segment':segment}
                        review_file.write((json.dumps(review,ensure_ascii=False)+'\n').encode('utf-8'))
                        state['review_segments'] += 1
                    except Exception as exc:
                        raise RuntimeError(f'{target_split} row {row_index} segment {seg_index} '
                                           f'meeting {row["meeting_id"]}: {exc}') from exc
                    for record in encoded:
                        output_file.write((json.dumps(record, ensure_ascii=False)+'\n').encode('utf-8'))
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    review_file.flush()
                    os.fsync(review_file.fileno())
                    state.update(next_row=row_index, next_segment=seg_index+1, bytes=output_file.tell(),
                                 review_bytes=review_file.tell(),
                                 segments=state['segments']+1, records=state['records']+len(encoded),
                                 shortened_silences=state['shortened_silences']+sum(r['shortened_silence'] for r in encoded))
                    atomic_json(state_path, state)
                    if state['segments'] % 100 == 0:
                        print(f'{target_split}: {state["segments"]} segments, '
                              f'{state["records"]} examples, row {row_index}', flush=True)
                    if max_segments and state['segments'] >= max_segments:
                        print(f'{target_split}: stopped at test limit; rerun without --max-segments to resume', flush=True)
                        return state
    os.replace(part_path, final_path)
    state['complete'] = True
    atomic_json(state_path, state)
    print(f'{target_split}: complete; {state["segments"]} segments -> {state["records"]} examples; '
          f'{state["shortened_silences"]} long silence spans shortened; '
          f'{state["review_segments"]} segments require review', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrow-dir', type=Path, required=True)
    parser.add_argument('--codec-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-duration', type=float, default=12.0)
    parser.add_argument('--max-segments', type=int, default=0, help='Per split; 0 processes all segments')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:<GPU index>')
    args = parser.parse_args()
    if not math.isfinite(args.max_duration) or not 0 < args.max_duration <= 30:
        parser.error('--max-duration must be > 0 and <= 30')
    if args.max_segments < 0 or args.threads < 1:
        parser.error('max-segments must be nonnegative and threads positive')
    torch.set_num_threads(args.threads)
    meetings = audit_meetings(args.arrow_dir)
    signature = fingerprint(args.arrow_dir, args.codec_dir, args.max_duration)
    print(f'SUMM-RE meetings per split: {meetings}; existing official splits retained', flush=True)
    codec = CosyVoiceSpeechTokenizer(args.codec_dir, num_threads=args.threads, device=args.device)
    print(f'ONNX Runtime providers: {codec.session.get_providers()}', flush=True)
    results = {}
    for source_split, target_split in SPLITS.items():
        results[target_split] = process_split(source_split, target_split, args.arrow_dir,
            args.output_dir, codec, signature, args.max_duration, args.max_segments)
    if all(state['complete'] for state in results.values()):
        atomic_json(args.output_dir / 'source.json', {'dataset': 'linagora/SUMM-RE',
            'config': 'default', 'scope': 'stage-1 ASR/TTS', 'arrow_dir': str(args.arrow_dir.resolve()),
            'codec_dir': str(args.codec_dir.resolve()), 'max_duration': args.max_duration,
            'signature': signature, 'meetings': meetings,
            'splits': {name: {k: state[k] for k in ('segments','records','shortened_silences','review_segments')}
                       for name, state in results.items()}})
        print('All splits are ready for Stage 1.', flush=True)
    else:
        print('Partial run: no final manifest is published until its split is complete.', flush=True)

if __name__ == '__main__':
    main()
