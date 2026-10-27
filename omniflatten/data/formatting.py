"""Paper sequence formats; all functions take unified vocabulary IDs."""
import math
import torch
from torch.nn.utils.rnn import pad_sequence


def sample(sequence, labels, max_seq_len, loss_mask=None, block_size=1):
    limit = max_seq_len // block_size * block_size
    sequence, labels = sequence[:limit], labels[:limit]
    if len(sequence) < 2 or not any(i != -100 for i in labels[1:]):
        raise ValueError('Sample has no supervised next tokens; split long utterances before encoding')
    result = {'input_ids': torch.tensor(sequence, dtype=torch.long),
              'labels': torch.tensor(labels, dtype=torch.long),
              'attention_mask': torch.ones(len(sequence), dtype=torch.long)}
    if loss_mask is not None:
        result['loss_mask'] = torch.tensor(loss_mask[:limit], dtype=torch.long)
    return result


def collate(batch, pad_token_id=0, label_pad_value=-100):
    result = {key: pad_sequence([s[key] for s in batch], batch_first=True, padding_value=pad)
              for key, pad in [('input_ids', pad_token_id), ('labels', label_pad_value), ('attention_mask', 0)]}
    if any('loss_mask' in s for s in batch):
        result['loss_mask'] = pad_sequence(
            [s.get('loss_mask', (s['labels'] != -100).long()) for s in batch],
            batch_first=True, padding_value=0)
    return result


def flatten_streams(user, speech, text, sid, speech_chunk_size=10, text_chunk_size=2):
    cs, ct = speech_chunk_size, text_chunk_size
    if cs <= 0 or ct <= 0:
        raise ValueError('Chunk sizes must be positive')
    steps = max(math.ceil(len(user)/cs), math.ceil(len(speech)/cs),
                math.ceil(len(text)/ct) if text is not None else 0)
    def padded(values, size, token):
        return list(values) + [token] * (size-len(values))
    user = padded(user, steps*cs, sid['SILENT_SPEECH'])
    speech = padded(speech, steps*cs, sid['SILENT_SPEECH'])
    if text is not None:
        text = padded(text, steps*ct, sid['SILENT_TEXT'])
    ids, streams = [], []
    for n in range(steps):
        ids.extend(user[n*cs:(n+1)*cs]); streams.extend([0]*cs)
        if text is not None:
            ids.extend(text[n*ct:(n+1)*ct]); streams.extend([1]*ct)
        ids.extend(speech[n*cs:(n+1)*cs]); streams.extend([2 if text is not None else 1]*cs)
    return ids, streams


def aligned_output(speech, turns, tokenizer, sid, token_rate=50, cs=10, ct=2, with_text=True):
    """Keep speech on its timeline and start text at its utterance's chunk.

    No forced word alignment is assumed. Reject text that cannot fit in its
    utterance's chunk interval; never silently move future text into the past.
    Truncated assistant utterances need corrected text before 3-stream use.
    """
    steps = math.ceil(len(speech)/cs)
    out_speech = [sid['SILENT_SPEECH']] * len(speech)
    out_text = [sid['SILENT_TEXT']] * (steps*ct)
    occupied = set()
    previous_end = 0.0
    for turn in sorted((t for t in turns if t['role']=='assistant'), key=lambda t:t['start_time']):
        start, end = float(turn['start_time']), float(turn['end_time'])
        if not 0 <= start < end or start < previous_end:
            raise ValueError('Invalid or overlapping assistant timestamps')
        previous_end = end
        a, b = round(start*token_rate), min(len(speech), round(end*token_rate))
        if a >= len(speech) or end*token_rate > len(speech)+2:
            raise ValueError('Assistant timestamps exceed encoded speech timeline')
        out_speech[a:b] = speech[a:b]
        if not with_text:
            continue
        if turn.get('truncated') and 'spoken_text' not in turn:
            raise ValueError('Interrupted 3-stream turn requires spoken_text for the audible prefix')
        text = tokenizer.encode(turn.get('spoken_text', turn['text']), add_special_tokens=False)
        first = math.floor(start*token_rate/cs)*ct
        last = min(steps, math.ceil(end*token_rate/cs))*ct
        if len(text) > last-first:
            raise ValueError('Text exceeds its speech chunk budget; use longer speech or corrected transcript')
        for i, token in enumerate(text, first):
            if i in occupied:
                raise ValueError('Assistant text chunks overlap')
            occupied.add(i)
            out_text[i] = token
    return out_speech, out_text
