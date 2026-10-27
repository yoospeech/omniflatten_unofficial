"""Four -> three -> two stream curriculum (paper §3.3)."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from omniflatten.tokens import layout_from_tokenizer
from omniflatten.data.data_utils import load_manifest, load_audio
from omniflatten.data.formatting import sample, collate, flatten_streams, aligned_output

class _DialogueDataset(Dataset):
    def __init__(self, data_dir, tokenizer, speech_tokenizer=None, max_seq_len=8192,
                 manifest_filename='dialogue_manifest.jsonl', sample_rate=16000):
        self.data_dir, self.tokenizer, self.speech_tokenizer = Path(data_dir), tokenizer, speech_tokenizer
        self.layout = layout_from_tokenizer(tokenizer)
        self.special_ids = {t.strip('[]'): i for t,i in self.layout.special_ids.items()}
        self.entries = load_manifest(self.data_dir / manifest_filename)
        self.max_seq_len, self.sample_rate = max_seq_len, sample_rate

    def __len__(self):
        return len(self.entries)

    @torch.inference_mode()
    def _encode_speech(self, path):
        if self.speech_tokenizer is None:
            raise ValueError('Provide a pretrained codec or precomputed speech_tokens')
        return self.layout.speech_ids(self.speech_tokenizer.encode(load_audio(path, self.sample_rate)))

    def speech(self, entry, prefix=''):
        key = prefix+'speech_tokens'
        if key in entry:
            return self.layout.speech_ids(entry[key])
        return self._encode_speech(self.data_dir / entry[prefix+'audio_path'])

class HalfDuplexDialogueDataset(_DialogueDataset):
    def _flatten(self, turns):
        sequence, labels = [], []
        s = self.special_ids
        for index, turn in enumerate(turns):
            if turn['role'] != ('user' if index%2==0 else 'assistant'):
                raise ValueError('Half-duplex turns must alternate, starting with user')
            speech = [s['SOS']] + self.speech(turn) + [s['EOS']]
            text = [s['SOT']] + self.tokenizer.encode(turn['text'], add_special_tokens=False) + [s['EOT']]
            tokens = speech+text if turn['role']=='user' else text+speech
            sequence.extend(tokens)
            labels.extend([-100]*len(tokens) if turn['role']=='user' else tokens)
        return sequence, labels

    def flatten_dialogue(self, turns):
        return self._flatten(turns)[0]

    def __getitem__(self, idx):
        ids, labels = self._flatten(self.entries[idx]['turns'])
        return sample(ids, labels, self.max_seq_len)

class FullDuplexThreeStreamDataset(_DialogueDataset):
    with_text = True

    def __init__(self, data_dir, tokenizer, speech_tokenizer=None, speech_chunk_size=10,
                 text_chunk_size=2, max_seq_len=8192, manifest_filename='fullduplex_manifest.jsonl',
                 sample_rate=16000, token_rate=50):
        super().__init__(data_dir, tokenizer, speech_tokenizer, max_seq_len, manifest_filename, sample_rate)
        self.Cs, self.Ct, self.token_rate = speech_chunk_size, text_chunk_size, token_rate
        if self.Cs<=0 or self.Ct<=0:
            raise ValueError('Chunk sizes must be positive')
        if speech_tokenizer is not None and speech_tokenizer.token_rate != token_rate:
            raise ValueError('Codec and dataset token rates disagree')

    def chunk_and_flatten(self, user_speech_tokens, assistant_text_tokens, assistant_speech_tokens):
        return flatten_streams(user_speech_tokens, assistant_speech_tokens, assistant_text_tokens,
                               self.special_ids, self.Cs, self.Ct)

    def create_loss_mask(self, streams):
        return [int(s != 0) for s in streams]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        user, speech = self.speech(entry, 'user_'), self.speech(entry, 'assistant_')
        if abs(len(user)-len(speech)) > 2:
            raise ValueError('Full-duplex channels must cover the same timeline')
        if 'turns' not in entry:
            raise ValueError('Full-duplex records require turns with start_time/end_time for silence and text alignment')
        speech, text = aligned_output(speech, entry['turns'], self.tokenizer, self.special_ids,
                                      self.token_rate, self.Cs, self.Ct, self.with_text)
        ids, streams = flatten_streams(user, speech, text if self.with_text else None,
                                       self.special_ids, self.Cs, self.Ct)
        mask = self.create_loss_mask(streams)
        labels = [t if m else -100 for t,m in zip(ids,mask)]
        block = 2*self.Cs + (self.Ct if self.with_text else 0)
        return sample(ids, labels, self.max_seq_len, mask, block)

class FullDuplexTwoStreamDataset(FullDuplexThreeStreamDataset):
    with_text = False

    def __init__(self, *args, manifest_filename='fullduplex_2stream_manifest.jsonl', **kwargs):
        super().__init__(*args, manifest_filename=manifest_filename, **kwargs)

    def chunk_and_flatten(self, user_speech_tokens, assistant_speech_tokens):
        return flatten_streams(user_speech_tokens, assistant_speech_tokens, None,
                               self.special_ids, self.Cs, self.Ct)

dialogue_collate_fn = collate
