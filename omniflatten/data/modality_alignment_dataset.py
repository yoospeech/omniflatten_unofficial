"""ASR/TTS paired-data SFT, exactly the task sequences in paper §3.2."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from omniflatten.tokens import layout_from_tokenizer
from omniflatten.data.data_utils import load_manifest, load_audio
from omniflatten.data.formatting import sample, collate

class ModalityAlignmentDataset(Dataset):
    def __init__(self, data_dir, tokenizer, speech_tokenizer, max_seq_len=1024,
                 task='both', manifest_filename='manifest.jsonl', sample_rate=16000):
        if task not in ('asr', 'tts', 'both'):
            raise ValueError('task must be asr, tts or both')
        self.data_dir, self.tokenizer, self.speech_tokenizer = Path(data_dir), tokenizer, speech_tokenizer
        self.layout = layout_from_tokenizer(tokenizer)
        self.entries = load_manifest(self.data_dir / manifest_filename)
        self.task, self.max_seq_len, self.sample_rate = task, max_seq_len, sample_rate
        self.special_ids = {t.strip('[]'): i for t,i in self.layout.special_ids.items()}

    def __len__(self):
        return len(self.entries) * (2 if self.task=='both' else 1)

    @torch.inference_mode()
    def _encode_speech(self, path):
        if self.speech_tokenizer is None:
            raise ValueError('Provide a pretrained codec or precomputed speech_tokens')
        return self.layout.speech_ids(self.speech_tokenizer.encode(load_audio(path, self.sample_rate)))

    def __getitem__(self, idx):
        task = ('asr' if idx%2==0 else 'tts') if self.task=='both' else self.task
        entry = self.entries[idx//2 if self.task=='both' else idx]
        if 'speech_tokens' in entry:
            speech = self.layout.speech_ids(entry['speech_tokens'])
        else:
            speech = self._encode_speech(self.data_dir / entry['audio_path'])
        text = self.tokenizer.encode(entry['text'], add_special_tokens=False)
        s = self.special_ids
        if task=='asr':
            # Predict [SOT] from the encoded speech [EOS], then transcribe text.
            # Keeping [SOT] in the supervised target teaches the modality
            # boundary instead of providing it only as an input prefix.
            prefix = [s['ASR'], s['SOS']] + speech + [s['EOS']]
            target = [s['SOT']] + text + [s['EOT']]
        else:
            prefix = [s['TTS'], s['SOT']] + text + [s['EOT']]
            target = [s['SOS']] + speech + [s['EOS']]
        # Condition on the source and supervise the target, a declared SFT choice.
        result = sample(prefix+target, [-100]*len(prefix)+target, self.max_seq_len)
        # Kept separate from vocabulary IDs: the trainer uses this to balance
        # ASR and TTS losses without changing either task's token sequence.
        result['task_id'] = torch.tensor(0 if task == 'asr' else 1, dtype=torch.long)
        return result

modality_alignment_collate_fn = collate
