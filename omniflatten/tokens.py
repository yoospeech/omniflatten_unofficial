"""One persisted vocabulary contract for preprocessing, training and inference."""
from dataclasses import dataclass
import torch

SPECIAL_TOKENS = ['[ASR]', '[TTS]', '[SOS]', '[EOS]', '[SOT]', '[EOT]',
                  '[SILENT_TEXT]', '[SILENT_SPEECH]']

@dataclass
class TokenLayout:
    text_vocab_size: int
    speech_offset: int
    num_speech_tokens: int
    special_ids: dict

    def speech_ids(self, codes):
        codes = normalize_codes(codes, self.num_speech_tokens)
        return [self.speech_offset + c for c in codes]

    def codebook_ids(self, ids):
        return normalize_codes([i - self.speech_offset for i in ids], self.num_speech_tokens)


def normalize_codes(value, size=4096):
    # Also accepts legacy (codes, auxiliary_loss) encoders.
    if isinstance(value, tuple):
        value = value[0]
    value = torch.as_tensor(value)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or (value.numel() and value.is_floating_point()):
        raise ValueError('Expected a single integer codebook sequence, shape (T,) or (1,T)')
    codes = value.cpu().tolist()
    if any(c < 0 or c >= size for c in codes):
        raise ValueError(f'Speech codebook IDs must be in [0, {size})')
    return codes


def layout_from_tokenizer(tokenizer):
    raw = tokenizer.init_kwargs.get('omniflatten_layout')
    if raw is None:
        raise ValueError('Use the tokenizer saved by OmniFlattenModel.prepare_tokenizer()')
    layout = TokenLayout(**raw)
    for token, idx in layout.special_ids.items():
        if tokenizer.encode(token, add_special_tokens=False) != [idx]:
            raise ValueError(f'Tokenizer layout mismatch: {token}')
    return layout
