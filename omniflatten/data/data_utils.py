"""
Utility functions for OmniFlatten data pipelines.

Provides helper functions for manifest loading, audio processing,
noise augmentation (Section 4.1), and sequence manipulation used
across all dataset classes.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import torch
import torchaudio

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL manifest file.

    Each line in the file must be a valid JSON object.  Blank lines and
    lines starting with ``#`` are silently skipped.

    Args:
        manifest_path: Path to the ``.jsonl`` manifest file.

    Returns:
        List of parsed dictionaries, one per non-empty line.

    Raises:
        FileNotFoundError: If *manifest_path* does not exist.
        json.JSONDecodeError: If a non-blank, non-comment line is invalid JSON.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid JSON on line %d of %s: %s",
                    line_no,
                    manifest_path,
                    exc,
                )
                raise
    logger.info("Loaded %d entries from %s", len(entries), manifest_path)
    return entries


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(
    audio_path: Union[str, Path],
    target_sample_rate: int = 16_000,
) -> torch.Tensor:
    """Load an audio file and resample to *target_sample_rate*.

    Args:
        audio_path: Path to an audio file readable by ``torchaudio``.
        target_sample_rate: Desired sample rate in Hz.

    Returns:
        Mono waveform tensor of shape ``(1, num_samples)``.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError: If the audio file cannot be decoded.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # SoundFile avoids torchaudio 2.9's optional TorchCodec/FFmpeg dependency.
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype='float32', always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())

    # Convert to mono if necessary
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    return waveform


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    power: float = 2.0,
) -> torch.Tensor:
    """Compute a mel spectrogram from a raw waveform.

    Args:
        waveform: Raw waveform tensor of shape ``(1, num_samples)`` or
            ``(num_samples,)``.
        sample_rate: Sample rate of the waveform in Hz.
        n_mels: Number of mel filter-bank channels.
        n_fft: FFT window size.
        hop_length: Hop length in samples.
        power: Exponent for the magnitude spectrogram (2.0 = power spectrogram).

    Returns:
        Mel spectrogram tensor of shape ``(n_mels, time_frames)``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    mel_spec = mel_transform(waveform)  # (1, n_mels, T)
    return mel_spec.squeeze(0)  # (n_mels, T)


# ---------------------------------------------------------------------------
# Noise augmentation (Section 4.1)
# ---------------------------------------------------------------------------

def add_noise(
    clean_audio: torch.Tensor,
    noise_audio: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Mix *noise_audio* into *clean_audio* at a specified SNR.

    The noise signal is repeated or truncated to match the length of the
    clean signal before mixing.

    Args:
        clean_audio: Clean waveform, shape ``(1, T)`` or ``(T,)``.
        noise_audio: Noise waveform, shape ``(1, T')`` or ``(T',)``.
        snr_db: Desired signal-to-noise ratio in decibels.

    Returns:
        Noisy waveform with the same shape as *clean_audio*.
    """
    # Normalise to 2-D
    squeeze = clean_audio.dim() == 1
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)
    if noise_audio.dim() == 1:
        noise_audio = noise_audio.unsqueeze(0)

    clean_len = clean_audio.size(-1)
    noise_len = noise_audio.size(-1)

    # Repeat / trim noise to match clean length
    if noise_len < clean_len:
        repeats = math.ceil(clean_len / noise_len)
        noise_audio = noise_audio.repeat(1, repeats)
    noise_audio = noise_audio[:, :clean_len]

    # Compute scaling factor for the desired SNR
    clean_power = clean_audio.pow(2).mean()
    noise_power = noise_audio.pow(2).mean()

    # Avoid division by zero for silent signals
    eps = 1e-10
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = torch.sqrt(clean_power / (noise_power * snr_linear + eps))

    noisy = clean_audio + scale * noise_audio

    if squeeze:
        noisy = noisy.squeeze(0)
    return noisy


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def pad_or_truncate(
    sequence: List[T],
    max_len: int,
    pad_value: T,
) -> List[T]:
    """Pad or truncate *sequence* to exactly *max_len* elements.

    Args:
        sequence: Input list of elements.
        max_len: Desired output length. Must be > 0.
        pad_value: Value used for right-padding when the sequence is shorter
            than *max_len*.

    Returns:
        A new list of length *max_len*.
    """
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [pad_value] * (max_len - len(sequence))


def pad_sequence_to_length(
    tensor: torch.Tensor,
    target_length: int,
    pad_value: int = 0,
) -> torch.Tensor:
    """Right-pad a 1-D tensor to *target_length*.

    Args:
        tensor: 1-D input tensor.
        target_length: Desired length. If the tensor is already longer it is
            truncated.
        pad_value: Scalar padding value.

    Returns:
        1-D tensor of length *target_length*.
    """
    length = tensor.size(0)
    if length >= target_length:
        return tensor[:target_length]
    padding = torch.full(
        (target_length - length,), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=0)


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Create a binary attention mask (1 = attend, 0 = padding).

    Args:
        input_ids: Token IDs tensor of shape ``(seq_len,)`` or
            ``(batch, seq_len)``.
        pad_token_id: Token ID used for padding.

    Returns:
        Tensor of the same shape with 1 where ``input_ids != pad_token_id``.
    """
    return (input_ids != pad_token_id).long()
