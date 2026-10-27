"""
Dialogue Data Synthesis and Simulation Pipeline (Section 4.1, Figure 2).

This pipeline synthesizes full-duplex spoken dialogue data from text dialogues,
following the OmniFlatten paper's data construction methodology.

Stages:
    1. **Prepare** multi-turn dialogue text from open-source datasets
       (Alpaca, Moss, BelleCN, ultraChat).
    2. **Filter** out inappropriate samples (code, math, >200 words, rare symbols).
    3. **Synthesize** speech using pretrained CosyVoice.
    4. **Simulate** interaction dynamics:
       - *Normal*: User finishes, assistant responds immediately.
       - *User interrupts*: User interrupts, assistant stops mid-speech.
       - *Assistant waits*: Assistant finishes, waits for user.
    5. **Augment** by adding background noise from the MUSAN dataset (SNR 15-30 dB).
    6. **Output** two-channel (user, assistant) audio + alignment manifests.

Output format — JSONL manifest, one record per dialogue::

    {
        "dialogue_id": "...",
        "user_audio_path": "...",
        "assistant_audio_path": "...",
        "turns": [
            {
                "role": "user" | "assistant",
                "text": "...",
                "start_time": float,
                "end_time": float
            },
            ...
        ],
        "interaction_type": "normal" | "user_interrupt" | "assistant_wait",
        "total_duration": float
    }

Usage::

    python -m omniflatten.data.simulate_dialogue \\
        --input dialogues.jsonl \\
        --output_dir ./simulated_data \\
        --noise_dir ./musan/noise \\
        --num_workers 1 --codec_dir /path/to/CosyVoice-300M-SFT --assistant_speaker_id SPEAKER

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default audio sample rate (Hz) used throughout the pipeline.
DEFAULT_SAMPLE_RATE: int = 16_000

#: Regex patterns used to detect code / math content that is unsuitable for TTS.
_CODE_PATTERNS: list[re.Pattern] = [
    re.compile(r"```"),                           # fenced code blocks
    re.compile(r"(?:def |class |import |from )"),  # Python keywords
    re.compile(r"[{}\[\];]"),                      # C-family braces / brackets
    re.compile(r"\\(?:frac|sum|int|sqrt)\b"),      # LaTeX math macros
    re.compile(r"\$\$.+?\$\$", re.DOTALL),         # display math
]

#: Characters considered "rare / unusual" for spoken dialogue.
_RARE_SYMBOL_PATTERN: re.Pattern = re.compile(
    r"[^\w\s.,!?;:'\"\-()&@#%/\\+\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"
)


# ---------------------------------------------------------------------------
# Configuration dataclass (lightweight — no external config library needed)
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All tuneable knobs for :class:`DialogueSimulationPipeline`.

    Attributes:
        max_text_words:       Maximum word / token count per utterance.
        noise_data_dir:       Path to the MUSAN ``noise`` subdirectory.
        dialogue_data_dir:    Root directory for all pipeline outputs.
        sample_rate:          Audio sample rate in Hz.
        interrupt_prob:       Probability of a *user-interrupt* interaction for any
                              turn pair.
        wait_silence_range:   (min, max) seconds of silence appended when the
                              assistant is waiting for the user.
        snr_min:              Minimum signal-to-noise ratio (dB) for noise
                              augmentation.
        snr_max:              Maximum signal-to-noise ratio (dB).
        num_workers:          Number of parallel workers for :meth:`run`.
    """

    max_text_words: int = 200
    noise_data_dir: Optional[str] = None
    dialogue_data_dir: str = "./simulated_data"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    interrupt_prob: float = 0.15
    wait_silence_range: Tuple[float, float] = (0.5, 2.0)
    snr_min: float = 15.0
    snr_max: float = 30.0
    num_workers: int = 1
    codec_dir: Optional[str] = None
    assistant_speaker_id: Optional[str] = None


# ============================================================================
# Stage 2 — Text Filtering
# ============================================================================


class TextDialogueFilter:
    """Filter text dialogues unsuitable for TTS synthesis (Section 4.1).

    Criteria for *rejecting* an utterance:

    * Contains a high proportion of code or LaTeX math markup.
    * Exceeds *max_words* words (Chinese characters each count as one word).
    * Contains a high proportion of rare / non-printable symbols.

    Args:
        max_words:             Maximum allowed word count per utterance.
        code_line_threshold:   If ≥ this fraction of lines match code patterns
                               the text is rejected.
        rare_symbol_threshold: If ≥ this fraction of characters are rare the
                               text is rejected.
    """

    def __init__(
        self,
        max_words: int = 200,
        code_line_threshold: float = 0.30,
        rare_symbol_threshold: float = 0.05,
    ) -> None:
        self.max_words = max_words
        self.code_line_threshold = code_line_threshold
        self.rare_symbol_threshold = rare_symbol_threshold

    # ------------------------------------------------------------------
    # Word counting helper (handles mixed CJK + Latin text)
    # ------------------------------------------------------------------

    @staticmethod
    def _count_words(text: str) -> int:
        """Return an approximate word count for mixed CJK / Latin text.

        Latin words are whitespace-delimited.  Each CJK character counts as
        one word because Chinese is not whitespace-segmented.
        """
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # Remove CJK chars, then count remaining space-delimited tokens.
        latin_text = re.sub(r"[\u4e00-\u9fff]", " ", text)
        latin_words = len(latin_text.split())
        return cjk_chars + latin_words

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _has_too_much_code(self, text: str) -> bool:
        """Return ``True`` if the text contains significant code / math."""
        lines = text.splitlines() or [""]
        if len(lines) == 0:
            return False
        flagged = sum(
            1
            for line in lines
            if any(pat.search(line) for pat in _CODE_PATTERNS)
        )
        return (flagged / max(len(lines), 1)) >= self.code_line_threshold

    def _too_many_words(self, text: str) -> bool:
        """Return ``True`` if the text exceeds the word-count limit."""
        return self._count_words(text) > self.max_words

    def _has_rare_symbols(self, text: str) -> bool:
        """Return ``True`` if the text has too many unusual characters."""
        if not text:
            return False
        rare_count = len(_RARE_SYMBOL_PATTERN.findall(text))
        return (rare_count / len(text)) >= self.rare_symbol_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid(self, text: str) -> bool:
        """Check whether *text* is suitable for TTS synthesis.

        Returns:
            ``True`` if the text passes all checks; ``False`` otherwise.
        """
        if not text or not text.strip():
            return False
        if self._has_too_much_code(text):
            logger.debug("Rejected (code/math): %.60s…", text)
            return False
        if self._too_many_words(text):
            logger.debug("Rejected (too many words): %.60s…", text)
            return False
        if self._has_rare_symbols(text):
            logger.debug("Rejected (rare symbols): %.60s…", text)
            return False
        return True

    def filter_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of dialogues, removing those with any unsuitable turn.

        Each dialogue is expected to contain a ``"turns"`` key whose value is a
        list of ``{"role": ..., "text": ...}`` dicts.

        Args:
            dialogues: Input dialogue dicts.

        Returns:
            A new list containing only the dialogues whose **every** turn
            passes :meth:`is_valid`.
        """
        original_count = len(dialogues)
        filtered: List[Dict[str, Any]] = []
        for dlg in dialogues:
            turns = dlg.get("turns", [])
            if not turns:
                continue
            if all(self.is_valid(turn.get("text", "")) for turn in turns):
                filtered.append(dlg)
        logger.info(
            "Text filter: kept %d / %d dialogues (%.1f%%)",
            len(filtered),
            original_count,
            100.0 * len(filtered) / max(original_count, 1),
        )
        return filtered


# ============================================================================
# Stage 3 — Interaction Simulation
# ============================================================================


class InteractionSimulator:
    """Simulate user-assistant interaction dynamics (Section 4.1, Figure 2).

    Three key interaction patterns are modelled:

    1. **Normal turn-taking** — the user finishes speaking, then the assistant
       responds with a brief natural pause (~0.1–0.4 s).
    2. **User interruption** — the user begins speaking while the assistant is
       still talking; the assistant's audio is truncated at the overlap point.
    3. **Assistant waiting** — the assistant has finished its turn but the user
       has not started yet; a random silence gap is inserted.

    Args:
        interrupt_prob:      Probability that a given turn pair is treated as a
                             user-interrupt interaction.
        wait_prob:           Probability that a given turn pair includes an
                             assistant-wait silence.  (*Normal* is the
                             complement:  ``1 - interrupt_prob - wait_prob``.)
        wait_silence_range:  (min, max) seconds of silence for assistant-wait.
        normal_pause_range:  (min, max) seconds of the brief pause between
                             normal turns.
        interrupt_overlap:   (min, max) fraction of the **assistant** turn that
                             the user overlaps when interrupting.
        sample_rate:         Audio sample rate in Hz.
    """

    def __init__(
        self,
        interrupt_prob: float = 0.15,
        wait_prob: float = 0.15,
        wait_silence_range: Tuple[float, float] = (0.5, 2.0),
        normal_pause_range: Tuple[float, float] = (0.1, 0.4),
        interrupt_overlap: Tuple[float, float] = (0.10, 0.50),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.interrupt_prob = interrupt_prob
        self.wait_prob = wait_prob
        self.wait_silence_range = wait_silence_range
        self.normal_pause_range = normal_pause_range
        self.interrupt_overlap = interrupt_overlap
        self.sample_rate = sample_rate

        # Validate probabilities.
        total = interrupt_prob + wait_prob
        if total > 1.0:
            raise ValueError(
                f"interrupt_prob ({interrupt_prob}) + wait_prob ({wait_prob}) "
                f"must not exceed 1.0"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seconds_to_samples(self, seconds: float) -> int:
        """Convert a duration in seconds to the number of samples."""
        return int(round(seconds * self.sample_rate))

    def _choose_interaction_type(self) -> str:
        """Randomly select an interaction type for one turn pair."""
        r = random.random()
        if r < self.interrupt_prob:
            return "user_interrupt"
        elif r < self.interrupt_prob + self.wait_prob:
            return "assistant_wait"
        return "normal"

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate_interaction(
        self,
        user_audios: List[np.ndarray],
        assistant_audios: List[np.ndarray],
        texts: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Simulate a realistic two-channel interaction from per-turn clips.

        Args:
            user_audios:     List of 1-D ``float32`` arrays — one per user turn,
                             in chronological order.
            assistant_audios: Corresponding assistant turns (same length).
            texts:           ``[{"role": "user", "text": "..."}, ...]`` in order.

        Returns:
            A dict with keys:

            * ``user_channel``   — ``np.ndarray``  (full user channel)
            * ``assistant_channel`` — ``np.ndarray``  (full assistant channel)
            * ``turn_timestamps``  — list of per-turn metadata dicts
            * ``interaction_type`` — dominant interaction type string
        """
        if len(user_audios) != len(assistant_audios):
            raise ValueError(
                f"Mismatch: {len(user_audios)} user audios vs "
                f"{len(assistant_audios)} assistant audios."
            )
        n_pairs = len(user_audios)
        if n_pairs == 0:
            raise ValueError("At least one turn pair is required.")

        turn_timestamps: List[Dict[str, Any]] = []
        interaction_types_used: List[str] = []

        # We build the two channels incrementally.  ``cursor`` tracks the
        # current sample position on the global timeline.
        user_segments: List[Tuple[int, np.ndarray]] = []
        asst_segments: List[Tuple[int, np.ndarray]] = []
        cursor: int = 0  # samples

        for i in range(n_pairs):
            u_audio = user_audios[i].astype(np.float32)
            a_audio = assistant_audios[i].astype(np.float32)
            u_len = len(u_audio)
            a_len = len(a_audio)

            itype = self._choose_interaction_type()
            if itype == "user_interrupt" and i == n_pairs-1:
                itype = "normal"
            interaction_types_used.append(itype)

            # ----- User turn placement -----
            user_start = cursor
            user_segments.append((user_start, u_audio))
            user_end = user_start + u_len
            turn_timestamps.append(
                {
                    "role": "user",
                    "text": texts[2 * i]["text"] if 2 * i < len(texts) else "",
                    "start_time": user_start / self.sample_rate,
                    "end_time": user_end / self.sample_rate,
                }
            )

            # ----- Assistant turn placement (depends on interaction type) -----
            if itype == "normal":
                # Brief pause after user finishes, then assistant responds.
                pause_sec = random.uniform(*self.normal_pause_range)
                pause_samples = self._seconds_to_samples(pause_sec)
                asst_start = user_end + pause_samples
                asst_segments.append((asst_start, a_audio))
                asst_end = asst_start + a_len
                cursor = asst_end

            elif itype == "user_interrupt":
                # The assistant starts right after the user, but the *next*
                # user turn will overlap the tail of this assistant turn.
                # We first place the assistant normally with a brief pause,
                # then truncate once the next user turn begins.
                pause_sec = random.uniform(*self.normal_pause_range)
                pause_samples = self._seconds_to_samples(pause_sec)
                asst_start = user_end + pause_samples

                # Determine how much of the assistant turn will be heard
                # before the next user interrupts.  We pick a random fraction
                # of the assistant audio that plays before being cut off.
                overlap_frac = random.uniform(*self.interrupt_overlap)
                heard_samples = max(
                    int(a_len * (1.0 - overlap_frac)),
                    self._seconds_to_samples(0.3),  # at least 0.3 s audible
                )
                heard_samples = min(heard_samples, a_len)
                truncated_audio = a_audio[:heard_samples]

                # Apply a short fade-out (20 ms) so the truncation is smooth.
                fade_len = min(self._seconds_to_samples(0.02), heard_samples)
                if fade_len > 0:
                    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    truncated_audio[-fade_len:] *= fade

                asst_segments.append((asst_start, truncated_audio))
                asst_end = asst_start + heard_samples
                # The next user starts BEFORE the assistant stops. A short
                # reaction interval creates genuine simultaneous audio.
                reaction = min(self._seconds_to_samples(0.1), max(1, heard_samples//2))
                cursor = asst_end - reaction

            elif itype == "assistant_wait":
                # Brief pause, then assistant responds, then extra silence
                # before the next user turn (assistant is "waiting").
                pause_sec = random.uniform(*self.normal_pause_range)
                pause_samples = self._seconds_to_samples(pause_sec)
                asst_start = user_end + pause_samples
                asst_segments.append((asst_start, a_audio))
                asst_end = asst_start + a_len
                # Append a waiting silence.
                wait_sec = random.uniform(*self.wait_silence_range)
                wait_samples = self._seconds_to_samples(wait_sec)
                cursor = asst_end + wait_samples

            else:
                raise ValueError(f"Unknown interaction type: {itype}")

            turn_timestamps.append(
                {
                    "role": "assistant",
                    "text": texts[2 * i + 1]["text"] if 2 * i + 1 < len(texts) else "",
                    "start_time": asst_start / self.sample_rate,
                    "end_time": asst_end / self.sample_rate,
                    "interaction_type": itype,
                    "truncated": itype == "user_interrupt" and heard_samples < a_len,
                }
            )

        # --- Merge segments into continuous channels ---
        total_samples = max([cursor] + [start+len(seg) for start,seg in user_segments+asst_segments])
        if total_samples == 0:
            total_samples = 1  # avoid zero-length arrays
        user_channel = np.zeros(total_samples, dtype=np.float32)
        asst_channel = np.zeros(total_samples, dtype=np.float32)

        for start, seg in user_segments:
            end = start + len(seg)
            if end > total_samples:
                seg = seg[: total_samples - start]
                end = total_samples
            user_channel[start:end] += seg

        for start, seg in asst_segments:
            end = start + len(seg)
            if end > total_samples:
                seg = seg[: total_samples - start]
                end = total_samples
            asst_channel[start:end] += seg

        # Determine the "dominant" interaction type.
        from collections import Counter

        dominant_type = Counter(interaction_types_used).most_common(1)[0][0]

        return {
            "user_channel": user_channel,
            "assistant_channel": asst_channel,
            "turn_timestamps": turn_timestamps,
            "interaction_type": dominant_type,
        }


# ============================================================================
# Stage 4 — Noise Augmentation
# ============================================================================


class NoiseAugmentor:
    """Add background noise to the user channel (Section 4.1, Stage 4).

    Noise clips are loaded lazily from a directory of WAV files (e.g. the MUSAN
    ``noise`` split).

    Args:
        noise_dir:   Path to a directory containing ``.wav`` noise files.  May
                     contain subdirectories.  If ``None`` or non-existent, the
                     augmentor becomes a no-op.
        snr_min:     Minimum target SNR in dB.
        snr_max:     Maximum target SNR in dB.
        sample_rate: Expected sample rate; noise files are resampled if needed.
    """

    _SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}

    def __init__(
        self,
        noise_dir: Optional[str] = None,
        snr_min: float = 15.0,
        snr_max: float = 30.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.sample_rate = sample_rate

        self.noise_files: List[str] = []
        if noise_dir is not None and os.path.isdir(noise_dir):
            for root, _dirs, files in os.walk(noise_dir):
                for fname in files:
                    if Path(fname).suffix.lower() in self._SUPPORTED_EXTENSIONS:
                        self.noise_files.append(os.path.join(root, fname))
            self.noise_files.sort()  # deterministic ordering
            logger.info("NoiseAugmentor: found %d noise files.", len(self.noise_files))
        else:
            logger.warning(
                "NoiseAugmentor: noise_dir=%s is invalid or missing — "
                "noise augmentation is disabled.",
                noise_dir,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rms(signal: np.ndarray) -> float:
        """Root-mean-square of *signal*, with a floor to avoid log(0)."""
        return float(max(np.sqrt(np.mean(signal.astype(np.float64) ** 2)), 1e-10))

    def _load_noise(self, path: str, target_length: int) -> np.ndarray:
        """Load a noise file and tile/truncate it to *target_length* samples.

        If the file's sample rate differs from ``self.sample_rate`` it is
        resampled using linear interpolation (fast, no torchaudio dependency
        at runtime).
        """
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        # If stereo, take the first channel.
        if data.ndim > 1:
            data = data[:, 0]

        # Resample if necessary (simple linear interpolation).
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, new_len)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)

        # Tile to cover *target_length*.
        if len(data) < target_length:
            repeats = math.ceil(target_length / len(data))
            data = np.tile(data, repeats)

        # Random offset crop.
        max_start = len(data) - target_length
        start = random.randint(0, max(max_start, 0))
        return data[start : start + target_length]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_noise(
        self,
        clean_audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Add random noise at a random SNR between *snr_min* and *snr_max*.

        The noise is scaled so that:

        .. math::

            \\text{SNR}_{\\text{dB}} = 20 \\cdot \\log_{10}
            \\left(\\frac{\\text{RMS}_{\\text{signal}}}
            {\\text{RMS}_{\\text{noise}}}\\right)

        Args:
            clean_audio: 1-D float32 array of the clean speech signal.
            sample_rate: (unused) Kept for API symmetry — the augmentor uses
                         its own configured rate.

        Returns:
            Noisy audio as a new ``float32`` array.
        """
        if len(self.noise_files) == 0:
            logger.debug("No noise files available; returning clean audio.")
            return clean_audio.copy()

        target_length = len(clean_audio)
        noise_path = random.choice(self.noise_files)
        noise = self._load_noise(noise_path, target_length)

        target_snr_db = random.uniform(self.snr_min, self.snr_max)

        # Compute required noise gain.
        signal_rms = self._rms(clean_audio)
        noise_rms = self._rms(noise)

        # desired_noise_rms = signal_rms / 10^(snr_db / 20)
        desired_noise_rms = signal_rms / (10.0 ** (target_snr_db / 20.0))
        gain = desired_noise_rms / noise_rms

        noisy = clean_audio + gain * noise
        return noisy.astype(np.float32)


# ============================================================================
# Pretrained TTS and dialogue simulation
# ============================================================================


class DialogueSimulationPipeline:
    """Orchestrate the complete dialogue data simulation pipeline.

    Stages executed in order:

    1. Load & filter text dialogues (:class:`TextDialogueFilter`).
    2. Synthesize per-turn audio with pretrained CosyVoice.
    3. Simulate interaction dynamics (:class:`InteractionSimulator`).
    4. Add noise to the user channel (:class:`NoiseAugmentor`).
    5. Write two-channel audio + manifest.

    Args:
        config: A :class:`PipelineConfig` instance.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.text_filter = TextDialogueFilter(max_words=config.max_text_words)
        self.simulator = InteractionSimulator(
            interrupt_prob=config.interrupt_prob,
            wait_silence_range=config.wait_silence_range,
            sample_rate=config.sample_rate,
        )
        self.noise_augmentor = NoiseAugmentor(
            noise_dir=config.noise_data_dir,
            snr_min=config.snr_min,
            snr_max=config.snr_max,
            sample_rate=config.sample_rate,
        )
        if not config.codec_dir:
            raise ValueError('A pretrained CosyVoice v1 --codec_dir is required; noise is not speech data')
        if config.sample_rate != 16000:
            raise ValueError('Simulation audio must use 16000 Hz for the codec frontend')
        from omniflatten.speech_tokenizer import CosyVoiceCodec
        self.tts = CosyVoiceCodec(config.codec_dir, config.assistant_speaker_id)
        self.tts._speaker()  # Validate fixed assistant voice before writing data.
        self.output_dir = Path(config.dialogue_data_dir)

    # ------------------------------------------------------------------
    # TTS helper (thin wrapper for easy subclassing)
    # ------------------------------------------------------------------

    def synthesize_speech(
        self,
        text: str,
        speaker_id: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesize speech from *text* using the configured TTS engine.

        Override this method (or swap ``self.tts``) to integrate CosyVoice or
        another real TTS system.

        Args:
            text:       Text to synthesize.
            speaker_id: Optional speaker identifier.

        Returns:
            1-D float32 waveform at ``self.config.sample_rate``.
        """
        return self.tts.synthesize(text, speaker_id=speaker_id)

    # ------------------------------------------------------------------
    # Single-dialogue processing
    # ------------------------------------------------------------------

    def process_dialogue(
        self,
        dialogue: Dict[str, Any],
        user_speaker_id: Optional[str] = None,
        assistant_speaker_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process a single text dialogue into a simulated full-duplex record.

        The dialogue dict must contain a ``"turns"`` list of
        ``{"role": "user"|"assistant", "text": "..."}`` dicts, strictly
        alternating and starting with ``"user"``.

        Args:
            dialogue:             Input dialogue dict.
            user_speaker_id:      TTS speaker ID for the user.
            assistant_speaker_id: TTS speaker ID for the assistant.

        Returns:
            A manifest record dict (see module docstring), or ``None`` if the
            dialogue was filtered out or could not be processed.
        """
        turns: List[Dict[str, str]] = dialogue.get("turns", [])
        if not turns:
            logger.debug("Skipping dialogue with no turns.")
            return None

        # Validate alternating roles starting with user.
        expected_roles = ["user", "assistant"] * ((len(turns) + 1) // 2)
        for i, turn in enumerate(turns):
            if turn.get("role") != expected_roles[i]:
                logger.debug(
                    "Skipping dialogue — unexpected role '%s' at turn %d.",
                    turn.get("role"),
                    i,
                )
                return None

        # Ensure even number of turns (complete pairs).
        if len(turns) % 2 != 0:
            turns = turns[:-1]
        if len(turns) < 2:
            return None

        # --- Stage 2: filter ---
        for turn in turns:
            if not self.text_filter.is_valid(turn["text"]):
                return None

        user_speaker_id = user_speaker_id or random.choice(list(self.tts.cosyvoice.frontend.spk2info))
        assistant_speaker_id = assistant_speaker_id or self.config.assistant_speaker_id
        # --- Stage 3a: TTS ---
        user_audios: List[np.ndarray] = []
        assistant_audios: List[np.ndarray] = []
        for i in range(0, len(turns), 2):
            u_wav = self.synthesize_speech(turns[i]["text"], speaker_id=user_speaker_id)
            a_wav = self.synthesize_speech(
                turns[i + 1]["text"], speaker_id=assistant_speaker_id
            )
            user_audios.append(u_wav)
            assistant_audios.append(a_wav)

        # Keep original uncut utterances for the preceding half-duplex stage.
        dialogue_id = str(dialogue.get('dialogue_id', uuid.uuid4().hex[:12]))
        # IDs are metadata, not filesystem paths.
        safe_id = uuid.uuid5(uuid.NAMESPACE_URL, dialogue_id).hex
        turn_dir = self.output_dir / 'turns' / safe_id
        turn_dir.mkdir(parents=True, exist_ok=True)
        half_duplex_turns = []
        for index, turn in enumerate(turns):
            audio = user_audios[index//2] if index%2==0 else assistant_audios[index//2]
            path = turn_dir / f'{index:04d}.wav'
            sf.write(path, audio, self.config.sample_rate)
            half_duplex_turns.append({'role':turn['role'], 'text':turn['text'],
                                      'audio_path':str(path.resolve())})

        # --- Stage 3b: interaction simulation ---
        sim_result = self.simulator.simulate_interaction(
            user_audios, assistant_audios, turns
        )

        # --- Stage 4: noise augmentation (user channel only) ---
        user_channel: np.ndarray = self.noise_augmentor.add_noise(
            sim_result["user_channel"], self.config.sample_rate
        )
        assistant_channel: np.ndarray = sim_result["assistant_channel"]

        # --- Stage 5: save audio ---
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        user_audio_path = audio_dir / f"{safe_id}_user.wav"
        asst_audio_path = audio_dir / f"{safe_id}_assistant.wav"

        sf.write(str(user_audio_path), user_channel, self.config.sample_rate)
        sf.write(str(asst_audio_path), assistant_channel, self.config.sample_rate)

        total_duration = max(len(user_channel), len(assistant_channel)) / self.config.sample_rate

        manifest_record: Dict[str, Any] = {
            "dialogue_id": str(dialogue_id),
            "user_audio_path": str(user_audio_path.resolve()),
            "assistant_audio_path": str(asst_audio_path.resolve()),
            "turns": sim_result["turn_timestamps"],
            "half_duplex_turns": half_duplex_turns,
            "interaction_type": sim_result["interaction_type"],
            "total_duration": round(total_duration, 4),
        }
        return manifest_record

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def _load_dialogues(self, input_path: str) -> List[Dict[str, Any]]:
        """Load dialogues from a JSONL file.

        Each line must be a valid JSON object with at least a ``"turns"`` key.
        """
        dialogues: List[Dict[str, Any]] = []
        with open(input_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    dialogues.append(obj)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping line %d: %s", line_no, exc)
        logger.info("Loaded %d dialogues from %s", len(dialogues), input_path)
        return dialogues

    def run(self, input_dialogues_path: str, output_manifest_path: str, num_workers: int = 1):
        if num_workers != 1:
            raise ValueError('Use one worker for the shared pretrained codec')
        dialogues = self.text_filter.filter_dialogues(self._load_dialogues(input_dialogues_path))
        path = Path(output_manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream, (path.parent/'half_duplex_manifest.jsonl').open('w', encoding='utf-8') as half:
            for dialogue in dialogues:
                record = self.process_dialogue(dialogue)
                if record is not None:
                    half.write(json.dumps({'dialogue_id':record['dialogue_id'],
                                           'turns':record.pop('half_duplex_turns')},ensure_ascii=False)+'\n')
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """Command-line entry point for the dialogue simulation pipeline."""
    parser = argparse.ArgumentParser(
        description="Simulate full-duplex dialogue data (OmniFlatten §4.1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input text dialogues JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for pipeline outputs (audio + manifest).",
    )
    parser.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        help="Path to MUSAN noise directory (optional).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Use 1 for pretrained codec synthesis.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate in Hz.",
    )
    parser.add_argument(
        "--max_text_words",
        type=int,
        default=200,
        help="Maximum word count per utterance.",
    )
    parser.add_argument(
        "--interrupt_prob",
        type=float,
        default=0.15,
        help="Probability of user-interrupt interaction per turn pair.",
    )
    parser.add_argument(
        "--snr_min",
        type=float,
        default=15.0,
        help="Minimum SNR (dB) for noise augmentation.",
    )
    parser.add_argument(
        "--snr_max",
        type=float,
        default=30.0,
        help="Maximum SNR (dB) for noise augmentation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument("--codec_dir", required=True)
    parser.add_argument("--assistant_speaker_id", required=True)
    args = parser.parse_args()

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Config ---
    config = PipelineConfig(
        max_text_words=args.max_text_words,
        noise_data_dir=args.noise_dir,
        dialogue_data_dir=args.output_dir,
        sample_rate=args.sample_rate,
        interrupt_prob=args.interrupt_prob,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        num_workers=args.num_workers,
        codec_dir=args.codec_dir,
        assistant_speaker_id=args.assistant_speaker_id,
    )

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")

    pipeline = DialogueSimulationPipeline(config)
    pipeline.run(
        input_dialogues_path=args.input,
        output_manifest_path=manifest_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
