"""Pretrained CosyVoice v1 adapter (4096 codes), paper §3.1.

Install the upstream FunAudioLLM/CosyVoice repository and its dependencies
separately. No randomly initialized codec or synthetic-noise fallback is used.
CosyVoice2/3 have different token inventories and are not interchangeable.
"""
from pathlib import Path
import numpy as np
import torch
from omniflatten.tokens import normalize_codes

class CosyVoiceSpeechTokenizer:
    """Training-only v1 ONNX encoder; does not load TTS or require a voice ID."""
    codebook_size = 4096
    sample_rate = 16000
    token_rate = 50

    def __init__(self, model_dir, num_threads=2, device='cpu'):
        import onnxruntime as ort
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device.startswith('cuda:') and device[5:].isdigit():
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA ONNX Runtime is unavailable; install onnxruntime-gpu and check PYTHONPATH')
            device_id = int(device[5:])
            if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
                raise RuntimeError(f'CUDA device {device_id} is not available')
            providers = [('CUDAExecutionProvider', {'device_id': device_id,
                         'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider']
        else:
            raise ValueError("device must be 'cpu' or 'cuda:<index>'")
        path = Path(model_dir) / 'speech_tokenizer_v1.onnx'
        if not path.is_file():
            raise FileNotFoundError(f'Missing CosyVoice v1 tokenizer: {path}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device != 'cpu' and 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('CUDA provider failed to initialize; refusing to silently run on CPU')
        self.device = device

    @torch.inference_mode()
    def encode(self, waveform):
        import whisper
        waveform = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1).cpu()
        codes = []
        for window in waveform.split(30 * self.sample_rate, dim=1):
            if not window.numel():
                continue
            if window.shape[1] <= 200:
                window = torch.nn.functional.pad(window, (0, 201-window.shape[1]))
            mel = whisper.log_mel_spectrogram(window, n_mels=128)
            inputs = self.session.get_inputs()
            output = self.session.run(None, {inputs[0].name: mel.numpy(),
                                            inputs[1].name: np.array([mel.shape[-1]], dtype=np.int32)})
            codes.extend(normalize_codes(np.asarray(output[0]).reshape(-1), self.codebook_size))
        return torch.tensor(codes, dtype=torch.long)


class CosyVoiceCodec:
    codebook_size = 4096
    sample_rate = 16000
    token_rate = 50

    def __init__(self, model_dir, speaker_id=None):
        model_dir = Path(model_dir)
        required = ['cosyvoice.yaml', 'speech_tokenizer_v1.onnx', 'flow.pt', 'hift.pt', 'llm.pt']
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f'CosyVoice v1 checkpoint required at {model_dir}; missing {missing}')
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
        except ImportError as exc:
            raise ImportError('Install FunAudioLLM/CosyVoice and add its repo and third_party/Matcha-TTS to PYTHONPATH') from exc
        self.cosyvoice = CosyVoice(str(model_dir), load_jit=False, load_trt=False, fp16=False)
        rate = float(self.cosyvoice.model.flow.input_frame_rate)
        if rate != self.token_rate:
            raise ValueError(f'Expected 50 Hz CosyVoice v1 tokens; checkpoint has {rate} Hz')
        self.output_sample_rate = self.cosyvoice.sample_rate
        self.speaker_id = speaker_id

    @torch.inference_mode()
    def encode(self, waveform):
        """One mono 16-kHz waveform -> flat CPU codebook IDs. Offline encoding.

        Long recordings are partitioned into <=30 s codec windows. Streaming
        microphone frontends must manage their own codec context/lookahead.
        """
        import whisper
        waveform = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1).cpu()
        if waveform.shape[1] == 0:
            return torch.empty(0, dtype=torch.long)
        session = self.cosyvoice.frontend.speech_tokenizer_session
        codes = []
        for window in waveform.split(30 * self.sample_rate, dim=1):
            # Reflect-padding in Whisper STFT requires more than 200 samples.
            if window.shape[1] <= 200:
                window = torch.nn.functional.pad(window, (0, 201-window.shape[1]))
            mel = whisper.log_mel_spectrogram(window, n_mels=128)
            inputs = session.get_inputs()
            output = session.run(None, {inputs[0].name: mel.numpy(),
                                       inputs[1].name: np.array([mel.shape[-1]], dtype=np.int32)})
            codes.extend(normalize_codes(np.asarray(output[0]).reshape(-1), self.codebook_size))
        return torch.tensor(codes, dtype=torch.long)

    def _speaker(self, speaker_id=None):
        speaker_id = speaker_id or self.speaker_id
        if speaker_id not in self.cosyvoice.frontend.spk2info:
            raise ValueError('Set a speaker_id from cosyvoice.list_available_spks() (SFT checkpoint)')
        return speaker_id

    @torch.inference_mode()
    def decode(self, token_ids):
        """Decode codebook IDs via the checkpoint's flow model and vocoder."""
        codes = normalize_codes(token_ids, self.codebook_size)
        if not codes:
            return torch.empty(0)
        speaker = self._speaker()
        embedding = self.cosyvoice.frontend.spk2info[speaker]['embedding']
        packets = self.cosyvoice.model.tts(
            source_speech_token=torch.tensor([codes], dtype=torch.int32),
            flow_embedding=embedding, stream=False)
        return torch.cat([p['tts_speech'].detach().cpu().reshape(-1) for p in packets])

    def synthesize(self, text, speaker_id=None):
        """Real SFT TTS for dialogue simulation, returned at 16 kHz."""
        import torchaudio
        speaker = self._speaker(speaker_id)
        packets = self.cosyvoice.inference_sft(text, speaker, stream=False)
        waveform = torch.cat([p['tts_speech'].detach().cpu().reshape(-1) for p in packets])
        waveform = torchaudio.functional.resample(waveform, self.output_sample_rate, self.sample_rate)
        return waveform.numpy()

# Both directions share the same pretrained codec and speaker conditioning.
SpeechTokenizer = CosyVoiceSpeechTokenizer
SpeechDetokenizer = CosyVoiceCodec
