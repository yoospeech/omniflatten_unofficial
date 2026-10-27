"""Documented experiment defaults; train.py exposes its effective settings via CLI.

Only values explicitly reported in §3.3.2 and §4.2 are called paper defaults.
Warmup fraction, microbatch size and clipping are implementation choices.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import yaml

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    codebook_size: int = 4096
    speech_token_rate: int = 50
    codec_dir: str | None = None

@dataclass
class ModelConfig:
    backbone_model: str = 'Qwen/Qwen2-0.5B'
    speech_vocab_size: int = 4096

@dataclass
class TrainingConfig:
    stage1_max_seq_len: int = 1024
    stage2_max_seq_len: int = 8192
    stage3_max_seq_len: int = 8192
    stage1_epochs: int = 5
    stage2_epochs: int = 5
    stage3_epochs: int = 5
    stage1_lr: float = 2e-5
    stage2_lr: float = 2e-5
    stage3_lr: float = 2e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    speech_chunk_size: int = 10
    text_chunk_size: int = 2
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    seed: int = 42

@dataclass
class DataConfig:
    noise_data_dir: str | None = None
    snr_min: float = 15.0
    snr_max: float = 30.0
    max_text_words: int = 200

@dataclass
class OmniFlattenConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_defaults(cls):
        return cls()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value)-{'audio','model','training','data'}
        if unknown:
            raise ValueError(f'Unknown configuration keys: {unknown}')
        return cls(**{name:typ(**value.get(name,{})) for name,typ in
                      [('audio',AudioConfig),('model',ModelConfig),('training',TrainingConfig),('data',DataConfig)]})

    @classmethod
    def from_yaml(cls,path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

def get_default_config():
    return OmniFlattenConfig()

def load_config(path):
    path=Path(path)
    raw=json.loads(path.read_text()) if path.suffix=='.json' else yaml.safe_load(path.read_text())
    return OmniFlattenConfig.from_dict(raw)

def save_config(config,path):
    path=Path(path)
    path.write_text(json.dumps(config.to_dict(),indent=2) if path.suffix=='.json' else yaml.safe_dump(config.to_dict()))
