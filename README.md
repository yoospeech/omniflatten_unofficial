# OmniFlatten — paper-aligned research implementation

[OmniFlatten 논문](https://arxiv.org/html/2410.17799v2)의 데이터 형식과 점진적 학습을 구현합니다. Qwen2의 Transformer 구조는 변경하지 않고 텍스트·음성 어휘를 확장합니다. 공식 모델 가중치나 논문 성능을 재현한 결과를 제공하는 저장소는 아닙니다.

## 논문과의 대응

| 단계 | 입력 시퀀스 | 최대 길이 |
|---|---|---:|
| `1` 모달리티 정렬 | `[ASR][SOS]음성[EOS][SOT]텍스트[EOT]` 또는 `[TTS][SOT]텍스트[EOT][SOS]음성[EOS]` | 1024 |
| `2` Half-duplex | 사용자 음성 → 사용자 텍스트 → 답변 텍스트 → 답변 음성, 턴별 반복 | 8192 |
| `3a` Full-duplex | 사용자 음성 10 → 답변 텍스트 2 → 답변 음성 10, 청크별 반복 | 8192 |
| `3b` Full-duplex | 사용자 음성 10 → 답변 음성 10, 청크별 반복 | 8192 |

- 기본 backbone: `Qwen/Qwen2-0.5B`. 대화 단계에서는 사용자 채널과 배치 padding을 loss에서 제외합니다.
- 음성 코덱: **CosyVoice v1, 4096-code vocabulary**. 사전학습된 semantic tokenizer, flow model, vocoder를 사용합니다. `CosyVoice2`라는 이름이 논문 PDF의 각주 번호와 붙어 보일 수 있지만, 새 세대의 CosyVoice2/3 체크포인트를 그대로 사용하면 안 됩니다.
- 실제 체크포인트의 `input_frame_rate=50`을 검사합니다. 이 경우 음성 10토큰은 200ms입니다. 음성 파일 입력은 16 kHz, 출력은 코덱의 출력 sample rate를 따릅니다.
- AdamW: lr `2e-5`, betas `(0.9, 0.95)`, weight decay `0.1`, warmup + cosine, 5 epochs. validation loss로 best checkpoint를 선택합니다.
- 논문의 batch 100M tokens는 기본 실행 설정에서 재현하지 않습니다. 기본값은 microbatch 1, gradient accumulation 8입니다. warmup 비율 0.05, gradient clipping 1.0, Stage 1 source-token loss masking은 이 구현의 선택입니다.
- 음성 ID offset은 `model.config.vocab_size`에서 추정하지 않습니다. 실제 tokenizer에 등록된 ID와 레이아웃을 checkpoint에 함께 저장합니다.

## 실행 환경

지정된 Python과 프로젝트 내부 호환 패키지를 사용하는 실행 래퍼입니다.

```bash
./scripts/python.sh -m unittest discover -s tests -v
./scripts/python.sh train.py --help
```

래퍼는 `/data/util/anaconda3/envs/duplexomni/bin/python`을 사용하고 `.runtime`을 `PYTHONPATH`에 추가합니다. 기존 환경에는 Transformers 4.57.6과 Hugging Face Hub 2.3.0의 버전 충돌이 있어, 환경 자체를 변경하지 않고 아래 호환 버전을 `.runtime`에 설치했습니다.

```bash
/data/util/anaconda3/envs/duplexomni/bin/python -m pip install \
  --target .runtime --no-deps \
  huggingface-hub==0.36.0 datasets==4.4.1 accelerate==1.12.0 fsspec==2025.10.0
```

새 환경에서는 `requirements.txt`를 사용하세요. PyTorch와 torchaudio는 동일 버전·동일 CUDA 계열로 설치해야 합니다.

## 실제 음성 코덱

[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)의 코드·의존성을 별도로 설치하고 `CosyVoice-300M-SFT` 계열의 **v1 로컬 체크포인트**를 준비합니다. 저장소와 `third_party/Matcha-TTS`를 `PYTHONPATH`에 추가해야 합니다.

```python
from omniflatten.speech_tokenizer import CosyVoiceCodec

codec = CosyVoiceCodec('/path/to/CosyVoice-300M-SFT')
print(codec.cosyvoice.list_available_spks())
# 디코딩/합성할 때는 위 목록의 고정 assistant 화자를 선택합니다.
codec.speaker_id = '실제 화자 ID'
```

필요 파일은 `cosyvoice.yaml`, `speech_tokenizer_v1.onnx`, `llm.pt`, `flow.pt`, `hift.pt`이며, upstream frontend가 요구하는 `campplus.onnx`, `spk2info.pt` 등도 함께 준비해야 합니다. 무작위 인코더·잡음 TTS 대체 경로는 없습니다. 코덱은 전체 학습과 별도로 고정해 사용합니다.

## 로컬 SUMM-RE 사용

로컬 캐시의 데이터는 `meeting_id`, `speaker_id`, `audio`, `segments`, `transcript`를 포함합니다. 각 segment에 `start`, `end`, `transcript`, `words`가 있습니다. 긴 회의 오디오를 발화 구간으로 잘라 **Stage 1의 음성↔전사 쌍**으로 사용합니다. 회의 참가자를 임의로 사용자·assistant로 바꾸지는 않습니다.

```bash
./scripts/python.sh scripts/prepare_summ_re.py \
  --arrow_dir /home/speech/.cache/huggingface/datasets/linagora___summ-re/default/0.0.0/47cb4ec3437ef5497b6092b8c6d1e0b5b1d86dc0 \
  --output_dir prepared_data/summ_re \
  --max_duration 12
```

기존 `train/dev/test` 분할을 `train/val/test`로 유지하며, 회의 ID가 분할 간 겹치면 오류를 냅니다. 위 명령은 원본 캐시를 수정하거나 데이터를 재다운로드하지 않습니다. `--max_segments 2`로 각 split의 2개 구간만 빠르게 확인할 수 있습니다. 실제 캐시에서 추출한 예시는 `prepared_data/summ_re_smoke/`에 있습니다.

SUMM-RE는 논문의 영어·중국어 데이터 구성과 다르므로 동일한 성능을 기대할 근거는 없습니다. 발화가 길거나 토큰 예산을 넘으면 구간을 더 잘라야 합니다. 12초 제한은 전처리 기본값이지 논문 설정이 아닙니다.

## 학습 데이터 형식

각 단계별 디렉터리에 `train.jsonl`, `val.jsonl`을 준비합니다. 경로는 해당 manifest 디렉터리 기준이며, 절대 경로도 지원합니다.

Stage 1:

```json
{"audio_path":"train/00000000.wav","text":"발화 전사"}
```

Stage 2:

```json
{"turns":[{"role":"user","audio_path":"u.wav","text":"질문"},{"role":"assistant","audio_path":"a.wav","text":"답변"}]}
```

Stage 3a/3b: 두 오디오는 동일한 시간축을 갖는 사용자·assistant 채널입니다. 무음 구간을 없애거나 발화를 이어 붙이면 안 됩니다.

```json
{"user_audio_path":"user.wav","assistant_audio_path":"assistant.wav","turns":[{"role":"user","start_time":0.0,"end_time":1.0,"text":"질문"},{"role":"assistant","start_time":1.2,"end_time":2.4,"text":"답변"}]}
```

assistant의 timestamp로 출력 음성의 비발화 구간을 `[SILENT_SPEECH]`로 채웁니다. 텍스트는 발화가 시작되는 청크부터 2토큰씩 배치하고 나머지는 `[SILENT_TEXT]`로 채웁니다. 이는 논문의 느슨한 청크 정렬을 구현한 선택이며, 단어별 강제 정렬을 재현했다고 주장하지 않습니다. 텍스트가 해당 발화의 청크 용량을 넘으면 명시적으로 오류를 냅니다.

중간에 잘린 답변은 `truncated: true`로 기록합니다. **Stage 3a에서는 실제 들린 부분의 `spoken_text`가 추가로 필요**합니다. 전체 답변 전사를 잘린 오디오에 붙이지 않습니다. Stage 3b는 텍스트를 쓰지 않으므로 이 보정이 필요하지 않습니다.

오디오 대신 **raw codebook ID**를 저장할 수도 있습니다. `audio_path` → `speech_tokens`, `user_audio_path` → `user_speech_tokens`, `assistant_audio_path` → `assistant_speech_tokens`입니다. 값은 `0..4095`이며, unified vocabulary offset을 미리 더하지 마세요.

```bash
./scripts/python.sh scripts/precompute_speech.py \
  --input prepared_data/summ_re/train.jsonl \
  --output prepared_data/summ_re_tokens/train.jsonl \
  --codec_dir /path/to/CosyVoice-300M-SFT
```

`val.jsonl`도 같은 방식으로 변환합니다. 온라인 음성 추출은 `--codec_dir`과 `--num_workers 0`을 사용하고, 대규모 학습에서는 토큰을 미리 계산하는 편이 적합합니다.

## 학습 실행

```bash
./scripts/python.sh train.py --stage 1 --data_dir prepared_data/stage1 --output_dir outputs
./scripts/python.sh train.py --stage 2 --data_dir prepared_data/stage2 \
  --checkpoint outputs/stage1/best_model --output_dir outputs
./scripts/python.sh train.py --stage 3a --data_dir prepared_data/stage3a \
  --checkpoint outputs/stage2/best_model --output_dir outputs
./scripts/python.sh train.py --stage 3b --data_dir prepared_data/stage3b \
  --checkpoint outputs/stage3a/best_model --output_dir outputs
```

위 예시는 미리 계산한 speech tokens를 사용합니다. 오디오 manifest에는 `--codec_dir`을 추가합니다. GPU 환경에서는 `--bf16` 또는 `--fp16`을 선택할 수 있습니다. 여러 GPU는 같은 Python 환경의 `accelerate launch`를 사용하세요.

`--checkpoint`는 이전 단계의 모델 가중치를 불러오고 optimizer를 새로 시작합니다. `--resume_from_checkpoint`는 같은 단계의 optimizer·scheduler·RNG·epoch·batch 위치를 복구합니다. Resume에는 기존 batch/epoch/seed/process 수 및 optimizer 설정을 유지해야 합니다. 체크포인트가 없거나 형식이 다르면 무작위 가중치로 조용히 진행하지 않습니다.

```bash
./scripts/python.sh train.py --stage 1 --data_dir prepared_data/stage1 \
  --resume_from_checkpoint outputs/stage1/last_model
```

`configs/config.py`는 기본값을 읽고 저장하는 참고용 dataclass 모듈입니다. 실행 설정은 `train.py`의 CLI에서 정하며, 지원하지 않는 `--config` 옵션은 문서에서 제거했습니다.

## 합성 대화

```bash
./scripts/python.sh -m omniflatten.data.simulate_dialogue \
  --input dialogues.jsonl --output_dir prepared_data/simulated \
  --codec_dir /path/to/CosyVoice-300M-SFT \
  --assistant_speaker_id '실제 화자 ID' --num_workers 1 \
  --noise_dir /path/to/musan/noise
```

텍스트 대화 필터링, 실제 TTS, 끼어들기·대기 배치, 사용자 채널 SNR 15–30 dB 잡음 추가를 수행합니다. assistant는 고정 화자, 사용자는 SFT 화자 목록에서 샘플링합니다. 논문의 LibriSpeech/3D-Speaker 참조 음색 샘플링과는 차이가 있습니다. 끼어들기에서는 사용자가 말하기 시작한 뒤 짧은 반응 구간을 두고 assistant를 중단하여 실제 중첩을 만듭니다. 시뮬레이션 타이밍 분포는 구현 선택입니다.

생성된 `manifest.jsonl`은 full-duplex용이며, 원본의 잘리지 않은 발화는 `half_duplex_manifest.jsonl`에 별도로 저장되어 Stage 2에 사용할 수 있습니다. 같은 dialogue ID를 같은 split에 넣는 도구로 단계별 98/1/1 분할을 준비합니다.

```bash
./scripts/python.sh scripts/split_manifest.py \
  --input prepared_data/simulated/half_duplex_manifest.jsonl --output_dir prepared_data/stage2
./scripts/python.sh scripts/split_manifest.py \
  --input prepared_data/simulated/manifest.jsonl --output_dir prepared_data/stage3a
```

분할은 dialogue ID와 seed의 해시를 사용하므로 두 단계 사이에서도 동일한 대화가 다른 split으로 넘어가지 않습니다. 비율은 근사치이며, 아주 작은 데이터에서는 val/test가 비어 있을 수 있습니다. Stage 3a의 잘린 답변 전사는 앞서 설명한 보정이 필요합니다. Stage 3b는 같은 full-duplex manifest를 텍스트 없이 읽습니다.

## 추론

```python
from omniflatten.model import OmniFlattenModel
from omniflatten.inference import OmniFlattenInference, InferenceConfig

model, tokenizer = OmniFlattenModel.from_pretrained('outputs/stage3b/best_model')
engine = OmniFlattenInference(model, codec, codec, tokenizer,
                             InferenceConfig(mode='3b', device='cpu'))

# 각 입력은 raw codebook ID 최대 10개. 출력은 다음 입력을 읽기 전에 yield합니다.
for packet in engine.stream_generate(user_codebook_chunks, decode_audio=True):
    audio = packet['audio']
```

`3a`는 텍스트 2 + 음성 10, `3b`는 음성 10을 정확히 생성합니다. 모달리티별 어휘를 제한하고 마지막 생성 토큰까지 KV-cache에 반영합니다. 사용자 iterator가 끝나면 제한된 수의 무음 입력 청크를 추가해 답변이 이어질 수 있게 합니다. 출력 silence 토큰을 삭제하지 않고 오디오 시간축에 보존합니다.

`full_duplex_generate(path)`는 **파일 전체를 읽는 오프라인 평가 래퍼**입니다. 마이크 입력 장치, 실시간 음성 토크나이저의 lookahead 관리, 연속 vocoder overlap/cache, 오디오 장치 출력은 별도 통합이 필요합니다. `decode_audio=True`의 현재 청크별 독립 디코딩은 경계 음질·지연을 보장하지 않습니다.

## 검증 범위

테스트는 작은 무작위 Qwen2를 사용해 다운로드 없이 실행합니다. 토큰 ID/체크포인트 왕복, 네 단계의 데이터 형식, 사용자·padding loss masking, cached/full-context logits 일치, 스트림 순서, 침묵 보존, 실제 중첩 시뮬레이션, 단계 간 학습 전달과 중간 checkpoint resume를 확인합니다.

실제 CosyVoice 가중치로의 합성·복원, Qwen2-0.5B 전체 학습, WER/CER, turn-taking, 음질, GPU 지연은 별도의 실험이 필요합니다. 논문의 비공개 데이터와 전체 학습 규모를 재현하지 않았습니다.

## SUMM-RE + 실제 Qwen2-0.5B 검증 (2026-10-27)

실행 Python은 `/data/util/anaconda3/envs/duplexomni/bin/python`입니다. **RTX 3070 8GB × 2**에서 DDP, microbatch 1/GPU, gradient checkpointing으로 각 단계를 1 epoch씩 실행했습니다. 메모리를 줄이기 위해 parameter·gradient·Adam state를 BF16으로 유지했으며 FP32 master weights는 사용하지 않았습니다. 이는 소규모 실행 검증용 설정입니다.

`datasets.load_dataset("linagora/SUMM-RE", "example")`의 `example` 설정은 로컬 캐시에 없었습니다. 실제로 저장된 `default` 설정을 사용했고, 원본 Arrow 파일을 프로젝트의 writable cache에 링크하여 다음 호출로 읽었습니다.

```python
import datasets

ds = datasets.load_dataset(
    'linagora/SUMM-RE', 'default', cache_dir='.validation/hf_cache'
)
# TorchCodec 없이 오디오 bytes/path를 읽고 SoundFile로 디코딩합니다.
sample = ds['train'].cast_column('audio', datasets.Audio(decode=False))[0]
```

캐시에는 train 684개, dev 130개, test 124개의 화자별 회의 녹음이 있습니다. 검증 도구는 split마다 서로 다른 회의에서 실제 중첩된 두 화자의 짧은 발화 쌍 2개를 선택했습니다. Stage 1은 이 발화들의 ASR/TTS 쌍, Stage 2는 턴별 포맷, Stage 3a/3b는 원래 중첩 시각을 보존한 두 채널 포맷입니다. 실제 `pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx`로 토큰을 추출했습니다.

**회의 화자를 user/assistant로 매핑한 실행 검증용 데이터입니다. 논문의 합성 assistant 대화 데이터나 대화 품질 평가를 대체하지 않습니다.** 각 예시는 약 0.56–0.76초로 매우 짧습니다. 긴 컨텍스트 학습, 수렴, 음질, turn-taking 성능을 검증한 결과는 아닙니다.

재실행:

```bash
# 원본 HF 캐시를 자동으로 찾아 링크하고 실제 음성을 토큰화합니다.
./scripts/python.sh scripts/validate_summ_re.py

# 내부에서 지정된 Python 경로의 torch.distributed.run을 호출합니다.
bash scripts/validate_gpu_stages.sh

# 학습에 사용하지 않은 test 회의로 평가합니다.
./scripts/python.sh scripts/evaluate_summ_re_validation.py
```

검증 산출물은 `prepared_data/summ_re_validation/preparation.json`,
`outputs/summ_re_validation/stage*/metrics.json`, `stage*.log`,
`outputs/summ_re_validation/validation_report.json`입니다.
각 단계의 `best_model/`에는 다음 단계에서 실제로 읽은 모델과 tokenizer가 저장되어 있습니다.

| 단계 | 학습 전 validation loss | 학습 후 validation loss |
|---|---:|---:|
| 1 — ASR/TTS | 16.0831 | 13.5692 |
| 2 — Half-duplex | 13.5704 | 13.0022 |
| 3a — Three-stream | 13.7217 | 13.8859 |
| 3b — Two-stream | 15.7398 | 13.5185 |

Stage 3a는 loss가 소폭 상승했습니다. 모든 단계에서 forward/backward, 유한한 gradient, optimizer update, validation, checkpoint 저장·다음 단계 로드를 확인했지만, 이 작은 실험으로 성능 향상을 보장할 수는 없습니다. 이번 짧은 시퀀스 실행의 PyTorch peak allocated memory는 GPU당 약 **4.68 GiB**였습니다. 그래픽 데스크톱·CUDA context의 메모리는 별도이며, 8192토큰 학습도 8GB에 들어간다는 의미는 아닙니다.

학습용 음성 추출은 `CosyVoiceSpeechTokenizer`만 로드하므로 TTS 의존성이나 화자 preset 없이 받은 `CosyVoice-300M`으로 동작합니다. 실제 음성 복원·TTS는 별도의 `CosyVoiceCodec`과 upstream CosyVoice 설치 및 화자 conditioning이 필요합니다.

## SUMM-RE 전체 Stage 1 데이터 준비

아래 스크립트는 로컬 `default` Arrow 캐시를 자동으로 찾고, 지정된 Python 환경과 `pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx`를 사용하여 모든 split의 음성 토큰을 미리 계산합니다.

```bash
bash scripts/prepare_summ_re_full.sh
```

결과는 `prepared_data/summ_re_full/train.jsonl`, `val.jsonl`, `test.jsonl` 및 `source.json`입니다. 각 줄에는 전사와 raw CosyVoice 코드북 ID가 들어 있어 Stage 1 학습 시 별도 음성 인코딩이 필요하지 않습니다. 기존 train/dev/test 분할을 보존하고, 회의 ID가 split 간 겹치면 중단합니다. 684개 녹음 행의 155,898개 발화 구간 전체를 순회합니다.

기본적으로 12초보다 긴 구간을 단어 timestamp로 나누어 전사 단어를 보존합니다. 12초보다 긴 `sil` 단어는 최대 12초로 줄이고 결과에 `shortened_silence: true`를 기록합니다. 단일 비무음 단어의 timestamp가 12초를 넘는 등 안전하게 분할할 수 없는 구간은 `<split>.review.jsonl`에 원본 정보와 사유를 보관하고 학습 manifest에서는 제외합니다. 현재 캐시에서 이러한 비무음 timestamp 이상치는 4개로 확인됐습니다. 그 외 예기치 못한 오류는 조용히 건너뛰지 않고 중단합니다.

실행 중 Ctrl+C 등으로 중단한 경우 **동일한 명령을 다시 실행하면 마지막으로 저장한 원본 발화 구간부터 재개**합니다. 완료 전에는 `<split>.jsonl.part`에 저장하며 해당 split이 완료되면 최종 JSONL을 게시합니다. 입력 파일·코덱·최대 길이가 바뀐 경우에는 기존 출력 디렉터리를 재사용하지 않고 다른 출력 경로를 지정해야 합니다. 세부 설정은 `bash scripts/prepare_summ_re_full.sh --help`에서 볼 수 있습니다.

전체 실행 전에 각 split에서 2개 구간만 시험하려면 별도 임시 출력 경로를 사용합니다.

```bash
SUMM_RE_OUTPUT_DIR=/tmp/summ-re-check SUMM_RE_MAX_SEGMENTS=2 \
  bash scripts/prepare_summ_re_full.sh
```

CPU ONNX 토크나이저를 사용하므로 전체 준비에는 시간이 걸립니다. 이 스크립트는 Stage 1의 음성-전사 데이터를 생성하며, assistant 답변이 포함된 Stage 2·3 학습 데이터는 별도로 준비해야 합니다.
