#!/usr/bin/env bash
# Prepare all cached SUMM-RE splits for Stage 1, with resumable speech encoding.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin=/data/util/anaconda3/envs/duplexomni/bin/python
export PYTHONPATH="$project_dir/.runtime:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

arrow_dir="${SUMM_RE_ARROW_DIR:-}"
codec_dir="${COSYVOICE_DIR:-$project_dir/pretrained_models/CosyVoice-300M}"
output_dir="${SUMM_RE_OUTPUT_DIR:-$project_dir/prepared_data/summ_re_full}"
max_duration="${SUMM_RE_MAX_DURATION:-12}"
threads="${SUMM_RE_THREADS:-2}"
max_segments="${SUMM_RE_MAX_SEGMENTS:-0}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Prepare every cached SUMM-RE speech/transcript segment for Stage 1.
Usage: bash scripts/prepare_summ_re_full.sh
Optional environment variables:
  SUMM_RE_ARROW_DIR     Cache directory containing train/dev/test Arrow shards
  COSYVOICE_DIR         CosyVoice v1 model directory
  SUMM_RE_OUTPUT_DIR    Output directory (default: prepared_data/summ_re_full)
  SUMM_RE_MAX_DURATION  Maximum fragment length in seconds (default: 12)
  SUMM_RE_THREADS       CPU threads for the ONNX tokenizer (default: 2)
  SUMM_RE_MAX_SEGMENTS  Limit per split for testing; 0 means all (default: 0)
Rerun the same command after interruption to resume from the last segment.
EOF
  exit 0
fi
if (( $# )); then
  printf 'Unknown arguments. Use --help; configure paths via environment variables.\n' >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Python not found: %s\n' "$python_bin" >&2
  exit 1
fi
if [[ -z "$arrow_dir" ]]; then
  arrow_dir="$("$python_bin" - <<'PY'
from pathlib import Path
from datasets.config import HF_DATASETS_CACHE
root = Path(HF_DATASETS_CACHE) / 'linagora___summ-re' / 'default'
matches = sorted({p.parent for p in root.rglob('*-train-*.arrow')})
if len(matches) != 1:
    raise SystemExit(f'Expected one cached default dataset directory, found {matches}; set SUMM_RE_ARROW_DIR')
print(matches[0])
PY
)"
fi
if [[ ! -f "$codec_dir/speech_tokenizer_v1.onnx" ]]; then
  printf 'CosyVoice v1 tokenizer not found in %s\n' "$codec_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
exec 9>"$output_dir/.prepare.lock"
if ! flock -n 9; then
  printf 'Another preparation process is running for %s\n' "$output_dir" >&2
  exit 1
fi
printf 'Python: %s\nArrow cache: %s\nCodec: %s\nOutput: %s\n' "$python_bin" "$arrow_dir" "$codec_dir" "$output_dir"
"$python_bin" scripts/prepare_summ_re_full.py \
  --arrow-dir "$arrow_dir" --codec-dir "$codec_dir" --output-dir "$output_dir" \
  --max-duration "$max_duration" --max-segments "$max_segments" --threads "$threads"
