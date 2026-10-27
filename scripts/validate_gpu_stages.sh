#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/.runtime:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python_bin=/data/util/anaconda3/envs/duplexomni/bin/python
output_dir=outputs/summ_re_validation
mkdir -p "$output_dir"
previous=''
for stage in 1 2 3a 3b; do
  dataset_stage="$stage"
  if [[ "$stage" == 3a || "$stage" == 3b ]]; then dataset_stage=3; fi
  extra=(--backbone pretrained_models/Qwen2-0.5B)
  if [[ -n "$previous" ]]; then extra=(--checkpoint "$previous"); fi
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --stage "$stage" --data_dir "prepared_data/summ_re_validation/stage${dataset_stage}" \
    --output_dir "$output_dir" --epochs 1 --batch_size 1 --eval_batch_size 1 \
    --gradient_accumulation_steps 1 --bf16 --model_dtype bfloat16 \
    --gradient_checkpointing --eval_before_training --log_interval 1 --num_workers 0 \
    "${extra[@]}" 2>&1 | tee "$output_dir/stage${stage}.log"
  previous="$output_dir/stage${stage}/best_model"
done
