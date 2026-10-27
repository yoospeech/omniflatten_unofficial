#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_dir}/.runtime:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec /data/util/anaconda3/envs/duplexomni/bin/python "$@"
