#!/usr/bin/env bash

MODEL=${1}
[[ -n "${MODEL}" ]] || {
  echo "model file not specified. usage: [CHUNKS=<chunk_count>] ${0} <model.f16.gguf> [imatrix.gguf_file]" 1>&2
  exit 1
}
OUTPUT=${2:-${MODEL%.gguf}-imatrix.gguf_file}

[[ -f "${MODEL}" ]] || { echo "gguf model file not found: ${MODEL}"; exit 1; }
[[ ! -f "${OUTPUT}" ]] || { echo "output file already exists: ${OUTPUT}"; exit 1; }

[[ -n "$(which llama-imatrix 2> /dev/null)" ]] || { echo "llama-imatrix not found"; exit 1; }
[[ -f calibration_datav5.txt ]] || {
  CALIBRATION_DATA_URL="https://gist.githubusercontent.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d/raw/ce111d8971a07caebd8234ef336b2102d6c5fb85/calibration_datav5.txt"
  curl -O "${CALIBRATION_DATA_URL}" || exit 1
}

set -x
llama-imatrix -f calibration_datav5.txt -m "${MODEL}" -o "${OUTPUT}" --chunks ${CHUNKS:-"-1"}
