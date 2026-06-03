#!/usr/bin/env bash
# quantize_gguf.sh — quantize an f16 GGUF with an optional imatrix
#
# Usage:
#   [QUANT=<quant>]  quantize_gguf.sh <input.f16.gguf> [imatrix.gguf_file] [output.gguf]
#   default QUANT is Q4_K_M
#   default output is <input>-${QUANT}.gguf
#
set -euo pipefail

function usage() {
  [[ -z "${1:-}" ]] || echo "error: ${1}" 1>&2
  echo "usage: ${0} <input.f16.gguf> [imatrix.gguf_file] [quant] [output.gguf]" 1>&2
  [[ -z "${1:-}" ]] && exit 0 || exit 1
}

QUANT=
INPUT=
IMATRIX=
OUTPUT=

while [[ -n "${1:-}" ]]; do
  case "${1}" in
    -h*|--h*|-u*|--u*) usage;;
    *.gguf_file|*imatrix*)
      [[ -z "${IMATRIX}" ]] && IMATRIX=${1} || usage "imatrix already set: ${IMATRIX}"
      ;;
    *.gguf)
      [[ -z "${OUTPUT}" ]] || usage "output already set: ${OUTPUT}"
      [[ -z "${INPUT}" ]] && INPUT=${1} || OUTPUT=${1}
      ;;
    Q*) [[ -z "${QUANT}" ]] && QUANT=${1} || usage "quant already set: ${QUANT}";;
      *) usage "unrecognized argument: ${1}";;
  esac
  shift
done

[[ -n "${INPUT}" ]] || usage "input gguf required."
QUANT=${QUANT:-Q4_K_M}
OUTPUT=${OUTPUT:-${INPUT%.gguf}-${QUANT}.gguf}

[[ -f "${INPUT}" ]] || usage "input gguf not found: ${INPUT}"
[[ ! -e "${OUTPUT}" ]] || { echo "output already exists (refusing to overwrite): ${OUTPUT}" 1>&2; exit 1; }
[[ -z "${IMATRIX}" || -f "${IMATRIX}" ]] || { echo "imatrix not found: ${IMATRIX}" 1>&2; exit 1; }

[[ -n "$(which llama-quantize 2> /dev/null)" ]] || {
  echo "llama-quantize not found on PATH (brew install llama-cpp)" 1>&2
  exit 1
}

OPTS=()
[[ -z "${IMATRIX}" ]] || OPTS+=(--imatrix "${IMATRIX}")
OPTS+=("${INPUT}" "${OUTPUT}" ${QUANT})

set -x
llama-quantize "${OPTS[@]}"
