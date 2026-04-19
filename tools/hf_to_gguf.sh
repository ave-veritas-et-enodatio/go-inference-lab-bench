#!/usr/bin/env bash
# hf_to_gguf.sh — Convert HuggingFace models to GGUF (tokenizer sidecar + passthrough)
#
# Usage:
#   hf_to_gguf.sh --bench-tokenizer <model-name-or-path>
#     Generates a tokenizer-only GGUF sidecar at models/<model-name>.st/tokenizer.gguf
#     If <model-name-or-path> contains '/', it is treated as an absolute/relative path,
#     otherwise resolved under models/<model-name-or-path>.st
#
#   hf_to_gguf.sh <convert_hf_to_gguf.py args...>
#     Full passthrough to convert_hf_to_gguf.py with all arguments forwarded verbatim.
#
# Prerequisites: run hf_to_gguf_setup.sh first (done automatically if tool/.venv/done.txt missing).
SCRIPT_DIR=$(dirname "${0}")
SCRIPT_DIR=$(cd "${SCRIPT_DIR}" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)


VENV="${SCRIPT_DIR}/.venv"

if [[ ! -f "${VENV}/done.txt" ]]; then
  echo "Completed venv not found — running setup..."
  "${SCRIPT_DIR}/setup_venv.sh" || {
    echo "Error: venv setup failed" 1>&2
    exit 1
  }
fi

PYTHON="${VENV}/bin/python3"

if [[ "${1}" == "--bench-tokenizer" ]]; then
  MODEL_NAME="${2}"
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "Error: usage: hf_to_gguf.sh --bench-tokenizer <model-name-or-path>" 1>&2
    exit 1
  fi

  if [[ "${MODEL_NAME}" == *"/"* ]]; then
    ST_DIR="${MODEL_NAME}"
  else
    ST_DIR="models/${MODEL_NAME}.st"
  fi

  mkdir -p "${PROJECT_ROOT}/${ST_DIR}"

  PYTHONPATH=${SCRIPT_DIR} "${PYTHON}" "${SCRIPT_DIR}/convert_hf_to_gguf.py" \
    "${PROJECT_ROOT}/${ST_DIR}" \
    --vocab-only \
    --outfile "${PROJECT_ROOT}/${ST_DIR}/tokenizer.gguf"

  EXIT_CODE=$?
  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "Error: tokenizer generation failed (exit code ${EXIT_CODE})" 1>&2
    exit ${EXIT_CODE}
  fi

  echo "Tokenizer sidecar written to ${ST_DIR}/tokenizer.gguf"
else
  "${PYTHON}" "${SCRIPT_DIR}/convert_hf_to_gguf.py" "$@"
  EXIT_CODE=$?
  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "Error: conversion failed (exit code ${EXIT_CODE})" 1>&2
    exit ${EXIT_CODE}
  fi
fi
