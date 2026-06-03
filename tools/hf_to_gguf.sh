#!/usr/bin/env bash
# hf_to_gguf.sh — Convert HuggingFace models to GGUF (tokenizer sidecar + passthrough)
#
# Usage:
#   hf_to_gguf.sh --bench-tokenizer <model-name-or-path>
#     Generates a tokenizer-only GGUF sidecar at models/<model-name>.st/tokenizer.gguf
#     If <model-name-or-path> contains '/', it is treated as an absolute/relative path,
#     otherwise resolved under models/<model-name-or-path>.st
#
#   hf_to_gguf.sh --bench-convert <model-name-or-path>
#     Bench-flavored full conversion: produces models/<name>.gguf (F16 decoder), and
#     additionally produces mmproj-<name>.gguf if the source config.json declares a
#     vision_config or audio_config (multimodal model). The mmproj second pass is
#     tolerant of failure — if convert_hf_to_gguf.py's --mmproj support doesn't cover
#     the target architecture, the decoder GGUF is still produced and a prominent
#     warning is printed explaining that vision/audio inputs will not work.
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

OUT_TYPE=${OUT_TYPE:-f16}

# resolve_st_dir <mode-name-for-error-msg> <model-name-or-path>
# Validates that a model name was given and resolves it to safetensors paths.
# Sets globals: ST_DIR (project-relative), ST_ABS (absolute path).
# Argument convention shared between --bench-tokenizer and --bench-convert:
#   - If the value contains '/', it's treated as an absolute or relative path.
#   - Otherwise it's resolved as models/<value>.st under the project root.
function resolve_st_dir() {
  local mode="${1}"
  local model_name="${2}"
  if [[ -z "${model_name}" ]]; then
    echo "Error: usage: hf_to_gguf.sh --${mode} <model-name-or-path>" 1>&2
    exit 1
  fi
  if [[ "${model_name}" == *"/"* ]]; then
    ST_DIR="${model_name}"
  else
    ST_DIR="models/${model_name}.st"
  fi
  ST_ABS="${PROJECT_ROOT}/${ST_DIR}"
}

if [[ "${1}" == "--bench-tokenizer" ]]; then
  resolve_st_dir "bench-tokenizer" "${2}"

  mkdir -p "${ST_ABS}"

  PYTHONPATH=${SCRIPT_DIR} "${PYTHON}" "${SCRIPT_DIR}/convert_hf_to_gguf.py" \
    "${ST_ABS}" \
    --vocab-only \
    --outfile "${ST_ABS}/tokenizer.gguf"

  EXIT_CODE=$?
  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "Error: tokenizer generation failed (exit code ${EXIT_CODE})" 1>&2
    exit ${EXIT_CODE}
  fi

  echo "Tokenizer sidecar written to ${ST_DIR}/tokenizer.gguf"

elif [[ "${1}" == "--bench-convert" ]]; then
  resolve_st_dir "bench-convert" "${2}"

  if [[ ! -d "${ST_ABS}" ]]; then
    echo "Error: ${ST_DIR} is not a directory" 1>&2
    exit 1
  fi
  if [[ ! -f "${ST_ABS}/config.json" ]]; then
    echo "Error: ${ST_DIR}/config.json missing — not an HF model directory?" 1>&2
    exit 1
  fi

  # Derive output paths: strip trailing .st from the basename so e.g.
  # models/gemma-4-E4B-it.st → models/gemma-4-E4B-it.gguf alongside it.
  ST_BASENAME=$(basename "${ST_ABS}")
  MODEL_BASE="${ST_BASENAME%.st}"
  OUT_DIR=$(dirname "${ST_ABS}")
  OUT_GGUF="${OUT_DIR}/${MODEL_BASE}-${OUT_TYPE}.gguf"
  MMPROJ_GGUF="${OUT_DIR}/mmproj-${MODEL_BASE}-${OUT_TYPE}.gguf"

  # ---- Decoder/text conversion (always runs) ----
  echo ">>> Decoder: ${ST_DIR} → ${OUT_GGUF##${PROJECT_ROOT}/}"
  PYTHONPATH=${SCRIPT_DIR} "${PYTHON}" "${SCRIPT_DIR}/convert_hf_to_gguf.py" \
    "${ST_ABS}" \
    --outtype ${OUT_TYPE} \
    --outfile "${OUT_GGUF}"

  EXIT_CODE=$?
  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "Error: decoder conversion failed (exit code ${EXIT_CODE})" 1>&2
    exit ${EXIT_CODE}
  fi

  # ---- Multimodal detection ----
  # Standard HF convention: config.json carries a nested <modality>_config block
  # when the model has a tower for that modality. The mmproj sidecar covers all
  # non-text encoders the converter knows about (vision/audio/video).
  #
  # Note: in many releases (incl. Gemma 4), video is handled as frame-sampled
  # input to the vision tower rather than via a dedicated video encoder — so
  # video_config rarely appears as a standalone block. We grep for it anyway:
  # cost is nil, and it catches the cases where a model does ship a dedicated
  # video encoder. Substring grep is reliable here because these key names are
  # distinctive — no plausible false positives.
  if grep -qE '"(vision|audio|video)_config"' "${ST_ABS}/config.json"; then
    echo ">>> Multimodal config detected; attempting mmproj sidecar"
    PYTHONPATH=${SCRIPT_DIR} "${PYTHON}" "${SCRIPT_DIR}/convert_hf_to_gguf.py" \
      "${ST_ABS}" \
      --mmproj \
      --outtype ${OUT_TYPE} \
      --outfile "${MMPROJ_GGUF}"
    MMPROJ_EXIT=$?
    if [[ ${MMPROJ_EXIT} -eq 0 ]]; then
      echo ">>> mmproj sidecar: ${MMPROJ_GGUF##${PROJECT_ROOT}/}"
    else
      # Prominent warning so the user doesn't silently get a vision-less GGUF.
      # convert_hf_to_gguf.py's --mmproj is marked "(Experimental) ... will only
      # work on some vision models." Failure here is upstream, not local.
      echo "" 1>&2
      echo "================================================================" 1>&2
      echo "  WARNING: mmproj sidecar conversion FAILED for ${MODEL_BASE}" 1>&2
      echo "" 1>&2
      echo "  The source model declares a vision/audio/video config block in" 1>&2
      echo "  config.json, but convert_hf_to_gguf.py's --mmproj pass exited" 1>&2
      echo "  with code ${MMPROJ_EXIT}." 1>&2
      echo "" 1>&2
      echo "  The decoder GGUF was produced and works for text input." 1>&2
      echo "  IMAGE / AUDIO / VIDEO INPUTS WILL NOT WORK until upstream" 1>&2
      echo "  support for this architecture is added to llama.cpp's converter." 1>&2
      echo "================================================================" 1>&2
      echo "" 1>&2
    fi
  fi

  echo ">>> Done."

else
  "${PYTHON}" "${SCRIPT_DIR}/convert_hf_to_gguf.py" "$@"
  EXIT_CODE=$?
  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "Error: conversion failed (exit code ${EXIT_CODE})" 1>&2
    exit ${EXIT_CODE}
  fi
fi
