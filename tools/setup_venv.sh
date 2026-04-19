#!/usr/bin/env bash
# hf_to_gguf_setup.sh — Set up a Python venv for llama.cpp's convert_hf_to_gguf.py
#
# Creates a venv at bin/.venv, symlinks the conversion script into bin/,
# generates a temporary requirements.txt via pipreqs, installs deps,
# and removes the requirements file. Idempotent — exits 0 if bin/.venv exists.

SCRIPT_DIR=$(dirname "${0}")
SCRIPT_DIR=$(cd "${SCRIPT_DIR}" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

VENV="${SCRIPT_DIR}/.venv"

if [[ -f "${VENV}/done.txt" ]]; then
  echo "Setup already done"
  exit 0
fi

PYTHON=$(which python3 2> /dev/null) || \
  PYTHON=$(which python 2>/dev/null) || \
  { echo "neither python3 nor python found. python 3.11 required." 1>&2; exit 1; }

PYVER=$("${PYTHON}" --version 2>&1)
PYVER=${PYVER/*ython /}
PYVER=${PYVER%.*}
[[ "${PYVER}" == 3.11 ]] || {
  echo "Error: python version ${PYVER} is not supported. python 3.11 required." 1>&2
  echo "  pyenv install 3.11 && pyenv global 3.11 is one easy way to fix this." 1>&2
  exit 1
}

(rm -rf "${VENV}" 2> /dev/null) || true
"${PYTHON}" -m venv "${VENV}" || {
  echo "Error: failed to create venv at ${VENV}" 1>&2
  exit 1
}

PYTHON="${VENV}/bin/python3"

"${PYTHON}" -m pip install --upgrade pip || {
  echo "Error: failed to upgrade pip" 1>&2
  exit 1
}

# "${PYTHON}" -m pip install Cython &&
"${PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt" || {
  echo "Error: failed to install dependencies" 1>&2
  exit 1
}

date > "${VENV}/done.txt"

echo "Setup complete: ${VENV} ready for hf-to-gguf conversion"
