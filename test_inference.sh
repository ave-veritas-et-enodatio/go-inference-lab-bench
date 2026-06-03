#!/usr/bin/env bash
THIS_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=$(command -v python3 2>/dev/null) || \
  PYTHON=$(command -v python 2>/dev/null) || \
  { echo "neither python3 nor python found. python3 required (sorry)." 1>&2; exit 1; }
cd "${THIS_DIR}"
exec "${PYTHON}" "${THIS_DIR}/tools/test_inference.py" "$@"
