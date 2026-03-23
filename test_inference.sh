#!/usr/bin/env bash
THIS_SCRIPT=$(basename "$0")
THIS_DIR=$(dirname "$0")
set -e

cd "${THIS_DIR}"

usage() {
  echo "Usage: ${THIS_SCRIPT} --loop|<message>"
  exit 1
}

LOOP_MODE=false
ALL_MODELS=${ALL_MODELS:-false}

if [[ "${1}" == "--loop" ]]; then
  LOOP_MODE=true
  shift
elif [[ "${1}" == -* ]]; then
  echo "unknown option: ${1}"
  usage
  exit 1
elif [[ -z "${1}" ]]; then
  usage
else
  if [[ "${1}" == "--all-models" ]]; then
    ALL_MODELS=true
    shift
  fi
  MSG="${*}"
fi

# port 11116 is the default server port.
# if you edit api_config.toml either set in environment or edit this script.
PORT=${PORT:-11116}
API_BASE_URL=${API_BASE_URL:-"http://localhost:${PORT}/api/v1"}
STATELESS=${STATELESS:-false}
THINK=${THINK:-null}
ELIDE_THINK=${ELIDE_THINK:-null}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0}
MODEL=${MODEL:-default}
STREAM=${STREAM:-false}

PYTHON=$(which python3 2> /dev/null) || \
  PYTHON=$(which python 2>/dev/null) || \
  { echo "python not found. LLM responses will be raw JSON." 1>&2; }


function parse_models() {
  # maybe fall back to a more brittle awk-based extractor ?
  [[ -x "${PYTHON}" ]] || { cat; return 1; }

  "${PYTHON}" -c '
import sys,json
models = json.load(sys.stdin)
models = [entry["id"] for entry in models["data"]]
print("\n".join(models))
' 2> /dev/null
}

MODEL_LIST=

function show_models() {
  local model
  local models
  local i
  models=$(curl -sf "${API_BASE_URL}/models") || return 1
  models=$(parse_models <<< "${models}")
  MODEL_LIST="${models} default"
  echo "models"
  echo "======"
  i=1
  for model in ${MODEL_LIST}; do
    [[ ${model} == ${MODEL} ]] && echo "${i}) ${model}*" || echo "${i}) ${model}"
    i=$((i + 1))
  done
  echo ""
}

# By default, assume a server is already running.
# Set FORCE_NEW_SERVER=true to start a fresh one (kills any existing bench process).
if [[ "${FORCE_NEW_SERVER:-false}" == "true" ]]; then
  [[ -x "./bin/bench" ]] || { echo "./bin/bench binary missing. 'make all' first." 1>&2; exit 1; }

  pkill -f './bin/bench' 2>/dev/null || true
  sleep 1
  ./bin/bench serve-api 2> ./bin/test_inference.log &
  SERVER_PID=$!

  function kill_server() {
    kill ${SERVER_PID} 2>/dev/null
    wait ${SERVER_PID} 2>/dev/null
  }

  trap kill_server EXIT

  # Wait for server to be ready (up to 30s)
  for _i in $(seq 1 60); do
    if show_models > /dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi


function parse_response() {
  # maybe fall back to a more brittle awk-based extractor ?
  [[ -x "${PYTHON}" ]] || { return 1; }

# when streaming is true the response is a sequence of disjointed dictionaries (not comma separated)
# needs handling implemented in the python snippet.
  "${PYTHON}" -c '
import sys,json
resp = json.load(sys.stdin)
resp = resp["choices"][0]["message"]["content"]
print(resp)
' 2> /dev/null
}

function query_one() {
  local msg="${1}"
  local payload=$(printf '{
      "model": "%s",
      "messages": [{"role":"user","content":"%s"}],
      "max_tokens": %s,
      "temperature": %s,
      "stateless": %s,
      "enable_thinking": %s,
      "elide_thinking": %s,
      "stream": %s
    }' \
    "${MODEL}" \
    "${msg}" \
    "${MAX_TOKENS}" \
    "${TEMPERATURE}" \
    "${STATELESS}" \
    "${THINK}" \
    "${ELIDE_THINK}" \
    "${STREAM}")
  ${DEBUG_POST:-false} && echo "[DBG] POST payload: ${payload}" || true
  local resp=$(curl -s -X POST "${API_BASE_URL}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${payload}")
  echo "${resp}" | parse_response || echo "${resp}"
}

function query() {
  local model
  if ${ALL_MODELS}; then
    for model in ${MODEL_LIST}; do
      [[ ${model} != default ]] || continue
      echo "${model} <-- \"${*}\""
      MODEL=${model} query_one "${@}"
    done
  else
    query_one "${@}"
  fi
}

function loop_help() {
  # stream: toggle streaming mode (currently: ${STREAM}) # STREAM=true NYI

  cat << __EOF
Acontextual Loop Mode
=====================
/help: show this help message
/all-models: toggle all-models mode (currently: ${ALL_MODELS})
/stateless: toggle stateless mode (currently: ${STATELESS})
/think: toggle think mode (currently: ${THINK})
/elide-think; toggle elision of thinking output (currently: ${ELIDE_THINK})
/max-tokens [token_count]: show or set MAX_TOKENS (currently: ${MAX_TOKENS})
/model [index]: show or set current model (currently: ${MODEL})
/temperature [temperature]: show or set TEMPERATURE (currently: ${TEMPERATURE})
/quit or /exit: exit loop mode
__EOF
}

function toggle() {
  case "${1}" in
    true) echo false;;
    *) echo true;;
  esac
}

function rotate() {
  case "${1}" in
    true) echo false;;
    false) echo null;;
    *) echo true;;
  esac
}

function loop_mode() {
  loop_help
  while true; do
    read -e -p "> " line || break
    [[ -n "${line}" ]] && history -s "${line}"
    case "${line}" in
      quit|exit)
        break;;
      /help|/)
        loop_help
        continue;;
      /model*)
        local MI=$(set ${line}; echo ${2})
        if [[ "${MI:-0}" -gt 0 ]]; then
          MODEL=$(set ${MODEL_LIST}; eval "echo \$${MI}")
        else
          show_models
        fi
        echo "MODEL=${MODEL}"
        continue;;
      /all-models)
        ALL_MODELS=$(toggle "${ALL_MODELS}")
        echo "ALL_MODELS=${ALL_MODELS}"
        continue;;
      /stateless)
        STATELESS=$(toggle "${STATELESS}")
        echo "STATELESS=${STATELESS}"
        continue;;
      /stream-NYI)
        STREAM=$(toggle "${STREAM}")
        echo "STREAM=${STREAM}"
        continue;;
      /think)
        THINK=$(rotate "${THINK}")
        echo "THINK=${THINK}"
        continue;;
      /elide-think)
        ELIDE_THINK=$(rotate "${ELIDE_THINK}")
        echo "ELIDE_THINK=${ELIDE_THINK}"
        continue;;
      /max-tokens*)
        local MC=$(set ${line}; echo ${2})
        if [[ "${MC:-0}" -gt 0 ]]; then
          MAX_TOKENS=${MC}
        fi
        echo "MAX_TOKENS=${MAX_TOKENS}"
        continue;;
      /temperature*)
        local T=$(set ${line}; echo ${2})
        if [[ "${T:-0}" -ge 0 ]]; then
          TEMPERATURE=${T}
        fi
        echo "TEMPERATURE=${TEMPERATURE}"
        continue;;
      /*)
        echo "unknown / command ${line}" 1>&2
        loop_help
        continue;;
      "")
      continue;;
    esac
    query "${line}"
  done
}

show_models || { echo "inference server not running. 'make serve' or use envar param FORCE_NEW_SERVER=true" 1>&2; exit 1; }

if [[ "${LOOP_MODE}" == true ]]; then
  loop_mode
else
  query "${MSG}"
fi
