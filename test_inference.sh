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
USE_LLAMA=${USE_LLAMA:-false}

TOP_LOGPROBS=${TOP_LOGPROBS:-0}
if [[ "${TOP_LOGPROBS}" -gt 0 ]]; then
  LOGPROBS=${LOGPROBS:-true}
else
  LOGPROBS=${LOGPROBS:-false}
fi

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

PORT=${PORT:-auto}
if [[ ${PORT} == "auto" ]]; then
  # parse the port out of server config
  PORT=$(awk '/^[ \t]*port[ \t]*=/ { gsub("=", " "); print $2; exit(0); }' config/api_config.toml)
  # echo PORT=${PORT} && exit 0
fi
SERVER_BASE_URL=${SERVER_BASE_URL:-"http://localhost:${PORT}"}
CTL_BASE_URL=${CTL_BASE_URL:-"${SERVER_BASE_URL}/ctl"}
API_BASE_URL=${API_BASE_URL:-"${SERVER_BASE_URL}/api/v1"}
STATELESS=${STATELESS:-false}
THINK=${THINK:-false}
ELIDE_THINK=${ELIDE_THINK:-null}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0}
MODEL=${MODEL:-default}
STREAM=${STREAM:-false}

(! [[ "${USE_LLAMA}" == "true" && "${MODEL}" == "default" && "${ALL_MODELS}" != "true" ]]) || { echo "ALL_MODELS must be true or a MODEL must be specified." 1>&2; exit 1; }

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


function parse_response() {
  # maybe fall back to a more brittle awk-based extractor ?
  [[ -x "${PYTHON}" ]] || { return 1; }

# when streaming is true the response is a sequence of disjointed dictionaries (not comma separated)
# needs handling implemented in the python snippet.
  MODEL=${MODEL} "${PYTHON}" -c '
import os,sys,json
resp = json.load(sys.stdin)
if "choices" in resp:
  resp = resp["choices"][0]
  resp_text = resp["message"]["content"]
  print(resp_text)
  if "logprobs" in resp:
    logprobs = json.dumps(resp["logprobs"]["content"][0]["top_logprobs"], sort_keys=True)
    model = os.getenv("MODEL")
    print(f"{model}|{logprobs}|{resp_text}")
else:
  print(resp)
' 2> /dev/null
}

function query_one() {
  local msg="${1}"
  local ext_params

  if ${USE_LLAMA}; then
    ext_params=$(printf '
      "chat_template_kwargs": { "enable_thinking": %s },
    ' \
    "${THINK}")
  else
    ext_params=$(printf '
      "enable_thinking": %s,
      "stateless": %s,
      "elide_thinking": %s,
    ' \
    "${THINK}" \
    "${STATELESS}" \
    "${ELIDE_THINK}" \
    )
  fi

  local payload=$(printf '{
      %s
      "messages": [{"role":"user","content":"%s"}],
      "model": "%s",
      "max_tokens": %s,
      "temperature": %s,
      "logprobs": %s,
      "top_logprobs": %s,
      "stream": %s
    }' \
    "${ext_params}" \
    "${msg}" \
    "${MODEL}" \
    "${MAX_TOKENS}" \
    "${TEMPERATURE}" \
    "${LOGPROBS}" \
    "${TOP_LOGPROBS}" \
    "${STREAM}" \
    )
  local completions_url=${COMPLETIONS_URL:-"${API_BASE_URL}/chat/completions"}
  ${DEBUG_POST:-false} && echo "[DBG] ${completions_url} POST payload: ${payload}" || true
  local resp=$(curl -s -X POST "${completions_url}" \
    -H 'Content-Type: application/json' \
    -d "${payload}")
  echo "${resp}" | parse_response || echo "${resp}"
}

function query() {
  local model
  if ${ALL_MODELS}; then
    for model in ${MODEL_LIST%default}; do
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

function show_models() {
  local model
  local models
  local i
  models=$(curl -sf "${API_BASE_URL}/models") || return 1
  models=$(parse_models <<< "${models}")
  MODEL_LIST="${models} default"

  # if using llama (which publishes models besides our ggufs
  # or trying to pull logits for comparison, ensure consistent model order
  ! (${LOGPROBS} || ${USE_LLAMA}) || {
    MODEL_LIST=$(cd models; ls *.gguf | sort)
    MODEL_LIST=${MODEL_LIST//\.gguf/}
  }
  echo "models"
  echo "======"
  i=1
  for model in ${MODEL_LIST}; do
    [[ ${model} == ${MODEL} ]] && echo "${i}) ${model}*" || echo "${i}) ${model}"
    i=$((i + 1))
  done
  echo ""
}

function quit_servers() {
  if [[ -n "${SERVER_PID}" ]]; then
    curl -s "${CTL_BASE_URL}?quit&now" > /dev/null || {
      ps aux | grep "bin/bench" | awk '!/grep/ { print $2; }' | xargs kill  sleep 1
    }
  fi

  if [[ -n "${LLAMA_PID}" ]]; then
    # blunt instrument
    ps aux | grep "llama-server" | awk '!/grep/ { print $2; }' | xargs kill
  fi
}

function wait_for_starting_server() {
  # Wait for server to be ready (up to 30s)
  for _i in $(seq 1 60); do
    if show_models > /dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "failed getting models from ${API_BASE_URL}" 1>&2
  return 1
}

function run_api_server() {
  ./bin/bench serve-api 2>&1
}

function start_new_api_server() {
  [[ -x "./bin/bench" ]] || { echo "./bin/bench binary missing. 'make all' first." 1>&2; exit 1; }

  ps aux | grep "bin/bench" | awk '!/grep/ { print $2; }' | xargs kill
  sleep 1
  run_api_server >> ./bin/test_inference.log &
  SERVER_PID=$!

  wait_for_starting_server
}

LLAMA_PORT=${LLAMA_PORT:-8080}
LLAMA_BASE_URL="http://localhost:${LLAMA_PORT}"

function run_llama_server() {
  llama-server --models-dir "$(pwd)/models" --port ${LLAMA_PORT} --ctx-size 8192 2>&1
}

function start_llama_server() {
  (which llama-server 1>&2) > /dev/null || { echo "llama-server not installed." 1>&2; exit 1; }

  ps aux | grep llama-server | awk '!/grep/ { print $2; }' | xargs kill
  sleep 1
  run_llama_server >> bin/llama-server.log &
  LLAMA_PID=$!

  API_BASE_URL=${LLAMA_BASE_URL}/v1
  SERVER_BASE_URL=${LLAMA_BASE_URL}
  PORT=${LLAMA_PORT}

  wait_for_starting_server
}

trap quit_servers EXIT

if [[ "${USE_LLAMA}" == "true" ]]; then
  start_llama_server
elif [[ "${FORCE_NEW_SERVER:-false}" == "true" ]]; then
  start_new_api_server
fi

show_models || { echo "inference server not running. 'make serve' or use envar param FORCE_NEW_SERVER=true" 1>&2; exit 1; }

if [[ "${LOOP_MODE}" == true ]]; then
  loop_mode
else
  query "${MSG}"
fi
