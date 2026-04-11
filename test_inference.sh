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
STATELESS=${STATELESS:-null}
# CULL_METHOD: null (no culling, use server config), "none", or "random".
CULL_METHOD=${CULL_METHOD:-null}
THINK=${THINK:-null}
ELIDE_THINK=${ELIDE_THINK:-null}
FLASH=${FLASH:-null}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0}
MODEL=${MODEL:-default}
STREAM=${STREAM:-false}
DEBUG_POST=${DEBUG_POST:-false}
DEBUG_RESPONSE=${DEBUG_RESPONSE:-false}


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
  DEBUG_RESPONSE=${DEBUG_RESPONSE} MODEL=${MODEL} "${PYTHON}" -c '
import os,sys,json
resp_json = str(sys.stdin.read())
try:
  resp_all = json.loads(resp_json)
except:
  resp_all = {}
printed = False
if "choices" in resp_all:
  resp_text = resp_all["choices"][0]
  resp_text = resp_text["message"]["content"]
  resp_text = resp_text if resp_text.strip() else "<NO-MODEL-RESPONSE>"
  usage = resp_all.get("usage", {})
  itps = usage.get("prompt_tokens_per_sec", 0.0)
  otps = usage.get("completion_tokens_per_sec", 0.0)
  ttps = usage.get("total_tokens_per_sec", 0.0)
  secs = usage.get("total_seconds", 0.0)
  print(f"{resp_text} {{itps:{itps:.2f}, otps:{otps:.2f}, ttps:{ttps:.2f}, s:{secs:.2f}}}")
  printed = True
  logprobs = resp_all.get("choices", [])
  if logprobs := (logprobs[0] if logprobs else {}).get("logprobs", {}):
    logprobs = json.dumps(logprobs["content"][0]["top_logprobs"], sort_keys=True)
    model = os.getenv("MODEL")
    print(f"{model}|{logprobs}|{resp_text}")

if (not printed) or (os.getenv("DEBUG_RESPONSE", "false") in ["true", "1"]):
  print(json.dumps(resp_all, indent=2) if resp_all else resp_json if resp_json.strip() else "<NO-API-RESPONSE>")
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
      "cull_method": %s,
      "elide_thinking": %s,
      "flash_attention": %s,
    ' \
    "${THINK}" \
    "${STATELESS}" \
    "${CULL_METHOD}" \
    "${ELIDE_THINK}" \
    "${FLASH}" \
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
  ${DEBUG_POST} && echo "[DBG] ${completions_url} POST payload: ${payload}" || true
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
      # llama server tries to hold all models in memory and fails
      # there's a ggml or metal driver bug that causes cummulative state resets.
      # this issue was investigated exhaustively and was proven to reside outside of this code base.
      # to produce incorrect results; cycle server to ensure a clean metal context.
      cycle_server
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
/debug-post: toggle display of post json (currently: ${DEBUG_POST})
/debug-response: toggle display of response json (currently: ${DEBUG_RESPONSE})
/flash: toggle flash attention mode (currently: ${FLASH})
/stateless: toggle stateless mode (currently: ${STATELESS})
/cull [method]: set cull method (currently: ${CULL_METHOD}); "random", "none", or null
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
      /flash)
        FLASH=$(rotate "${FLASH}")
        echo "FLASH=${FLASH}"
        continue;;
      /stateless)
        STATELESS=$(rotate "${STATELESS}")
        echo "STATELESS=${STATELESS}"
        continue;;
      /cull*)
        local CM=$(set ${line}; echo ${2})
        if [[ -n "${CM}" ]]; then
          if [[ "${CM}" == "null" || "${CM}" == "none" ]]; then
            CULL_METHOD=null
          else
            CULL_METHOD="\"${CM}\""
          fi
        fi
        echo "CULL_METHOD=${CULL_METHOD}"
        continue;;
      /debug-post)
        DEBUG_POST=$(toggle "${DEBUG_POST}")
        echo "DEBUG_POST=${DEBUG_POST}"
        continue;;
      /debug-response)
        DEBUG_RESPONSE=$(toggle "${DEBUG_RESPONSE}")
        echo "DEBUG_RESPONSE=${DEBUG_RESPONSE}"
        continue;;
      /stream-NYI)
        STREAM=$(rotate "${STREAM}")
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

function start_api_server() {
  [[ -x "./bin/bench" ]] || { echo "./bin/bench binary missing. 'make all' first." 1>&2; exit 1; }
  ./bin/bench serve-api serve-api --log ${LOG:-"bin/test_inference.log"} --log-level NONE &
  SERVER_PID=$!
  wait_for_starting_server
}

function run_llama_server() {
  llama-server --models-dir "$(pwd)/models" --port ${LLAMA_PORT} --ctx-size 8192 2>&1
}

function start_llama_server() {
  (which llama-server 2>&1) > /dev/null || { echo "llama-server not installed." 1>&2; exit 1; }
  run_llama_server >> ${LOG:-"bin/test_inference_llama.log"} &
  LLAMA_PID=$!
  API_BASE_URL=${LLAMA_BASE_URL}/v1
  SERVER_BASE_URL=${LLAMA_BASE_URL}
  PORT=${LLAMA_PORT}
  wait_for_starting_server
}

function quit_api_server() {
  (curl -s "${CTL_BASE_URL}/?quit&now" > /dev/null && sleep 1) || {
    ps aux | grep "bin/bench" | awk '!/grep/ { print $2; }' | xargs kill
  }
}

function quit_llama_server() {
  # blunt instrument
  ps aux | grep "llama-server" | awk '!/grep/ { print $2; }' | xargs kill
}

function quit_server() {
  [[ -n "${SERVER_PID}" ]] && quit_api_server || true
  [[ -n "${LLAMA_PID}" ]] && quit_llama_server || true
}

LLAMA_PORT=${LLAMA_PORT:-8080}
LLAMA_BASE_URL="http://localhost:${LLAMA_PORT}"

function cycle_server() {
    quit_server
    if ${USE_LLAMA}; then start_llama_server; else start_api_server; fi
}

trap quit_server EXIT

if ${FORCE_NEW_SERVER:-false}; then
  if ${USE_LLAMA}; then
    quit_llama_server
    start_llama_server
  else
    quit_api_server
    start_api_server
  fi
fi

if ${LOOP_MODE}; then
  show_models || { echo "inference server not running. 'make serve' or use envar param FORCE_NEW_SERVER=true" 1>&2; exit 1; }
  loop_mode
else
  show_models > /dev/null || { echo "inference server not running. 'make serve' or use envar param FORCE_NEW_SERVER=true" 1>&2; exit 1; }
  query "${MSG}"
fi
