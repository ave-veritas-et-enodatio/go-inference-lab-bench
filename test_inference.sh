#!/usr/bin/env bash
THIS_SCRIPT=$(basename "$0")
THIS_DIR=$(dirname "$0")

cd "${THIS_DIR}"

usage() {
  echo "Usage: ${THIS_SCRIPT} --loop|<message>"
  exit 1
}

LOOP_MODE=false
ALL_MODELS_NO_DIFFUSION=${ALL_MODELS_NO_DIFFUSION:-false}
${ALL_MODELS_NO_DIFFUSION} && ALL_MODELS=true || ALL_MODELS=${ALL_MODELS:-false}
USE_LLAMA=${USE_LLAMA:-false}
# prefer safetensors over gguf
PREFER_ST=${PREFER_ST:-false}

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

# parse the port out of server config
BENCH_PORT=$(awk '/^[ \t]*port[ \t]*=/ { gsub("=", " "); print $2; exit(0); }' config/api_config.toml)
BENCH_BASE_URL=http://localhost:${BENCH_PORT}
BENCH_API_BASE_URL=${BENCH_BASE_URL}/api/v1
BENCH_CTL_URL=${BENCH_BASE_URL}/ctl

LLAMA_PORT=${LLAMA_PORT:-8080}
LLAMA_BASE_URL=http://localhost:${LLAMA_PORT}
LLAMA_API_BASE_URL=${LLAMA_BASE_URL}/v1


STATELESS=${STATELESS:-null}
THINK=${THINK:-null}
ELIDE_THINK=${ELIDE_THINK:-null}
FLASH=${FLASH:-null}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0}
MODEL=${MODEL:-default}
STREAM=${STREAM:-false}
DEBUG_POST=${DEBUG_POST:-false}
DEBUG_RESPONSE=${DEBUG_RESPONSE:-false}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-32}
DIFFUSION_BLOCK_LENGTH=${DIFFUSION_BLOCK_LENGTH:-64}
# diffusion inference context size
DIFFUSION_TOKENS=${DIFFUSION_TOKENS:-128}
# value of 99 forces llama-diffusion-cli to do all computation on GPU
LLAMA_DIFFUSION_NGL=${LLAMA_DIFUSE_NGL:-99}
# llama-diffusion-cli microbatch size (we don't implement microbatching yet)
LLAMA_DIFFUSION_UB=${LLAMA_DIFUSE_UB:-512}
FORCE_DIFFUSION_CLI=${FORCE_DIFFUSION_CLI:-false}
USE_RLB_GEN=${USE_RLB_GEN:-false}
RLB_PREFILL=${RLB_PREFILL:-false}
RLB_ALPHA=${RLB_ALPHA:-auto}
RLB_HALT_RULE=${RLB_HALT_RULE:-}
# RLB_TERMINAL_HALT_RULE: separate halt rule for the terminal block only; empty
# = reuse RLB_HALT_RULE for every block. Lets a run use a Tier 2 convergence
# rule (e.g. dH_threshold) on upstream blocks while a Tier 1 logit-shape rule
# runs on the terminal block where lm_head output is actually meaningful.
RLB_TERMINAL_HALT_RULE=${RLB_TERMINAL_HALT_RULE:-}
RLB_MAG_NORM=${RLB_MAG_NORM:-false}
# Collect arch names that declare generation = "diffusion" from the arch TOMLs.
DIFFUSION_ARCH_NAMES=$(grep -rl 'generation\s*=\s*"diffusion"' models/arch/*.arch.toml 2>/dev/null \
  | sed 's|.*/||; s|\.arch\.toml||' | tr '\n' ' ')
DIFFUSION_ARCH_NAMES=${DIFFUSION_ARCH_NAMES% }

PYTHON=$(which python3 2> /dev/null) || \
  PYTHON=$(which python 2>/dev/null) || \
  { echo "neither python3 nor python found. python3 required (sorry)." 1>&2; exit 1; }


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

MODEL_LIST=()
FORCE_MODEL_LIST=${FORCE_MODEL_LIST:-}
FORCE_MODEL_LIST=(${FORCE_MODEL_LIST//,/ })

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
  ctok = usage.get("completion_tokens", 0)
  ttok = usage.get("thinking_tokens", 0)
  stats = f"itps:{itps:.2f}, otps:{otps:.2f}, ttps:{ttps:.2f}, s:{secs:.2f}"
  if ctok > 0:
    stats += f", otok:{ctok}, think:{ttok}"
  print(f"{resp_text} {{{stats}}}")
  printed = True
  logprobs = resp_all.get("choices", [])
  if logprobs := (logprobs[0] if logprobs else {}).get("logprobs", {}):
    model_type = "diffusion" if os.getenv("IS_DIFFUSION", "false") in ["true", "1"] else "autoregression"
    logprobs = json.dumps(logprobs["content"][0]["top_logprobs"], sort_keys=True)
    model = os.getenv("MODEL", "<NO-MODEL>")
    print(f"{model}|{model_type}|{logprobs}|{resp_text}")

if (not printed) or (os.getenv("DEBUG_RESPONSE", "false") in ["true", "1"]):
  print(json.dumps(resp_all, indent=2) if resp_all else resp_json if resp_json.strip() else "<NO-API-RESPONSE>")
' 2> /dev/null
}

function is_diffusion_model() {
  # mac bash is too old to have ${var,,} tolower syntax
  local model_name=$(awk "BEGIN{print tolower(\"${1}\"); exit(0);}")
  if ${FORCE_DIFFUSION_CLI}; then
    return 0
  fi
  local name
  for name in ${DIFFUSION_ARCH_NAMES}; do
    [[ "${model_name}" == *"${name}"* ]] && return 0;
  done
  return 1
}

function filter_diffusion_cli_output() {
  ! ${DEBUG_RESPONSE} || { cat; return 0; }
  awk '
    /^load_backend\: loaded/,/^total time\: /  {next; }
    /^~|^ggml_metal_free\: / { next; }
    /^[ \t]*$/ { if(printing != 1) next; }
    /.*/ { print $0; printing=1; }
  '
}

function query_one_diffusion_cli() {
  local msg="${1}"

  local model_path="models/${MODEL}.gguf"
  [[ -f "${model_path}" ]] || model_path="models/${MODEL}.st"
  [[ -d "${model_path}" ]] || { echo "[ERR] model ${MODEL} not found as .gguf or .st" 1>&2; return 1; }

  local cli_args=(-m "${model_path}" -p "${msg}")
  cli_args+=(-n "${DIFFUSION_TOKENS}")
  cli_args+=(-ngl "${LLAMA_DIFFUSION_NGL}")
  cli_args+=(-ub "${LLAMA_DIFFUSION_UB}")
  # context size
  cli_args+=(-c "${DIFFUSION_TOKENS}")
  cli_args+=(--diffusion-steps "${DIFFUSION_STEPS}")
  cli_args+=(--diffusion-block-length "${DIFFUSION_BLOCK_LENGTH}")
  cli_args+=(--temp "${TEMPERATURE}")
  [[ "${FLASH}" == "true" ]] && cli_args+=(-fa on) || cli_args+=(-fa off)

  ${DEBUG_POST} && echo "[DBG] llama-diffusion-cli ${cli_args[*]}" || true
  llama-diffusion-cli "${cli_args[@]}" 2>&1 | filter_diffusion_cli_output
  if ${LOGPROBS}; then
    local out=$(llama-diffusion-cli "${cli_args[@]}" 2>&1 | filter_diffusion_cli_output)
    echo "${out}"
    echo "${MODEL}|diffusion|<logprob not supported>|${out}"
  else
    llama-diffusion-cli "${cli_args[@]}" 2>&1 | filter_diffusion_cli_output
  fi
}

function query_one() {
  local msg="${1}"
  local ext_params
  local is_diffusion=false
  # alias global locally so we can change it for diffusion models
  local MAX_TOKENS=${MAX_TOKENS}
  is_diffusion_model "${MODEL}" && is_diffusion=true || true

  if ${USE_LLAMA} && ${is_diffusion}; then
    query_one_diffusion_cli "${msg}"
    return
  fi

  # chat_template_kwargs is the llama.cpp-compatible shape for template
  # variables like enable_thinking; both bench and llama-server accept it.
  if [[ ${THINK} != null ]]; then
    ext_params+=$(printf '
      "chat_template_kwargs": { "enable_thinking": %s },' \
    "${THINK}")
  fi

  if ! ${USE_LLAMA}; then
    local bench_params

    [[ -n "${STATELESS}" && "${STATELESS}" != "null" ]] && bench_params+="\"stateless\": ${STATELESS},"
    [[ -n "${ELIDE_THINK}" && "${ELIDE_THINK}" != "null" ]] && bench_params+="\"elide_thinking\": ${ELIDE_THINK},"
    [[ -n "${FLASH}" && "${FLASH}" != "null" ]] && bench_params+="\"flash_attention\": ${FLASH},"

    if ${is_diffusion}; then
      local diffusion_inner
      diffusion_inner+="\"steps\": ${DIFFUSION_STEPS},"
      diffusion_inner+="\"block_length\": ${DIFFUSION_BLOCK_LENGTH},"
      bench_params+="\"diffusion\": {${diffusion_inner%,}},"
      # local alias only
      MAX_TOKENS=${DIFFUSION_TOKENS}
    fi

    if ${USE_RLB_GEN}; then
      # special value -1.0 used by generate_rlb to denote "auto" mode
      # it's a bit of a hack but for test code it made more sense than
      # supporting string-or-float values
      local auto_alpha=-1.0
      bench_params+="\"use_rlb_gen\": true,"
      [[ -n "${RLB_PREFILL}" ]] && \
        bench_params+="\"enable_rlb_on_prefill\": ${RLB_PREFILL},"
      # RLB_ALPHA="auto" is the human-facing sentinel for "use the halt rule's
      # confidence-derived alpha per iter"; the server expects 0.0 for that
      # mode. Translate at payload-construction time — keep RLB_ALPHA itself
      # untouched so subsequent sweep logic / logs see the word "auto".
      local rlb_alpha="${RLB_ALPHA}"
      [[ "${rlb_alpha}" == "auto" ]] && rlb_alpha=${auto_alpha}
      [[ -n "${rlb_alpha}" ]] && bench_params+="\"rlb_alpha\": ${rlb_alpha},"
      [[ -n "${RLB_HALT_RULE}" ]] && bench_params+="\"rlb_halt_rule\": \"${RLB_HALT_RULE}\","
      # Terminal halt rule is optional — only include when set, so the server
      # sees an empty string and reuses RLB_HALT_RULE for every block.
      [[ -n "${RLB_TERMINAL_HALT_RULE}" ]] && \
        bench_params+="\"rlb_terminal_halt_rule\": \"${RLB_TERMINAL_HALT_RULE}\","
      [[ "${RLB_MAG_NORM}" == "true" ]] && \
        bench_params+="\"rlb_magnitude_norm\": true,"
    fi

    if [[ -n "${bench_params}" ]]; then
      ext_params+="\"bench_custom\": {${bench_params%,}},"
    fi
  fi

  if ${LOGPROBS}; then
    ext_params+=$(printf '
      "logprobs": true,
      "top_logprobs": %s,' \
    "${TOP_LOGPROBS}" \
    )
  fi

  local payload=$(printf '{
      "messages": [{"role":"user","content":"%s"}],
      %s
      "model": "%s",
      "max_tokens": %s,
      "temperature": %s,
      "stream": %s
    }' \
    "${msg}" \
    "${ext_params%}" \
    "${MODEL}" \
    "${MAX_TOKENS}" \
    "${TEMPERATURE}" \
    "${STREAM}" \
    )
  local _payload
  _payload=$(python -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read()), indent=2))' <<< "${payload%}") || {
    echo "${payload}"
    return 1
  }
  payload=${_payload} && unset _payload
  local completions_url=${API_BASE_URL}/chat/completions
  ${DEBUG_POST} && echo "[DBG] ${completions_url} POST payload: ${payload}" || true
  local resp=$(curl -s -X POST "${completions_url}" \
    -H 'Content-Type: application/json' \
    -d "${payload}")
  echo "${resp}" | IS_DIFFUSION=${is_diffusion} parse_response || echo "${resp}"
}

function query() {
  local model
  if ${ALL_MODELS}; then
    for model in "${MODEL_LIST[@]}"; do
      echo "${model} <-- \"${*}\""
      MODEL=${model} query_one "${@}"
      # llama server tries to hold all models in memory and fails
      # there's a ggml or metal driver bug that causes cummulative state resets.
      # this issue was investigated exhaustively and was proven to reside outside of this code base.
      # to produce incorrect results; cycle server to ensure a clean metal context.
      ! ${FORCE_NEW_SERVER} || cycle_server
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
/all-models: toggle all-models mode (currently: ${ALL_MODELS})
/cls: clear the screen
/debug-post: toggle display of post json (currently: ${DEBUG_POST})
/debug-response: toggle display of response json (currently: ${DEBUG_RESPONSE})
/diffusion-block-length [length]: show or set DIFFUSION_BLOCK_LENGTH (currently: ${DIFFUSION_BLOCK_LENGTH})
/diffusion-steps [steps]: show or set DIFFUSION_STEPS (currently: ${DIFFUSION_STEPS})
/diffusion-tokens [count]: show or set DIFFUSION_TOKENS (currently: ${DIFFUSION_TOKENS})
/elide-think: toggle elision of thinking output (currently: ${ELIDE_THINK})
/flash: toggle flash attention mode (currently: ${FLASH})
/help: show this help message
/llama: toggle between using llama-server and bench serve-api. (currently: USE_LLAMA=${USE_LLAMA})
/max-tokens [token_count]: show or set MAX_TOKENS (currently: ${MAX_TOKENS})
/model [index]: show or set current model (currently: ${MODEL})
/new-server: shuts down old server and starts a new one.
/prefer-st: toggle prefer safetensors mode (currently: ${PREFER_ST})
/rlb-alpha <0-1>|auto: set RLB SSM state alpha blending factor (currently: ${RLB_ALPHA})
/rlb-gen: toggle RLB generation mode (currently: ${USE_RLB_GEN})
/rlb-halt-rule [name]: set RLB halt rule (currently: ${RLB_HALT_RULE})
/rlb-mag-norm: toggle RLB magnitude normalization (currently: ${RLB_MAG_NORM})
/rlb-prefill: toggle RLB during prefill (currently: ${RLB_PREFILL})
/rlb-terminal-halt-rule [name|-]: set RLB halt rule for terminal block only; "-" to clear (currently: ${RLB_TERMINAL_HALT_RULE:-<reuse halt-rule>})
/stateless: toggle stateless mode (currently: ${STATELESS})
/temperature [temperature]: show or set TEMPERATURE (currently: ${TEMPERATURE})
/think: toggle think mode (currently: ${THINK})
/quit: exit loop mode

__EOF

  ! ${USE_LLAMA} || {
    cat << ____EOF
Note: with USE_LLAMA=true, diffusion models (${DIFFUSION_ARCH_NAMES}) invoke llama-diffusion-cli
fresh for each prompt — no persistent server, so each query loads the model from scratch.

____EOF
  }
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

function force_cycle_server() {
  quit_api_server
  quit_llama_server
  if ${USE_LLAMA}; then
    start_llama_server && echo "new llama server started." || echo "failed to start llama server."
  else
    start_api_server && echo "new api server started." || echo "failed to start api server."
  fi
}

function loop_mode() {
  local line
  loop_help
  while true; do
    read -e -p "> " line || break
    line=${line% }
    [[ -n "${line}" ]] && history -s "${line}"
    case "${line}" in
      /all-models)
        ALL_MODELS=$(toggle "${ALL_MODELS}")
        echo "ALL_MODELS=${ALL_MODELS}"
        continue;;
      /cls)
        clear
        continue;;
      /new-server)
        force_cycle_server
        continue;;
      /llama)
        USE_LLAMA=$(toggle "${USE_LLAMA}")
        echo "USE_LLAMA=${USE_LLAMA}"
        force_cycle_server
        continue;;
      /help|/h|/|/\?)
        loop_help
        continue;;
      /model*|/m)
        local MI=$(set ${line}; echo ${2})
        if [[ "${MI:-0}" -gt 0 ]]; then
          MODEL=${MODEL_LIST[$((MI - 1))]}
        else
          update_model_list
          show_models
        fi
        echo "MODEL=${MODEL}"
        continue;;
      /prefer-st)
        PREFER_ST=$(toggle "${PREFER_ST}")
        echo "PREFER_ST=${PREFER_ST}"
        force_cycle_server
        continue;;
      /flash)
        FLASH=$(rotate "${FLASH}")
        echo "FLASH=${FLASH}"
        continue;;
      /stateless)
        STATELESS=$(rotate "${STATELESS}")
        echo "STATELESS=${STATELESS}"
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
      /diffusion-steps*)
          local DS=$(set ${line}; echo ${2})
          if [[ "${DS:-0}" -gt 0 ]]; then
            DIFFUSION_STEPS=${DS}
          fi
          echo "DIFFUSION_STEPS=${DIFFUSION_STEPS}"
          continue;;
      /diffusion-block-length*)
          local DBL=$(set ${line}; echo ${2})
          if [[ "${DBL:-0}" -gt 0 ]]; then
            DIFFUSION_BLOCK_LENGTH=${DBL}
          fi
          echo "DIFFUSION_BLOCK_LENGTH=${DIFFUSION_BLOCK_LENGTH}"
          continue;;
      /diffusion-tokens*)
          local DT=$(set ${line}; echo ${2})
          if [[ "${DT:-0}" -gt 0 ]]; then
            DIFFUSION_TOKENS=${DT}
          fi
          echo "DIFFUSION_TOKENS=${DIFFUSION_TOKENS}"
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
      /rlb-gen)
        USE_RLB_GEN=$(toggle "${USE_RLB_GEN}")
        echo "USE_RLB_GEN=${USE_RLB_GEN}"
        continue;;
      /rlb-mag-norm)
        RLB_MAG_NORM=$(toggle "${RLB_MAG_NORM}")
        echo "RLB_MAG_NORM=${RLB_MAG_NORM}"
        continue;;
      /rlb-prefill)
        RLB_PREFILL=$(toggle "${RLB_PREFILL}")
        echo "RLB_PREFILL=${RLB_PREFILL}"
        continue;;
      /rlb-alpha*)
        local RA
        RA=$(set ${line}; echo ${2})
        if [[ "${RA}" == "auto" ]] || echo "${RA:-1}" | awk '{exit !($1 >= 0 && $1 <= 1)}'; then
          RLB_ALPHA=${RA}
        fi
        echo "RLB_ALPHA=${RLB_ALPHA}"
        continue;;
      /rlb-terminal-halt-rule*)
        # Must precede /rlb-halt-rule* in the case list — bash case-match is
        # first-match, and /rlb-halt-rule* prefixes /rlb-terminal-halt-rule.
        local THR
        THR=$(set ${line}; echo ${2})
        case "${THR}" in
          "") ;; # no arg: just print current
          -) RLB_TERMINAL_HALT_RULE= ;; # explicit clear → reuse main rule
          *) RLB_TERMINAL_HALT_RULE="${THR}" ;;
        esac
        echo "RLB_TERMINAL_HALT_RULE=${RLB_TERMINAL_HALT_RULE:-<reuse halt-rule>}"
        continue;;
      /rlb-halt-rule*)
        local HR
        HR=$(set ${line}; echo ${2})
        if [[ -n "${HR}" ]]; then
          RLB_HALT_RULE="${HR}"
        fi
        echo "RLB_HALT_RULE=${RLB_HALT_RULE}"
        continue;;
      quit|exit)
        break;;
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

function query_models() {
  curl -sf "${API_BASE_URL}/models"
}

function update_model_list() {
  local model
  local models
  local use_api=false

  (${LOGPROBS} || ${USE_LLAMA}) || use_api=true

  # if using llama (which publishes models besides what's in models/
  # or trying to pull logits for comparison, ensure consistent model order
  if ${use_api}; then
    models=$(query_models) || return 1
    models=$(parse_models <<< "${models}")
  else
    models=$(find models \( -iname \*.gguf -o -iname \*.st \) -print)
  fi

  models=${models//models\//}
  models=${models//\.gguf/}
  models=${models//\.st/}
  models=$(sort -u <<< "${models}")

  MODEL_LIST=()

  if ${ALL_MODELS_NO_DIFFUSION}; then
    for model in ${models}; do
      is_diffusion_model "${model}" || MODEL_LIST+=("${model}")
    done
  else
    for model in ${models}; do
      MODEL_LIST+=("${model}")
    done
  fi
  [[ -n "${FORCE_MODEL_LIST[*]}" ]] && MODEL_LIST=("${FORCE_MODEL_LIST[@]}")
  [[ ${MODEL} == "default" ]] && MODEL=${MODEL_LIST[0]}
}

function show_models() {
  local model
  local i
  echo "models"
  echo "======"
  i=1
  for model in "${MODEL_LIST[@]}"; do
    [[ "${model}" == "${MODEL}" ]] && echo "${i}) ${model}*" || echo "${i}) ${model}"
    i=$((i + 1))
  done
  echo ""
}

function wait_for_starting_server() {
  # Wait for server to be ready (up to 30s)
  for _i in $(seq 1 60); do
    if query_models > /dev/null; then
      update_model_list
      return 0
    fi
    sleep 0.5
  done
  echo "failed getting models from ${API_BASE_URL}/models" 1>&2
  return 1
}

function start_api_server() {
  [[ -x "./bin/bench" ]] || { echo "./bin/bench binary missing. 'make all' first." 1>&2; return 1; }
  local prefer_st_arg=""
  [[ ${PREFER_ST} == true ]] && prefer_st_arg="--prefer-st"
  API_BASE_URL=${BENCH_API_BASE_URL}
  ./bin/bench serve-api --log ${LOG:-"bin/test_inference.log"} --log-level NONE ${prefer_st_arg} &
  SERVER_PID=$!
  wait_for_starting_server
}

function run_llama_server() {
  llama-server --models-dir "$(pwd)/models" --port ${LLAMA_PORT} --ctx-size 8192 2>&1
}

function start_llama_server() {
  (which llama-server 2>&1) > /dev/null || { echo "llama-server not installed." 1>&2; return 1; }
  API_BASE_URL=${LLAMA_API_BASE_URL}
  run_llama_server >> ${LOG:-"bin/test_inference_llama.log"} &
  LLAMA_PID=$!
  wait_for_starting_server
}

function quit_api_server() {
  (curl -s "${BENCH_CTL_URL}/?quit&now" > /dev/null && sleep 1) || {
    ps aux | awk '/awk/ { next; } /bin\/bench/{ print $2; }' | xargs kill
  }
  SERVER_PID=
}

function quit_llama_server() {
  # blunt instrument
  ps aux | awk '/awk/ { next; } /llama-server/ { print $2; }' | xargs kill
  LLAMA_PID=
}

function quit_server() {
  [[ -n "${SERVER_PID}" ]] && quit_api_server || true
  [[ -n "${LLAMA_PID}" ]] && quit_llama_server || true
}

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
else
  ${USE_LLAMA} && API_BASE_URL=${LLAMA_API_BASE_URL} || API_BASE_URL=${BENCH_API_BASE_URL}
  update_model_list
fi

if ${LOOP_MODE}; then
  loop_mode
else
  query "${MSG}"
fi
