#!/usr/bin/env bash
THIS_SCRIPT=$(basename "$0")
THIS_DIR=$(dirname "$0")

cd "${THIS_DIR}"

EQUIV=${1:-llama}

# iterate over all ggufs in models/
[[ -n "${MODEL}" ]] || export ALL_MODELS=true
# keep max tokens small - we want quick answers
export MAX_TOKENS=250
# no thinking, keep it quick
export THINK=false
# always fresh server
export FORCE_NEW_SERVER=true
# get the logits for comparison
export TOP_LOGPROBS=1

# use the same prompt for llama and bench api-serve
export PROMPT=${PROMPT:-"in one numeral answer 3+1=?"}

# override log file location
export LOG="./bin/test_${EQUIV}_equiv.log"

PASS_THRESH=${PASS_THRESH:-0.001}

rm -f "${LOG}" 2> /dev/null

PYTHON=$(which python3 2> /dev/null) || \
  PYTHON=$(which python 2>/dev/null) || \
  { echo "neither python3 nor python not found." 1>&2; exit 1;}

# =============================================================================
# Helpers
# =============================================================================

function compare_logprobs() {
  local pass_thresh="${1}"
  EQUIV=${EQUIV} "${PYTHON}" -c '
import os,sys,json,math
pass_thresh = float(sys.argv[1])
equiv = os.environ.get("EQUIV", "")
ptable = {}
while True:
    try:
      line = input()
    except EOFError:
        break
    line = line.strip()
    if not line:
        continue
    line = line.split("|")
    model = line[0]
    model_type = line[1]
    # normally envar ALL_MODELS_NO_DIFFUSION=true, so this should not get used, but belt and suspenders are good.
    if equiv == "llama" and model_type == "diffusion":
        print(f"skipping diffusion model {model} (llama-diffusion-cli does not support logprob).")
        continue
    prob_data = line[2]
    ptable.setdefault(model, []).append(json.loads(prob_data)[0]["logprob"])
for k, v in ptable.items():
    ref = v[0] if len(v) > 0 else math.nan
    check = v[1] if len(v) > 1 else math.nan
    delta = check - ref
    result = "pass" if math.fabs(delta) <= pass_thresh else "FAIL"
    print(f"model:{k} ref:{ref:.5f} chk:{check:.5f} diff:{delta:.5f} result:{result}")
' "${pass_thresh}"
}

function collect_logprobs() {
  local all_out=$(./test_inference.sh "${PROMPT}" 2>&1)
  grep -v logprob 1>&2 <<< "${all_out}"
  grep logprob <<< "${all_out}" || { echo "FAIL: logprob extraction failed" 1>&2; return 1; }
  if [[ "${all_out}" == *"<NO-MODEL-RESPONSE>"* ]]; then
    echo "FAIL: model did not produce a response" 1>&2
    return 1
  fi
}

# =============================================================================
# Per-mode phases.
#
# Each equivalence mode <name> defines three functions (with '-' mapped to '_'
# in the function name):
#   setup_equiv_<name>  — pre-collection setup (exports, model-list assembly)
#   gather_check_<name> — collect the BENCH (system-under-test) result
#   gather_ref_<name>   — collect the REFERENCE (known-good) result
#
# The driver at the bottom sequences them: setup → check → interstitial → ref
# → finish. Bench is gathered FIRST so a hard bench failure (crash, no response,
# exception) exits immediately, before the (often slow) reference collection;
# a merely-bad numerical value still falls through to the final comparison.
#
# Both collections share one accumulating ${LOG} (removed at startup). A
# gather_check_* assertion therefore sees only its own run's lines (absolute
# presence/absence); a gather_ref_* assertion sees check+ref and confirms the
# reference run ADDED its expected lines without disturbing the check run's.
#
# Cross-function state is global by convention: CHECK_PROBS, REF_PROBS, MODELS,
# ST_CHECK (gguf-st cross-check), COLLECT_LOGPROBS_FAIL. Per-function scratch is
# `local`.
# =============================================================================

COLLECT_LOGPROBS_FAIL=false

# ---- llama: bench vs llama-server (the authoritative reference) -------------
function setup_equiv_llama() {
  # llama-diffusion-cli does not support extracting logprobs from diffusion models
  export ALL_MODELS_NO_DIFFUSION=true
  # llama-server cannot load safetensors (.st), so drop .st models: the llama-mode
  # cross-backend comparison only makes sense for GGUF models both backends can
  # load. .st models are exercised by the gguf-st mode (bench .st vs bench GGUF).
  export GGUF_ONLY=true
}
function gather_check_llama() {
  CHECK_PROBS=$(USE_LLAMA=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
}
function gather_ref_llama() {
  REF_PROBS=$(USE_LLAMA=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
}

# ---- stateless: bench cached vs bench stateless -----------------------------
function setup_equiv_stateless() {
  # stateless vs cached has no meaning for diffusion models
  export ALL_MODELS_NO_DIFFUSION=true
}
function gather_check_stateless() {
  CHECK_PROBS=$(STATELESS=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
  # the cached (non-stateless) check run must NOT have logged any stateless lines
  local stateless_lines=$(grep 'stateless=true' "${LOG}")
  [[ -z "${stateless_lines}" ]] || {
    echo "FAIL: stateless true when it should be false" 1>&2; exit 1;
  }
}
function gather_ref_stateless() {
  REF_PROBS=$(STATELESS=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
  # the stateless reference run must have added stateless lines
  local stateless_lines=$(grep 'stateless=true' "${LOG}")
  [[ -n "${stateless_lines}" ]] || {
    echo "FAIL: stateless false when it should be true" 1>&2; exit 1;
  }
}

# ---- standard-attention: bench flash vs bench non-flash ---------------------
function setup_equiv_standard_attention() {
  : # no pre-collection setup
}
function gather_check_standard_attention() {
  CHECK_PROBS=$(FLASH=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
  # the flash check run must NOT have logged any non-flash lines
  local flash_lines=$(grep 'flash_attention_used=false' "${LOG}")
  [[ -z "${flash_lines}" ]] || {
    echo "FAIL: flash false when it should be true" 1>&2; exit 1;
  }
}
function gather_ref_standard_attention() {
  REF_PROBS=$(FLASH=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
  # the non-flash reference run must have added non-flash lines
  local flash_lines=$(grep 'flash_attention_used=false' "${LOG}")
  [[ -n "${flash_lines}" ]] || {
    echo "FAIL: flash true when it should be false" 1>&2; exit 1;
  }
}

# ---- gguf-st: same model loaded as safetensors vs gguf ----------------------
function setup_equiv_gguf_st() {
  [[ -n "${MODEL}" ]] && {
    M=${MODEL}
    # model was specified explicitly - an incomplete .gguf/.st pair is an error
    [[ -f models/${M}.st/config.json ]] || { echo "${M} is not a safetensor model." 1>&2; exit 1; }
    [[ -f models/${M}.st/tokenizer.gguf ]] || { echo "${M} is missing tokenizer.gguf. ('make st-tok-ggufs')" 1>&2; exit 1; }
    [[ -f models/${M}.gguf ]] || { echo "${M} has no gguf. ('make st-ggufs')" 1>&2; exit 1; }
    MODELS=("${M}")
    unset M
  } || {
    MODELS_ST=$(find models -name \*.st -maxdepth 1 | sort)
    # trim relative directory and extension from elements
    MODELS_ST=${MODELS_ST//models\//}
    MODELS_ST=${MODELS_ST//\.st/}
    MODELS=()
    for M in ${MODELS_ST}; do
      # we're auto-assembling a list of .gguf/.st pairs - an incomplete pair is not an error
      [[ -f models/${M}.st/config.json ]] || { echo "${M} is not a safetensor model. skipping."; continue; }
      [[ -f models/${M}.st/tokenizer.gguf ]] || { echo "${M} is missing tokenizer.gguf. skipping. ('make st-tok-ggufs')"; continue; }
      [[ -f models/${M}.gguf ]] || { echo "${M} has no gguf. skipping. ('make st-ggufs')"; continue; }
      MODELS+=("${M}")
    done
    unset M MODELS_ST
    [[ -n "${MODELS[*]}" ]] || { echo "No .gguf/.st models pairs to test. skipping."; exit 0; }
  }
}
function gather_check_gguf_st() {
  CHECK_PROBS=$(FORCE_MODEL_LIST="${MODELS[*]}" PREFER_ST=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
  local gguf_check=$(grep 'ModelReader\[gguf\] created' "${LOG}")
  ST_CHECK=$(grep 'ModelReader\[safetensors\] created' "${LOG}") # global: re-read by gather_ref_gguf_st
  [[ -z "${gguf_check}" && -n "${ST_CHECK}" ]] || {
    echo "FAIL: model should have been loaded as safetensors but was loaded as gguf." 1>&2
    exit 1
  }
}
function gather_ref_gguf_st() {
  REF_PROBS=$(FORCE_MODEL_LIST="${MODELS[*]}" PREFER_ST=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
  local gguf_ref=$(grep 'ModelReader\[gguf\] created' "${LOG}")
  local st_ref=$(grep 'ModelReader\[safetensors\] created' "${LOG}")
  # the gguf reference run must have added a gguf-created line, and the
  # safetensors lines from the check run must be unchanged (no new st load).
  [[ -n "${gguf_ref}" && "${st_ref}" == "${ST_CHECK}" ]] || {
    echo "FAIL: model should have been loaded as gguf but was loaded as safetensors." 1>&2
    exit 1
  }
}

# ---- common phases ----------------------------------------------------------

# interstitial: between the check and reference collections. Validate the check
# result and print it; a hard bench failure exits here, before the reference is
# collected.
function interstitial() {
  [[ -n "${CHECK_PROBS}" ]] || {
    echo "FAIL: no lab-bench results collected" 1>&2
    exit 1
  }
  if ${COLLECT_LOGPROBS_FAIL}; then
    exit 1
  fi
  echo ""
  echo "${EQUIV} check results:"
  echo "${CHECK_PROBS}"
}

# finish: after the reference collection. Validate the reference result, run the
# comparison, and emit the final verdict.
function finish() {
  [[ -n "${REF_PROBS}" ]] || {
    echo "FAIL: no reference results collected" 1>&2
    exit 1
  }
  if ${COLLECT_LOGPROBS_FAIL}; then
    exit 1
  fi
  echo ""
  echo "${EQUIV} referent results:"
  echo "${REF_PROBS}"

  local results=$( (echo "${REF_PROBS}" && echo "${CHECK_PROBS}") | compare_logprobs "${PASS_THRESH}" )
  echo ""
  echo "EQUIV RESULTS:"
  echo "${results}"

  echo ""
  [[ "${results}" == *"FAIL"* ]] && { echo "FAIL: one or more results failed"; exit 1; }
  local errors=$(grep '\[ERROR\]' "${LOG}")
  [[ -z "${errors}" ]] || {
    echo "FAIL: log contains [ERROR] messages" 1>&2
    echo "${errors}" 1>&2
    exit 1
  }
  echo "All checked models passed ${EQUIV} inference equivalence test"
}

# =============================================================================
# Driver — one place sequences every phase; per-mode logic lives in the
# functions above, dispatched by name ('-' → '_').
# =============================================================================

EQUIV_FN="${EQUIV//-/_}"

case "${EQUIV}" in
  llama|stateless|standard-attention|gguf-st)
    setup_equiv_${EQUIV_FN}
    echo "Collecting ${EQUIV} check results..."
    gather_check_${EQUIV_FN}
    interstitial
    echo ""
    echo "Collecting ${EQUIV} referent results..."
    gather_ref_${EQUIV_FN}
    finish
    ;;
  *)
    echo "Usage: [MODEL=<model>] ${THIS_SCRIPT} [llama|stateless|standard-attention|gguf-st]" 1>&2
    echo "unknown arg: ${EQUIV}" 1>&2
    exit 1
    ;;
esac

exit 0
