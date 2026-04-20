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

echo "Collecting ${EQUIV} referent results..."

COLLECT_LOGPROBS_FAIL=false

case "${EQUIV}" in
  llama)
    # llama-diffusion-cli does not support extracting logprobs from diffusion models
    export ALL_MODELS_NO_DIFFUSION=true
    REF_PROBS=$(USE_LLAMA=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    ;;
  stateless)
    # stateless vs cached has no meaning for diffusion models
    export ALL_MODELS_NO_DIFFUSION=true
    REF_PROBS=$(STATELESS=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    STATELESS_CHECK0=$(grep 'stateless=true' "${LOG}")
    [[ -n "${STATELESS_CHECK0}" ]] || {
      echo "FAIL: stateless false when it should be true" 1>&2
      exit 1
    }
    ;;
  standard-attention)
    REF_PROBS=$(FLASH=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    FLASH_CHECK0=$(grep 'flash_attention_used=false' "${LOG}")
    [[ -n "${FLASH_CHECK0}" ]] || {
      echo "FAIL: flash_attention true when it should be false" 1>&2
      exit 1
    }
    ;;
  gguf-st)
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

    REF_PROBS=$(FORCE_MODEL_LIST="${MODELS[*]}" PREFER_ST=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    GGUF_CHECK0=$(grep 'ModelReader\[gguf\] created' "${LOG}")
    ST_CHECK0=$(grep 'ModelReader\[safetensors\] created' "${LOG}")
    [[ -n "${GGUF_CHECK0}" && -z "${ST_CHECK0}" ]] || {
      echo "FAIL: model should have been loaded as gguf but was loaded as safetensors." 1>&2
      exit 1
    }
    ;;
  *)
    echo "Usage: [MODEL=<model>] ${THIS_SCRIPT} [llama|stateless|standard-attention|gguf-st]" 1>&2
      echo "unknown arg: ${EQUIV}" 1>&2
      exit 1
      ;;
esac

[[ -n "${REF_PROBS}" ]] || {
  echo "FAIL: no reference results collected" 1>&2
  exit 1
}

echo ""
echo "${EQUIV} referent results:"
echo "${REF_PROBS}"

if ${COLLECT_LOGPROBS_FAIL}; then
  exit 1
fi

echo ""
echo "Collecting ${EQUIV} check results..."

case "${EQUIV}" in
  llama)
    CHECK_PROBS=$(USE_LLAMA=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    ;;
  stateless)
    CHECK_PROBS=$(STATELESS=false collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    STATELESS_CHECK1=$(grep 'stateless=true' "${LOG}")
    # there should be no new stateless lines
    [[ "${STATELESS_CHECK1}" == "${STATELESS_CHECK0}" ]] || {
      echo "FAIL: stateless true when it should be false" 1>&2; exit 1;
    }
    ;;
  standard-attention)
    CHECK_PROBS=$(FLASH=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
    FLASH_CHECK1=$(grep 'flash_attention_used=false' "${LOG}")
    [[ "${FLASH_CHECK1}" == "${FLASH_CHECK0}" ]] || {
      echo "FAIL: flash false when it should be true" 1>&2; exit 1;
    }
    ;;
  gguf-st)
      CHECK_PROBS=$(FORCE_MODEL_LIST="${MODELS[*]}" PREFER_ST=true collect_logprobs) ||COLLECT_LOGPROBS_FAIL=true
      GGUF_CHECK1=$(grep 'ModelReader\[gguf\] created' "${LOG}")
      ST_CHECK1=$(grep 'ModelReader\[safetensors\] created' "${LOG}")
      [[ "${GGUF_CHECK1}" == "${GGUF_CHECK0}" && -n "${ST_CHECK1}" ]] || {
        echo "FAIL: model should have been loaded as safetensors but was loaded as gguf." 1>&2
        exit 1
      }
    ;;
  *)
    echo "internal error. unhandled case: ${EQUIV}" 1>&2
    exit 1;;
esac

[[ -n "${CHECK_PROBS}" ]] || {
  echo "FAIL: no lab-bench results collected" 1>&2
  exit 1
}

echo ""
echo "${EQUIV} check results:"
echo "${CHECK_PROBS}"

if ${COLLECT_LOGPROBS_FAIL}; then
  exit 1
fi

RESULTS=$( (echo "${REF_PROBS}" && echo "${CHECK_PROBS}") | compare_logprobs "${PASS_THRESH}" )

echo ""
echo "EQUIV RESULTS:"
echo "${RESULTS}"

echo ""
[[ "${RESULTS}" == *"FAIL"* ]] && { echo "FAIL: one or more results failed"; exit 1; }
ERRORS=$(grep '\[ERROR\]' "${LOG}") && {
  echo "FAIL: log contains [ERROR] messages" 1>&2
  echo "${ERRORS}" 1>&2
  exit 1
}
echo "All checked models passed ${EQUIV} inference equivalence test"
exit 0
