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
  grep logprob <<< "${all_out}"
}

echo "Collecting reference ${EQUIV} results..."

case "${EQUIV}" in
  llama)
    # llama-diffusion-cli does not support extracting logprobs from diffusion models
    export ALL_MODELS_NO_DIFFUSION=true
    REF_PROBS=$(USE_LLAMA=true collect_logprobs)
    ;;
  stateless)
    # stateless vs cached has no meaning for diffusion models
    export ALL_MODELS_NO_DIFFUSION=true
    REF_PROBS=$(STATELESS=true collect_logprobs)
      STATELESS_CHECK0=$(grep 'stateless=true' "${LOG}")
      [[ -n "${STATELESS_CHECK0}" ]] || {
        echo "FAIL: stateless false when it should be true" 1>&2
        exit 1
      }
    ;;
  standard-attention)
    REF_PROBS=$(FLASH=false collect_logprobs)
      FLASH_CHECK0=$(grep 'flash_attention_used=false' "${LOG}")
      [[ -n "${FLASH_CHECK0}" ]] || {
        echo "FAIL: flash_attention true when it should be false" 1>&2
        exit 1
      }
    ;;
  gguf-st)
    unset ALL_MODELS
    MODEL_ARG=${MODEL:-$(find models \( -name \*.st -o -name \*.st.bin \) -maxdepth 1 | sort | head -1)}
    [[ ! -n ${MODEL} && ! -n ${MODEL_ARG} ]] && {
      echo "No safetensor models present to test. skipping."
      exit 0
    }
    # trim potential leading ./ and trailing /
    MODEL_NAME=${MODEL_ARG/\.\//}
    MODEL_NAME=${MODEL_NAME%/}
    MODEL_NAME=${MODEL_NAME/models\//}
    MODEL_NAME=${MODEL_NAME%.bin}
    MODEL_NAME=${MODEL_NAME%.gguf}
    MODEL_NAME=${MODEL_NAME%.st}
    MODEL_ST=models/${MODEL_NAME}.st
    MODEL_ST_BIN=${MODEL_ST}.bin
    MODEL_GGUF=models/${MODEL_NAME}.gguf
    MODEL_GGUF_BIN=${MODEL_GGUF}.bin
    MODEL_GGUF_BIN_RENAMED=false
    MODEL_GGUF_HIDE=${MODEL_GGUF}.$$.bin

    ([[ -f "${MODEL_ST}/config.json" || -f "${MODEL_ST_BIN}/config.json" ]] && [[ -f "${MODEL_GGUF}" || -f "${MODEL_GGUF_BIN}" ]]) || {
      echo "model: ${MODEL_ARG}" 1>&2
      [[ -f "${MODEL_ST}/config.json" ]] || echo "${MODEL_ST} is not safetensors directory" 1>&2
      [[ -f "${MODEL_GGUF}" || -f "${MODEL_GGUF_BIN}" ]] || echo "${MODEL_GGUF} or ${MODEL_GGUF_BIN} must exist" 1>&2
      exit 1
    }
    [[ -f "${MODEL_ST}/tokenizer.gguf" || -f "${MODEL_ST_BIN}/tokenizer.gguf" ]] || {
      echo "error: ${MODEL_ST}/tokenizer.gguf missing. 'make ${MODEL_ST}/tokenizer.gguf' first." 1>&2
      exit 1
    }

    [[ -f "${MODEL_GGUF}" || -f "${MODEL_GGUF_BIN}" ]] || {
      echo " models/${MODEL_NAME}.gguf does not exist. create one with 'make models/${MODEL_NAME}.gguf' first."
      exit 1
    }

    # ensure a gguf
    [[ -f "${MODEL_GGUF_BIN}" && ! -f "${MODEL_GGUF}" ]] && {
      MODEL_GGUF_BIN_RENAMED=true
      mv "${MODEL_GGUF_BIN}" "${MODEL_GGUF}"
    }
    [[ -f ${MODEL_GGUF} ]] || {
      echo "internal error: the ${MODEL_GGUF} should have be visible" 1>&2
      exit 1
    }
    REF_PROBS=$(MODEL=${MODEL_NAME} collect_logprobs)
    # hide the gguf so the server will find the .st
    ${MODEL_GGUF_BIN_RENAMED} && mv "${MODEL_GGUF}" "${MODEL_GGUF_BIN}" || mv "${MODEL_GGUF}" "${MODEL_GGUF_HIDE}"
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

echo "Collecting lab-bench results..."

case "${EQUIV}" in
  llama)
    BENCH_PROBS=$(USE_LLAMA=false collect_logprobs)
    ;;
  stateless)
    BENCH_PROBS=$(STATELESS=false collect_logprobs)
    STATELESS_CHECK1=$(grep 'stateless=true' "${LOG}")
    # there should be no new stateless lines
    [[ "${STATELESS_CHECK1}" == "${STATELESS_CHECK0}" ]] || {
      echo "FAIL: stateless true when it should be false" 1>&2; exit 1;
    }
    ;;
  standard-attention)
    BENCH_PROBS=$(FLASH=true collect_logprobs)
    FLASH_CHECK1=$(grep 'flash_attention_used=false' "${LOG}")
    [[ "${FLASH_CHECK1}" == "${FLASH_CHECK0}" ]] || {
      echo "FAIL: flash false when it should be true" 1>&2; exit 1;
    }
    ;;
  gguf-st)
    [[ ! -f ${MODEL_GGUF} ]] || {
      echo "internal error: the ${MODEL_GGUF} should have been hidden as a .bin file" 1>&2
      exit 1
    }
    # ensure .st
    MODEL_ST_UNHIDDEN=false
    [[ -f "${MODEL_ST}/config.json" ]] || {
      mv "${MODEL_ST_BIN}" "${MODEL_ST}"
      MODEL_ST_UNHIDDEN=true
    }

    BENCH_PROBS=$(MODEL=${MODEL_NAME} collect_logprobs)
    # unhide the gguf if we did the hiding
    [[ -f "${MODEL_GGUF_HIDE}" ]] && mv "${MODEL_GGUF_HIDE}" "${MODEL_GGUF}"
    # re-hide the ST if we did the unhiding
    ${MODEL_ST_UNHIDDEN} && mv "${MODEL_ST}" "${MODEL_ST_BIN}"
    ;;
  *)
    echo "internal error. unhandled case: ${EQUIV}" 1>&2
    exit 1;;
esac

[[ -n "${BENCH_PROBS}" ]] || {
  echo "FAIL: no lab-bench results collected" 1>&2
  exit 1
}

RESULTS=$( (echo "${REF_PROBS}"; echo "${BENCH_PROBS}") | compare_logprobs "${PASS_THRESH}")

echo "RESULTS:"
echo "${RESULTS}"
echo ""

[[ "${RESULTS}" == *"FAIL"* ]] && { echo "FAIL: one or more results failed"; exit 1; }

ERRORS=$(grep '\[ERROR\]' "${LOG}")
[[ -n "${ERRORS}" ]] && {
  echo "FAIL: log contains [ERROR] messages" 1>&2
  echo "${ERRORS}" 1>&2
  exit 1
}

echo "All checked models passed ${EQUIV} inference equivalence test"
exit 0
