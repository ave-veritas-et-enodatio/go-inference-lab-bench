#!/usr/bin/env bash
THIS_SCRIPT=$(basename "$0")
THIS_DIR=$(dirname "$0")
set -e

cd "${THIS_DIR}"

USE_LLAMA=false
STATELESS=false
EQUIV=${1:-llama}

case "${EQUIV}" in
*llama) USE_LLAMA=true;;
*stateless) STATELESS=true;;
*)
  echo "Usage: ${THIS_SCRIPT} [llama|stateless]" 1>&2
  echo "unknown arg: ${EQUIV}" 1>&2
  exit 1;;
esac

# iterate over all ggufs in models/
[[ -n "${MODEL}" ]] || export ALL_MODELS=true
# keep max tokens small - we want quick answers
export MAX_TOKENS=250
# no thinking, keep it quick
export THINK=false
# get the logits for comparison
export TOP_LOGPROBS=1

# use the same prompt for llama and bench api-serve
export PROMPT=${PROMPT:-"in one numeral answer 3+1=?"}

# override log file location
export LOG="./bin/test_${EQUIV}_equiv.log"

rm -f "${LOG}" 2> /dev/null

PYTHON=$(which python3 2> /dev/null) || \
  PYTHON=$(which python 2>/dev/null) || \
  { echo "python not found." 1>&2; exit 1;}

function compare_logprobs() {
  "${PYTHON}" -c '
import json,math
pass_thresh = 0.00025
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
    prob_data = line[1]
    ptable.setdefault(model, []).append(json.loads(prob_data)[0]["logprob"])
for k, v in ptable.items():
    ref = v[0] if len(v) > 0 else math.nan
    check = v[1] if len(v) > 1 else math.nan
    delta = check - ref
    result = "pass" if math.fabs(delta) <= pass_thresh else "FAIL"
    print(f"model:{k} ref:{ref:.5f} chk:{check:.5f} diff:{delta:.5f} result:{result}")
'
}

echo "Collecting reference ${EQUIV} results..."
REF_PROBS=$(USE_LLAMA=${USE_LLAMA} STATELESS=${STATELESS} FORCE_NEW_SERVER=${STATELESS} \
  ./test_inference.sh "${PROMPT}" 2> /dev/null | grep logprob)

${USE_LLAMA} || {
  STATELESS_CHECK0=$(grep 'stateless=true' "${LOG}")
  [[ -n "${STATELESS_CHECK0}" ]] || {
    echo "FAIL: stateless false when it should be true" 1>&2
    exit 1
  }
}

echo "Collecting lab-bench results..."
BENCH_PROBS=$(FORCE_NEW_SERVER=true STATELESS=false \
  ./test_inference.sh "${PROMPT}" | grep logprob)

${USE_LLAMA} || {
  STATELESS_CHECK1=$(grep 'stateless=true' "${LOG}")
  # there should be no new stateless lines
  [[ "${STATELESS_CHECK0}" == "${STATELESS_CHECK1}" ]] || {
    echo "FAIL: stateless true when it should be false" 1>&2; exit 1;
  }
}

RESULTS=$( (echo "${REF_PROBS}"; echo "${BENCH_PROBS}") | compare_logprobs)

echo "RESULTS:"
echo "${RESULTS}"
echo ""

[[ "${RESULTS}" == *"FAIL"* ]] && { echo "FAIL: one or more results failed"; exit 1; }
echo "All checked models passed ${EQUIV} inference equivalence test"
exit 0
