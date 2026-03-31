#!/usr/bin/env bash
# THIS_SCRIPT=$(basename "$0")
THIS_DIR=$(dirname "$0")
set -e

cd "${THIS_DIR}"

# iterate over all ggufs in models/
export ALL_MODELS=true
# keep max tokens small - we want quick answers
export MAX_TOKENS=250
# no thinking, keep it quick
export THINK=false
# get the logits for comparison
export TOP_LOGPROBS=1

# use the same prompt for llama and bench api-serve
export PROMPT=${PROMPT:-"in one numeral answer 3+1=?"}

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
    ref = v[0]
    check = v[1]
    delta = check - ref
    result = "pass" if math.fabs(delta) <= pass_thresh else "FAIL"
    print(f"model:{k} ref:{ref:.5f} chk:{check:.5f} diff:{delta:.5f} result:{result}")
'
}

echo "Collecting reference results..."
LLAMA_PROBS=$(USE_LLAMA=true ./test_inference.sh "${PROMPT}" 2> /dev/null | grep logprob)

echo "Collecting lab-bench results..."
BENCH_PROBS=$(FORCE_NEW_SERVER=true ./test_inference.sh "${PROMPT}" | grep logprob)

RESULTS=$((echo "${LLAMA_PROBS}"; echo "${BENCH_PROBS}") | compare_logprobs)

echo "RESULTS:"
echo "${RESULTS}"
echo ""

[[ "${RESULTS}" == *"FAIL"* ]] && { echo "FAIL: one or more results failed"; exit 1; }
echo "All checked models passed llama-serve inference equivalence test"
exit 0
