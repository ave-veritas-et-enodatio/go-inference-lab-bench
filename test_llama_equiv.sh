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


# llama-serve
LLAMA_PROBS=$(USE_LLAMA=true ./test_inference.sh "${PROMPT}" 2> /dev/null | grep logprobs)
echo "LLAMA_PROBS:"
echo "${LLAMA_PROBS}"

# bench serve-api
BENCH_PROBS=$(FORCE_NEW_SERVER=true ./test_inference.sh "${PROMPT}" | grep logprobs)
echo "BENCH_PROBS:"
echo "${BENCH_PROBS}"
