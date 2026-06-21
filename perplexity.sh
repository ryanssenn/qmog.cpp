#!/usr/bin/env bash
#
# Perplexity benchmark + regression tracker.
#
# Computes Q8F16-engine perplexity (fast) and the HF fp32 reference perplexity
# (slow, cached) on the same prompt, prints both plus the gap, and compares the
# engine number against a saved baseline so improvements/regressions are visible
# after each bugfix.
#
# Usage:
#   ./perplexity.sh [prompt]     # default prompt: scripts/perplexity_prompt.txt
#   ./perplexity.sh --check      # CI gate: engine PPL only, exit 1 on regression
#   ./perplexity.sh --hf [...]   # force-recompute the HF fp32 reference
#   ./perplexity.sh --save [...] # record current numbers as the new baseline
#   ./perplexity.sh --help
#
# Env overrides:
#   MODEL_MOG   engine model file (default: ./mistral-7B-Q8F16.mog)
#   PYTHON      python interpreter for the HF script (default: python3)
#   PPL_ATOL    max allowed PPL increase for --check (default: 0.001)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_MOG="${MODEL_MOG:-./mistral-7B-Q8F16.mog}"
ENGINE="./build/qmog-cli"
PYTHON="${PYTHON:-python3}"
HF_SCRIPT="scripts/test/mistral/perplexity.py"
PROMPT_FILE="scripts/perplexity_prompt.txt"
BASELINE="perplexity_baseline.json"
PPL_ATOL="${PPL_ATOL:-0.001}"

force_hf=0
save_baseline=0
check_only=0
prompt=""

while [ $# -gt 0 ]; do
    case "$1" in
        --hf) force_hf=1 ;;
        --save) save_baseline=1 ;;
        --check) check_only=1 ;;
        -h|--help)
            sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) prompt="$1" ;;
    esac
    shift
done

if [ -z "$prompt" ]; then
    [ -f "$PROMPT_FILE" ] || { echo "Missing $PROMPT_FILE and no prompt argument given" >&2; exit 1; }
    prompt="$(cat "$PROMPT_FILE")"
fi
prompt="${prompt%$'\n'}"

[ -x "$ENGINE" ] || { echo "Engine not built: $ENGINE (run: cmake --build build)" >&2; exit 1; }
[ -f "$MODEL_MOG" ] || { echo "Model file not found: $MODEL_MOG" >&2; exit 1; }

prompt_sha="$(printf '%s' "$prompt" | sha1sum | cut -d' ' -f1)"

# Read a flat-JSON numeric/string field from the baseline file.
json_get() {
    [ -f "$BASELINE" ] || return 0
    grep -oE "\"$1\"[[:space:]]*:[[:space:]]*\"?[^,\"}]*" "$BASELINE" 2>/dev/null \
        | head -n1 | sed -E "s/.*:[[:space:]]*\"?//"
}

base_sha="$(json_get prompt_sha)"
base_hf="$(json_get hf_ppl)"
base_q8f16="$(json_get q8f16_ppl)"
if [ -z "$base_q8f16" ]; then
    base_q8f16="$(json_get int8_ppl)"
fi
base_tokens="$(json_get tokens)"

# --- Q8F16 engine perplexity -----------------------------------------------
echo "Computing Q8F16 engine perplexity..."
eng_out="$("$ENGINE" "$MODEL_MOG" "$prompt" --ppl)"
q8f16_ppl="$(printf '%s\n' "$eng_out" | grep -E '^perplexity:' | awk '{print $2}')"
q8f16_tokens="$(printf '%s\n' "$eng_out" | grep -E '^tokens:' | awk '{print $2}')"
[ -n "$q8f16_ppl" ] || { echo "Failed to parse engine perplexity from:" >&2; echo "$eng_out" >&2; exit 1; }

if [ "$check_only" -eq 1 ]; then
    echo "Q8F16 PPL: $q8f16_ppl  tokens: $q8f16_tokens  prompt sha: ${prompt_sha:0:12}"

    if [ -z "$base_q8f16" ]; then
        echo "ERROR: no q8f16_ppl in $BASELINE" >&2
        exit 1
    fi
    if [ "$base_sha" != "$prompt_sha" ]; then
        echo "ERROR: prompt_sha mismatch (baseline ${base_sha:0:12}, got ${prompt_sha:0:12})" >&2
        exit 1
    fi
    if [ -n "$base_tokens" ] && [ "$base_tokens" != "$q8f16_tokens" ]; then
        echo "ERROR: token count mismatch (baseline $base_tokens, got $q8f16_tokens)" >&2
        exit 1
    fi

    awk -v cur="$q8f16_ppl" -v base="$base_q8f16" -v atol="$PPL_ATOL" 'BEGIN {
        delta = cur - base;
        if (delta > atol) {
            printf "REGRESSED: delta=%+.4f (max allowed +%.4f)\n", delta, atol;
            exit 1;
        }
        if (delta < -0.01) {
            printf "IMPROVED: delta=%+.4f — consider ./perplexity.sh --save\n", delta;
        } else {
            printf "OK: delta=%+.4f (baseline %.5f)\n", delta, base;
        }
    }'
    exit $?
fi

# --- HF fp32 reference (cached unless forced / prompt changed) ---------------
hf_ppl=""
hf_tokens=""
if [ "$force_hf" -eq 0 ] && [ -n "$base_hf" ] && [ "$base_sha" = "$prompt_sha" ]; then
    hf_ppl="$base_hf"
    hf_tokens="$base_tokens"
    echo "HF reference perplexity: $hf_ppl (cached)"
else
    echo "Computing HF reference perplexity (this is slow)..."
    hf_out="$("$PYTHON" "$HF_SCRIPT" "$prompt")"
    hf_ppl="$(printf '%s\n' "$hf_out" | grep -E '^perplexity:' | awk '{print $2}')"
    hf_tokens="$(printf '%s\n' "$hf_out" | grep -E '^tokens:' | awk '{print $2}')"
    [ -n "$hf_ppl" ] || { echo "Failed to parse HF perplexity from:" >&2; echo "$hf_out" >&2; exit 1; }
    echo "HF fp32 perplexity: $hf_ppl"
fi

# --- Report -------------------------------------------------------------------
echo
echo "===================================================="
echo "  perplexity            prompt sha: ${prompt_sha:0:12}"
echo "===================================================="
printf "  HF reference        %s\n" "$hf_ppl"
printf "  Q8F16 engine        %s\n" "$q8f16_ppl"
if [ -n "$hf_ppl" ] && [ -n "$q8f16_ppl" ]; then
    awk -v a="$q8f16_ppl" -v b="$hf_ppl" 'BEGIN{printf "  gap (Q8F16 - HF)    %.4f\n", a-b}'
fi

if [ -n "$hf_tokens" ] && [ -n "$q8f16_tokens" ] && [ "$hf_tokens" != "$q8f16_tokens" ]; then
    echo "  WARNING: HF tokenized $hf_tokens tokens but engine tokenized $q8f16_tokens;"
    echo "           perplexities are not over the identical sequence."
fi

if [ -n "$base_q8f16" ] && [ "$base_sha" = "$prompt_sha" ]; then
    awk -v cur="$q8f16_ppl" -v base="$base_q8f16" 'BEGIN{
        d = cur - base;
        tag = (d < -0.0001) ? "IMPROVED" : (d > 0.0001) ? "REGRESSED" : "unchanged";
        printf "  vs baseline Q8F16   %.4f  (%+.4f, %s)\n", base, d, tag;
    }'
else
    echo "  vs baseline Q8F16   (no baseline for this prompt; use --save to record)"
fi
echo "===================================================="

# --- Save baseline ------------------------------------------------------------
if [ "$save_baseline" -eq 1 ]; then
    cat > "$BASELINE" <<EOF
{
  "prompt_sha": "$prompt_sha",
  "tokens": ${q8f16_tokens:-0},
  "hf_ppl": $hf_ppl,
  "q8f16_ppl": $q8f16_ppl,
  "date": "$(date -u +%Y-%m-%d)"
}
EOF
    echo "Saved baseline to $BASELINE"
fi
