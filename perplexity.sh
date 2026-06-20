#!/usr/bin/env bash
#
# Perplexity benchmark + regression tracker.
#
# Computes int8-engine perplexity (fast) and the HF fp32 reference perplexity
# (slow, cached) on the same prompt, prints both plus the gap, and compares the
# int8 number against a saved baseline so improvements/regressions are visible
# after each bugfix.
#
# Usage:
#   ./perplexity.sh [prompt]     # default prompt: scripts/perplexity_prompt.txt
#   ./perplexity.sh --hf [...]   # force-recompute the HF fp32 reference
#   ./perplexity.sh --save [...] # record current numbers as the new baseline
#   ./perplexity.sh --help
#
# Env overrides:
#   MODEL_MOG   engine model file (default: ./mistral.mog)
#   PYTHON      python interpreter for the HF script (default: python3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_MOG="${MODEL_MOG:-./mistral.mog}"
ENGINE="./build/mistral.cpp"
PYTHON="${PYTHON:-python3}"
HF_SCRIPT="scripts/test/mistral/perplexity.py"
PROMPT_FILE="scripts/perplexity_prompt.txt"
BASELINE="perplexity_baseline.json"

force_hf=0
save_baseline=0
prompt=""

while [ $# -gt 0 ]; do
    case "$1" in
        --hf) force_hf=1 ;;
        --save) save_baseline=1 ;;
        -h|--help)
            sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
base_int8="$(json_get int8_ppl)"

# --- HF fp32 reference (cached unless forced / prompt changed) ---------------
hf_ppl=""
hf_tokens=""
if [ "$force_hf" -eq 0 ] && [ -n "$base_hf" ] && [ "$base_sha" = "$prompt_sha" ]; then
    hf_ppl="$base_hf"
    hf_tokens="$(json_get tokens)"
    echo "HF reference perplexity: $hf_ppl (cached)"
else
    echo "Computing HF reference perplexity (this is slow)..."
    hf_out="$("$PYTHON" "$HF_SCRIPT" "$prompt")"
    hf_ppl="$(printf '%s\n' "$hf_out" | grep -E '^perplexity:' | awk '{print $2}')"
    hf_tokens="$(printf '%s\n' "$hf_out" | grep -E '^tokens:' | awk '{print $2}')"
    [ -n "$hf_ppl" ] || { echo "Failed to parse HF perplexity from:" >&2; echo "$hf_out" >&2; exit 1; }
    echo "HF fp32 perplexity: $hf_ppl"
fi

# --- int8 engine perplexity (fast path) --------------------------------------
echo "Computing int8 engine perplexity..."
eng_out="$("$ENGINE" "$MODEL_MOG" "$prompt" --ppl)"
int8_ppl="$(printf '%s\n' "$eng_out" | grep -E '^perplexity:' | awk '{print $2}')"
int8_tokens="$(printf '%s\n' "$eng_out" | grep -E '^tokens:' | awk '{print $2}')"
[ -n "$int8_ppl" ] || { echo "Failed to parse engine perplexity from:" >&2; echo "$eng_out" >&2; exit 1; }

# --- Report -------------------------------------------------------------------
echo
echo "===================================================="
echo "  perplexity            prompt sha: ${prompt_sha:0:12}"
echo "===================================================="
printf "  HF reference        %s\n" "$hf_ppl"
printf "  int8 engine         %s\n" "$int8_ppl"
if [ -n "$hf_ppl" ] && [ -n "$int8_ppl" ]; then
    awk -v a="$int8_ppl" -v b="$hf_ppl" 'BEGIN{printf "  gap (int8 - HF)     %.4f\n", a-b}'
fi

if [ -n "$hf_tokens" ] && [ -n "$int8_tokens" ] && [ "$hf_tokens" != "$int8_tokens" ]; then
    echo "  WARNING: HF tokenized $hf_tokens tokens but engine tokenized $int8_tokens;"
    echo "           perplexities are not over the identical sequence."
fi

if [ -n "$base_int8" ] && [ "$base_sha" = "$prompt_sha" ]; then
    awk -v cur="$int8_ppl" -v base="$base_int8" 'BEGIN{
        d = cur - base;
        tag = (d < -0.0001) ? "IMPROVED" : (d > 0.0001) ? "REGRESSED" : "unchanged";
        printf "  vs baseline int8    %.4f  (%+.4f, %s)\n", base, d, tag;
    }'
else
    echo "  vs baseline int8    (no baseline for this prompt; use --save to record)"
fi
echo "===================================================="

# --- Save baseline ------------------------------------------------------------
if [ "$save_baseline" -eq 1 ]; then
    cat > "$BASELINE" <<EOF
{
  "prompt_sha": "$prompt_sha",
  "tokens": ${int8_tokens:-0},
  "hf_ppl": $hf_ppl,
  "int8_ppl": $int8_ppl,
  "date": "$(date -u +%Y-%m-%d)"
}
EOF
    echo "Saved baseline to $BASELINE"
fi
