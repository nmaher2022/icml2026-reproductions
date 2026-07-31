#!/usr/bin/env bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
COST=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
INPUT_TOKENS=$(echo "$input" | jq -r '.context_window.current_usage.input_tokens // 0')
OUTPUT_TOKENS=$(echo "$input" | jq -r '.context_window.current_usage.output_tokens // 0')
CACHE_CREATE=$(echo "$input" | jq -r '.context_window.current_usage.cache_creation_input_tokens // 0')
CACHE_READ=$(echo "$input" | jq -r '.context_window.current_usage.cache_read_input_tokens // 0')
CTX_SIZE=$(echo "$input" | jq -r '.context_window.context_window_size // 200000')

# Color codes
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
RESET='\033[0m'

# Context color based on thresholds (ccusage style)
if [ "$PCT" -ge 80 ]; then
  CTX_COLOR="$RED"
elif [ "$PCT" -ge 50 ]; then
  CTX_COLOR="$YELLOW"
else
  CTX_COLOR="$GREEN"
fi

# Format cost to 4 decimal places
COST_FMT=$(printf "%.4f" "$COST")

# Format token counts as K
CTX_TOKENS=$(( INPUT_TOKENS + CACHE_CREATE + CACHE_READ ))
CTX_K=$(awk "BEGIN {printf \"%.1f\", $CTX_TOKENS / 1000}")
MAX_K=$(awk "BEGIN {printf \"%.0f\", $CTX_SIZE / 1000}")

# Progress bar (20 chars wide)
BAR_WIDTH=20
FILLED=$(( PCT * BAR_WIDTH / 100 ))
EMPTY=$(( BAR_WIDTH - FILLED ))
BAR=""
for ((i=0; i<FILLED; i++)); do BAR+="█"; done
for ((i=0; i<EMPTY; i++)); do BAR+="░"; done

echo -e "🧠 ${CTX_COLOR}${PCT}%${RESET} ${DIM}[${BAR}]${RESET} ${CTX_K}K/${MAX_K}K  💰 ${CYAN}\$${COST_FMT}${RESET}  ${DIM}${MODEL}${RESET}"
