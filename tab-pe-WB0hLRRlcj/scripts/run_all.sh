#!/bin/bash
# Sequential driver for the remaining Tab-PE reproduction runs (Claims 1, 2, 3, 4).
# Run one at a time to avoid CPU oversubscription with the AIM baseline job already in flight.
set -uo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "=== $(date) waiting for AIM artificial_characters smoketest job to finish (if still running) ==="
while pgrep -f "aim_baseline.py --dataset artificial_characters" > /dev/null; do
  sleep 30
done

echo "=== $(date) artificial_characters.py (Tab-PE) ==="
python3 -u artificial_characters.py

echo "=== $(date) person_activity.py (Tab-PE) ==="
python3 -u person_activity.py

for f in 1 2 3 4 5; do
  echo "=== $(date) xor_stress_test_xgb.py --num-features $f ==="
  python3 -u xor_stress_test_xgb.py --num-features "$f"
done

echo "=== $(date) aim_baseline.py --dataset person_activity ==="
python3 -u aim_baseline.py --dataset person_activity

echo "=== $(date) ALL_RUNS_DONE ==="
