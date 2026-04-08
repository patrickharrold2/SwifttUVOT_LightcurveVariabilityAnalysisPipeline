#!/usr/bin/env bash
# set -e intentionally omitted: sub-scripts handle their own errors and log
# failures individually. Aborting the master script on a single obsid failure
# would leave partial outputs that pass the weak size-only validation on rerun.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== PSR B1259 Swift UVOT pipeline ==="
echo "Pipeline dir: $SCRIPT_DIR"
echo

# --- CALDB guard ---
# uvotsource uses apercorr=CURVEOFGROWTH which requires a valid CALDB.
# If these variables are missing, uvotsource either fails silently or applies
# no aperture correction, producing systematically wrong magnitudes.
for var in CALDB CALDBCONFIG CALDBALIAS; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set. Run your HEASoft/CALDB exports first:" >&2
    echo "  conda activate henv" >&2
    echo "  export CALDB=~/heasoft_caldb" >&2
    echo "  export CALDBCONFIG=\$CALDB/caldb.config" >&2
    echo "  export CALDBALIAS=\$CALDB/alias_config.fits" >&2
    echo "  export CALDB_LOCAL=1" >&2
    exit 1
  fi
done

# --- HEASoft / Python command checks ---
for cmd in uvotimsum uvotsource python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found in PATH: $cmd" >&2
    exit 1
  fi
done

# Initialise a fresh log for this run
LOG="$SCRIPT_DIR/../products/pipeline_run_log.txt"
mkdir -p "$(dirname "$LOG")"
{
  echo "# Pipeline run log"
  echo "# Run started: $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
  echo "# This file is overwritten at the start of each pipeline run."
  echo "# Format: STAGE | OBSID | FILTER | RESULT"
  echo "#"
} > "$LOG"

echo "Step 1/4: uvotimsum (stack images)"
"$SCRIPT_DIR/run_uvotimsum.sh"
echo

echo "Step 2/4: uvotsource (photometry)"
"$SCRIPT_DIR/run_uvotsource.sh"
echo

echo "Step 3/4: lightcurve (tables + plots)"
python3 "$SCRIPT_DIR/lightcurve.py"
echo

echo "Step 4/4: variability (chi-squared analysis)"
python3 "$SCRIPT_DIR/variability.py"
echo

echo "=== DONE ==="
echo "Outputs:"
echo "  ../products/summed_images/"
echo "  ../products/uvotsource_results/"
echo "  ../products/tables/"
echo "  ../products/plots/"