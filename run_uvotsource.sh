#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ROOT="$SCRIPT_DIR/../products/summed_images"
OUT="$SCRIPT_DIR/../products/uvotsource_results"
FIELD_OUT="$SCRIPT_DIR/../products/fieldstar_results"
LOG="$SCRIPT_DIR/../products/pipeline_run_log.txt"

mkdir -p "$OUT"
mkdir -p "$FIELD_OUT"
shopt -s nullglob

# --- CALDB guard (catches running this script standalone outside run_pipeline.sh) ---
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

log_entry() {
  echo "UVOTSOURCE | $1 | $2 | $3" >> "$LOG"
}

# ----------------------------------------------------------
# Validate a FITS image file before uvotsource
# Checks:
# - exists and non-empty
# - FITS opens
# - contains at least one image HDU with finite data
# ----------------------------------------------------------
validate_image_fits() {
  local f="$1"
  [ -s "$f" ] || return 1

  python3 - "$f" <<'PY'
from astropy.io import fits
import numpy as np
import sys

path = sys.argv[1]

try:
    with fits.open(path, memmap=False) as h:
        found = False
        for hdu in h:
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and hdu.data is not None:
                arr = np.asarray(hdu.data)
                if arr.size == 0:
                    continue
                if not np.isfinite(arr).any():
                    continue
                found = True
                break
        sys.exit(0 if found else 1)
except Exception:
    sys.exit(1)
PY
}

# ----------------------------------------------------------
# Check that sky image and exposure map are compatible
# ----------------------------------------------------------
validate_image_pair() {
  local sk="$1"
  local ex="$2"

  validate_image_fits "$sk" || return 1
  validate_image_fits "$ex" || return 1

  python3 - "$sk" "$ex" <<'PY'
from astropy.io import fits
import sys

def first_image_shape(path):
    with fits.open(path, memmap=False) as h:
        for hdu in h:
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and hdu.data is not None:
                return hdu.data.shape
    return None

sk_shape = first_image_shape(sys.argv[1])
ex_shape = first_image_shape(sys.argv[2])

if sk_shape is None or ex_shape is None:
    sys.exit(1)

sys.exit(0 if sk_shape == ex_shape else 1)
PY
}

# ----------------------------------------------------------
# Basic DS9 region-file validation
# Checks:
# - file exists and non-empty
# - contains fk5 or image coords
# - contains at least one region primitive
# ----------------------------------------------------------
validate_region_file() {
  local reg="$1"
  [ -s "$reg" ] || return 1

  python3 - "$reg" <<'PY'
import sys
import re

path = sys.argv[1]

try:
    text = open(path, "r", encoding="utf-8", errors="ignore").read().lower()
except Exception:
    sys.exit(1)

has_coords = ("fk5" in text) or ("\nimage" in text) or text.startswith("image")
has_shape = any(tok in text for tok in ("circle(", "annulus(", "ellipse(", "polygon(", "box("))

sys.exit(0 if (has_coords and has_shape) else 1)
PY
}

# ----------------------------------------------------------
# Validate uvotsource output FITS more strictly
# Checks:
# - FITS opens
# - MAGHIST or ext 1 exists with data
# - MAG and MAG_ERR exist
# - if scalar values are readable, MAG and MAG_ERR are finite
# ----------------------------------------------------------
validate_uvotsource_fits() {
  local f="$1"
  python3 - "$f" <<'PY'
from astropy.io import fits
import numpy as np
import sys

f = sys.argv[1]

def scalarize(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    v = arr.ravel()[0]
    try:
        return float(v)
    except Exception:
        return None

try:
    with fits.open(f, memmap=False) as h:
        hdu = h["MAGHIST"] if "MAGHIST" in h else (h[1] if len(h) > 1 else None)
        if hdu is None or hdu.data is None:
            sys.exit(1)

        cols = set(hdu.columns.names or [])
        if "MAG" not in cols or "MAG_ERR" not in cols:
            sys.exit(1)

        mag = scalarize(hdu.data["MAG"])
        magerr = scalarize(hdu.data["MAG_ERR"])

        if mag is None or magerr is None:
            sys.exit(1)
        if not np.isfinite(mag) or not np.isfinite(magerr):
            sys.exit(1)

        sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

# ----------------------------------------------------------
# Log output metadata / soft warnings
# This does not fail the run.
# ----------------------------------------------------------
log_uvotsource_qc() {
  local obsid="$1"
  local filter="$2"
  local outfile="$3"

  python3 - "$obsid" "$filter" "$outfile" "$LOG" <<'PY'
from astropy.io import fits
import numpy as np
import sys
import os

obsid, filt, path, log = sys.argv[1:]

def scalarize(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    v = arr.ravel()[0]
    try:
        return float(v)
    except Exception:
        return None

def append(msg):
    with open(log, "a") as f:
        f.write(f"UVOTSOURCE | {obsid} | {filt} | {msg}\n")

try:
    with fits.open(path, memmap=False) as h:
        hdu = h["MAGHIST"] if "MAGHIST" in h else (h[1] if len(h) > 1 else None)
        if hdu is None or hdu.data is None:
            append(f"QC_WARN no_maghist_data file={os.path.basename(path)}")
            sys.exit(0)

        names = set(hdu.columns.names or [])
        data = hdu.data

        mag         = scalarize(data["MAG"]) if "MAG" in names else None
        mag_err     = scalarize(data["MAG_ERR"]) if "MAG_ERR" in names else None
        mag_lim     = scalarize(data["MAG_LIM"]) if "MAG_LIM" in names else None
        mag_coi_lim = scalarize(data["MAG_COI_LIM"]) if "MAG_COI_LIM" in names else None
        rate        = scalarize(data["RATE"]) if "RATE" in names else None
        rate_err    = scalarize(data["RATE_ERR"]) if "RATE_ERR" in names else None
        flux        = scalarize(data["FLUX"]) if "FLUX" in names else None
        exposure    = scalarize(data["EXPOSURE"]) if "EXPOSURE" in names else None
        snr         = None

        if rate is not None and rate_err is not None and np.isfinite(rate) and np.isfinite(rate_err) and rate_err > 0:
            snr = rate / rate_err

        append(
            f"QC_INFO file={os.path.basename(path)} "
            f"mag={mag} mag_err={mag_err} mag_lim={mag_lim} mag_coi_lim={mag_coi_lim} "
            f"rate={rate} rate_err={rate_err} snr={snr} flux={flux} exposure={exposure}"
        )

        # soft warnings only
        if exposure is not None and np.isfinite(exposure) and exposure <= 0:
            append(f"QC_WARN nonpositive_exposure={exposure} file={os.path.basename(path)}")

        if mag_err is not None and np.isfinite(mag_err) and mag_err > 0.5:
            append(f"QC_WARN large_mag_err={mag_err:.3f} file={os.path.basename(path)}")

        if snr is not None and np.isfinite(snr) and snr < 3:
            append(f"QC_WARN low_snr={snr:.3f} file={os.path.basename(path)}")

        if mag is not None and np.isfinite(mag) and not (5.0 <= mag <= 30.0):
            append(f"QC_WARN suspicious_mag={mag:.3f} file={os.path.basename(path)}")

        if mag_coi_lim is not None and np.isfinite(mag_coi_lim) and not (5.0 <= mag_coi_lim <= 25.0):
            append(f"QC_WARN suspicious_mag_coi_lim={mag_coi_lim:.3f} file={os.path.basename(path)}")

except Exception as e:
    append(f"QC_WARN metadata_read_failed file={os.path.basename(path)} err={e}")
PY
}

# ----------------------------------------------------------
# CONFIGURE FOR YOUR SOURCE
# Replace the filenames below with your own DS9 region files.
# ----------------------------------------------------------
SRC_REG="$SCRIPT_DIR/b1259.reg"
BKG_REG="$SCRIPT_DIR/bg_final.reg"

# Field star region files for systematic check
FIELD_SRC_REG="$SCRIPT_DIR/field.reg"
FIELD_BKG_REG="$SCRIPT_DIR/fieldbg.reg"
# ----------------------------------------------------------

if [ ! -f "$SRC_REG" ] || [ ! -f "$BKG_REG" ]; then
  echo "Region file missing." >&2
  exit 1
fi

if ! validate_region_file "$SRC_REG"; then
  echo "Invalid source region file: $SRC_REG" >&2
  exit 1
fi

if ! validate_region_file "$BKG_REG"; then
  echo "Invalid background region file: $BKG_REG" >&2
  exit 1
fi

for obs in "$ROOT"/*; do
  [ -d "$obs" ] || continue
  obsid="$(basename "$obs")"

  for sk in "$obs"/*_summed.img; do
    base="${sk%_summed.img}"
    ex="${base}_summed_ex.img"

    if [ ! -f "$ex" ]; then
      log_entry "$obsid" "$(basename "$sk")" "MISSING_EXPMAP"
      continue
    fi

    if ! validate_image_pair "$sk" "$ex"; then
      log_entry "$obsid" "$(basename "$sk")" "INVALID_IMAGE_OR_EXPMAP"
      continue
    fi

    fname="$(basename "$base")"

    filter="unknown"
    case "$fname" in
      *uw2) filter="w2" ;;
      *uw1) filter="w1" ;;
      *um2) filter="m2" ;;
      *uuu) filter="u"  ;;
      *ubb) filter="bb" ;;
      *uvv) filter="vv" ;;
      *uwh) filter="wh" ;;
    esac

    outname="${OUT}/${obsid}_${filter}.fits"

    # Validate existing FITS if present
    if [ -f "$outname" ]; then
      if validate_uvotsource_fits "$outname"; then
        log_entry "$obsid" "$filter" "EXISTS_VALID"
        log_uvotsource_qc "$obsid" "$filter" "$outname"
        continue
      else
        log_entry "$obsid" "$filter" "EXISTS_INVALID_REDO"
        rm -f "$outname"
      fi
    fi

    tmp_log=$(mktemp)

    if uvotsource \
        image="$sk" \
        expfile="$ex" \
        srcreg="$SRC_REG" \
        bkgreg="$BKG_REG" \
        apercorr=CURVEOFGROWTH \
        sigma=5 \
        outfile="$outname" \
        clobber=yes \
        mode=h \
        chatter=1 \
        > "$tmp_log" 2>&1
    then
      if validate_uvotsource_fits "$outname"; then
        log_entry "$obsid" "$filter" "OK"
        log_uvotsource_qc "$obsid" "$filter" "$outname"
      else
        rm -f "$outname"
        log_entry "$obsid" "$filter" "OUTPUT_INVALID"
      fi
    else
      rm -f "$outname"

      if grep -q "containment is 0.000" "$tmp_log"; then
        reason="SOURCE_OUTSIDE_FOV"
      elif grep -qi "invalid exposure" "$tmp_log"; then
        reason="ZERO_EXPOSURE"
      elif grep -qi "low .* in FOV" "$tmp_log"; then
        reason="REGION_OUTSIDE_DETECTOR"
      elif grep -qi "background.*outside" "$tmp_log"; then
        reason="BACKGROUND_REGION_INVALID"
      elif grep -qi "source.*outside" "$tmp_log"; then
        reason="SOURCE_REGION_INVALID"
      elif grep -qi "cannot read.*region" "$tmp_log"; then
        reason="REGION_PARSE_FAILED"
      else
        reason="UVOTSOURCE_FAILED"
      fi

      log_entry "$obsid" "$filter" "$reason"
    fi

    rm -f "$tmp_log"
  done
done

# ==========================================================
# FIELD STAR PHOTOMETRY (systematic check)
# ==========================================================

echo "Running field star photometry for systematic check..."

if [ ! -f "$FIELD_SRC_REG" ] || [ ! -f "$FIELD_BKG_REG" ]; then
  echo "WARNING: Field star region files not found — skipping field star check." >&2
  echo "  Expected: $FIELD_SRC_REG" >&2
  echo "  Expected: $FIELD_BKG_REG" >&2
else
  if ! validate_region_file "$FIELD_SRC_REG"; then
    echo "WARNING: Invalid field star source region — skipping field star check." >&2
  elif ! validate_region_file "$FIELD_BKG_REG"; then
    echo "WARNING: Invalid field star background region — skipping field star check." >&2
  else
    for obs in "$ROOT"/*; do
      [ -d "$obs" ] || continue
      obsid="$(basename "$obs")"

      for sk in "$obs"/*_summed.img; do
        base="${sk%_summed.img}"
        ex="${base}_summed_ex.img"

        [ -f "$ex" ] || continue
        validate_image_pair "$sk" "$ex" || continue

        fname="$(basename "$base")"

        filter="unknown"
        case "$fname" in
          *uw2) filter="w2" ;;
          *uw1) filter="w1" ;;
          *um2) filter="m2" ;;
          *uuu) filter="u"  ;;
          *ubb) filter="bb" ;;
          *uvv) filter="vv" ;;
          *uwh) filter="wh" ;;
        esac

        outname="${FIELD_OUT}/${obsid}_${filter}.fits"

        if [ -f "$outname" ]; then
          if validate_uvotsource_fits "$outname"; then
            log_entry "$obsid" "FIELD_${filter}" "EXISTS_VALID"
            continue
          else
            log_entry "$obsid" "FIELD_${filter}" "EXISTS_INVALID_REDO"
            rm -f "$outname"
          fi
        fi

        tmp_log=$(mktemp)

        if uvotsource \
            image="$sk" \
            expfile="$ex" \
            srcreg="$FIELD_SRC_REG" \
            bkgreg="$FIELD_BKG_REG" \
            apercorr=CURVEOFGROWTH \
            sigma=5 \
            outfile="$outname" \
            clobber=yes \
            mode=h \
            chatter=1 \
            > "$tmp_log" 2>&1
        then
          if validate_uvotsource_fits "$outname"; then
            log_entry "$obsid" "FIELD_${filter}" "OK"
          else
            rm -f "$outname"
            log_entry "$obsid" "FIELD_${filter}" "OUTPUT_INVALID"
          fi
        else
          rm -f "$outname"

          if grep -q "containment is 0.000" "$tmp_log"; then
            reason="SOURCE_OUTSIDE_FOV"
          elif grep -qi "invalid exposure" "$tmp_log"; then
            reason="ZERO_EXPOSURE"
          elif grep -qi "low .* in FOV" "$tmp_log"; then
            reason="REGION_OUTSIDE_DETECTOR"
          elif grep -qi "background.*outside" "$tmp_log"; then
            reason="BACKGROUND_REGION_INVALID"
          elif grep -qi "source.*outside" "$tmp_log"; then
            reason="SOURCE_REGION_INVALID"
          else
            reason="UVOTSOURCE_FAILED"
          fi

          log_entry "$obsid" "FIELD_${filter}" "$reason"
        fi

        rm -f "$tmp_log"
      done
    done
    echo "Field star photometry complete. Results in: $FIELD_OUT"
  fi
fi

shopt -u nullglob