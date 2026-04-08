#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RAW_DIR="$SCRIPT_DIR/../all_years_observations"
OUTROOT="$SCRIPT_DIR/../products/summed_images"
LOG="$SCRIPT_DIR/../products/pipeline_run_log.txt"
# Cache directory for pre-sum mag=99 screening results.
# Each checked snapshot leaves a small .prescan file here so reruns skip
# snapshots that were already screened, without re-running uvotsource.
PRESCAN_CACHE="$SCRIPT_DIR/../products/prescan_cache"
# File listing all obsid/filter products excluded by the mag=99 pre-scan.
# Format per line:
#   obsid|FILTER
# Example:
#   00030966095|UVW2
#
# Read by lightcurve.py to exclude only the affected product,
# not the whole obsid.
PRESCAN_EXCLUDED="$SCRIPT_DIR/../products/prescan_excluded.txt"

mkdir -p "$OUTROOT"
mkdir -p "$PRESCAN_CACHE"

# Initialise the excluded obsids file fresh each run so it always
# reflects the current state of the prescan cache exactly.
: > "$PRESCAN_EXCLUDED"
shopt -s nullglob

log_entry() {
  echo "UVOTIMSUM | $1 | $2 | $3" >> "$LOG"
}

infer_filter_from_filename() {
  local name
  name="$(basename "$1")"

  case "$name" in
    *uw2*) echo "UVW2" ;;
    *uw1*) echo "UVW1" ;;
    *um2*) echo "UVM2" ;;
    *uuu*) echo "U"    ;;
    *ubb*) echo "B"    ;;
    *uvv*) echo "V"    ;;
    *uwh*) echo "WHITE" ;;
    *)     echo "UNKNOWN" ;;
  esac
}
# ----------------------------------------------------------
# Pre-sum mag=99 screening (bad CCD region check)
#
# Runs uvotsource on a single raw (unsummed) snapshot using the
# science region files. If uvotsource returns MAG=99.000 the
# snapshot is on a bad CCD region (SSS defect or dead spot) and
# must be excluded before coadding — uvotimsum will NOT flag this,
# per UVOT helpdesk advice.
#
# Results are cached: a .prescan file records PASS or MAG99 so
# reruns never re-screen the same snapshot.
#
# Arguments: obsid  sk_file  ex_file  src_reg  bkg_reg
# Returns:
#   0 = clean (safe to coadd)
#   1 = MAG=99 detected OR uvotsource failed (exclude to be safe)
# ----------------------------------------------------------
screen_mag99() {
  local obsid="$1"
  local sk="$2"
  local ex="$3"
  local src_reg="$4"
  local bkg_reg="$5"

  local cache_key
  cache_key="$PRESCAN_CACHE/$(basename "${sk%.img}").prescan"

  # Return cached result if available
  if [ -f "$cache_key" ]; then
    local cached
    local filter
    cached="$(cat "$cache_key")"
    filter="$(infer_filter_from_filename "$sk")"

    if [ "$cached" = "PASS" ]; then
      return 0
    else
      if [ "$cached" = "MAG99" ]; then
        log_entry "$obsid" "$(basename "$sk")" "PRESCAN_MAG99_IMAGE_EXCLUDED (cached)"
        echo "${obsid}|${filter}" >> "$PRESCAN_EXCLUDED"
      else
        log_entry "$obsid" "$(basename "$sk")" "PRESCAN_UVOTSOURCE_FAILED_IMAGE_EXCLUDED (cached)"
      fi
      return 1
    fi
  fi

  local tmp_out
  tmp_out=$(mktemp --suffix=.fits)
  local tmp_log
  tmp_log=$(mktemp)

  local result=0

  if uvotsource \
      image="$sk" \
      expfile="$ex" \
      srcreg="$src_reg" \
      bkgreg="$bkg_reg" \
      apercorr=CURVEOFGROWTH \
      sigma=5 \
      outfile="$tmp_out" \
      clobber=yes \
      mode=h \
      chatter=1 \
      > "$tmp_log" 2>&1
  then
    if python3 - "$tmp_out" "$cache_key" <<'PY'
from astropy.io import fits
import numpy as np
import sys

outfile, cache_key = sys.argv[1], sys.argv[2]

def write_cache(val):
    with open(cache_key, "w") as f:
        f.write(val)

try:
    with fits.open(outfile, memmap=False) as h:
        hdu = h["MAGHIST"] if "MAGHIST" in h else (h[1] if len(h) > 1 else None)
        if hdu is None or hdu.data is None:
            write_cache("FAILED")
            sys.exit(1)

        names = set(hdu.columns.names or [])
        if "MAG" not in names:
            write_cache("FAILED")
            sys.exit(1)

        mag_raw = hdu.data["MAG"]
        arr = np.asarray(mag_raw).ravel()
        if arr.size == 0:
            write_cache("FAILED")
            sys.exit(1)

        mag = float(arr[0])

        if abs(mag - 99.0) < 0.01:
            write_cache("MAG99")
            sys.exit(1)
        else:
            write_cache("PASS")
            sys.exit(0)

except Exception:
    write_cache("FAILED")
    sys.exit(1)
PY
    then
      result=0
    else
      result=1
    fi
  else
    echo "FAILED" > "$cache_key"
    result=1
  fi

  rm -f "$tmp_out" "$tmp_log"

  if [ "$result" -ne 0 ]; then
    local cached_val
    local filter
    cached_val="$(cat "$cache_key" 2>/dev/null || echo FAILED)"
    filter="$(infer_filter_from_filename "$sk")"

    if [ "$cached_val" = "MAG99" ]; then
      log_entry "$obsid" "$(basename "$sk")" "PRESCAN_MAG99_IMAGE_EXCLUDED"
      echo "${obsid}|${filter}" >> "$PRESCAN_EXCLUDED"
    else
      log_entry "$obsid" "$(basename "$sk")" "PRESCAN_UVOTSOURCE_FAILED_IMAGE_EXCLUDED"
    fi
    return 1
  fi

  return 0
}

# ----------------------------------------------------------
# Validate a FITS image file before/after uvotimsum
# Checks:
# - file exists and is non-empty
# - FITS opens cleanly
# - contains at least one image HDU with data
# - image data are not all-NaN / all-nonfinite
# Returns:
#   0 = valid
#   1 = invalid
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
# Compare summed sky image and summed exposure map
# Checks:
# - both are valid FITS images
# - first science image HDU dimensions match
# Returns:
#   0 = OK
#   1 = invalid/problem
# ----------------------------------------------------------
validate_imsum_pair() {
  local out_sk="$1"
  local out_ex="$2"

  validate_image_fits "$out_sk" || return 1
  validate_image_fits "$out_ex" || return 1

  python3 - "$out_sk" "$out_ex" <<'PY'
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
# Log lightweight QC metadata / warnings
# This does NOT fail the pipeline; it just records suspicious cases.
# Logs things like:
# - number of image HDUs
# - first image shape
# - EXPOSURE keyword if present
# - aspect-related keywords if present
# - warnings for missing WCS/aspect keywords
# ----------------------------------------------------------
log_qc_metadata() {
  local obsid="$1"
  local label="$2"
  local fitsfile="$3"

  python3 - "$obsid" "$label" "$fitsfile" "$LOG" <<'PY'
from astropy.io import fits
import sys
import os

obsid, label, path, log = sys.argv[1:]

def append(msg):
    with open(log, "a") as f:
        f.write(f"UVOTIMSUM | {obsid} | {label} | {msg}\n")

try:
    with fits.open(path, memmap=False) as h:
        image_hdus = []
        for i, hdu in enumerate(h):
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and hdu.data is not None:
                image_hdus.append((i, hdu))

        if not image_hdus:
            append(f"QC_WARN no_image_hdu file={os.path.basename(path)}")
            sys.exit(0)

        idx, hdu = image_hdus[0]
        hdr = hdu.header
        shape = getattr(hdu.data, "shape", None)

        exposure = hdr.get("EXPOSURE", h[0].header.get("EXPOSURE", "NA"))
        aspcorr  = hdr.get("ASPCORR", h[0].header.get("ASPCORR", "NA"))
        ra_obj   = hdr.get("RA_OBJ", h[0].header.get("RA_OBJ", "NA"))
        dec_obj  = hdr.get("DEC_OBJ", h[0].header.get("DEC_OBJ", "NA"))

        append(
            f"QC_INFO file={os.path.basename(path)} "
            f"image_hdus={len(image_hdus)} first_hdu={idx} shape={shape} "
            f"exposure={exposure} aspcorr={aspcorr} ra_obj={ra_obj} dec_obj={dec_obj}"
        )

        # soft warnings only
        if len(image_hdus) > 1:
            append(f"QC_WARN multiple_image_hdus={len(image_hdus)} file={os.path.basename(path)}")

        has_wcs = any(k in hdr for k in ("CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"))
        if not has_wcs:
            append(f"QC_WARN missing_basic_wcs file={os.path.basename(path)}")

        if str(aspcorr).strip().upper() in ("NONE", "UNKNOWN", "FALSE"):
            append(f"QC_WARN suspicious_aspcorr={aspcorr} file={os.path.basename(path)}")

except Exception as e:
    append(f"QC_WARN metadata_read_failed file={os.path.basename(path)} err={e}")
PY
}

# ----------------------------------------------------------
# Region files for pre-sum mag=99 screening.
# These must be the same files used by run_uvotsource.sh.
# ----------------------------------------------------------
SRC_REG="$SCRIPT_DIR/b1259.reg"
BKG_REG="$SCRIPT_DIR/bg_final.reg"

if [ ! -f "$SRC_REG" ] || [ ! -f "$BKG_REG" ]; then
  echo "ERROR: Region files missing — cannot run pre-sum mag=99 screening." >&2
  echo "  Expected: $SRC_REG" >&2
  echo "  Expected: $BKG_REG" >&2
  exit 1
fi

for obsdir in "$RAW_DIR"/*/ ; do
  [ -d "${obsdir}uvot/image" ] || continue

  obsid="$(basename "${obsdir%/}")"
  imdir="${obsdir}uvot/image"
  outdir="$OUTROOT/$obsid"
  mkdir -p "$outdir"

  echo "Processing $obsid"

  for sk in "$imdir"/*_sk.img; do
    base="${sk%_sk.img}"
    ex="${base}_ex.img"

    if [ ! -f "$ex" ]; then
      log_entry "$obsid" "$(basename "$sk")" "MISSING_EXPMAP"
      continue
    fi

    # Validate raw inputs before doing anything
    if ! validate_image_fits "$sk"; then
      log_entry "$obsid" "$(basename "$sk")" "INPUT_SK_INVALID"
      continue
    fi

    if ! validate_image_fits "$ex"; then
      log_entry "$obsid" "$(basename "$sk")" "INPUT_EXPMAP_INVALID"
      continue
    fi

    # Log QC metadata for raw inputs
    log_qc_metadata "$obsid" "$(basename "$sk")" "$sk"
    log_qc_metadata "$obsid" "$(basename "$ex")" "$ex"

    # ----------------------------------------------------------
    # Pre-sum mag=99 screening
    # Run uvotsource on the raw snapshot before coadding.
    # Snapshots on bad CCD regions return MAG=99; uvotimsum does
    # NOT propagate this flag into the summed image, so we must
    # catch and exclude them here.
    # ----------------------------------------------------------
    if ! screen_mag99 "$obsid" "$sk" "$ex" "$SRC_REG" "$BKG_REG"; then
      continue
    fi

    stem="$(basename "$base")"
    out_sk="$outdir/${stem}_summed.img"
    out_ex="$outdir/${stem}_summed_ex.img"

    # If outputs exist, validate them more strictly
    if [ -f "$out_sk" ] || [ -f "$out_ex" ]; then
      if validate_imsum_pair "$out_sk" "$out_ex"; then
        log_entry "$obsid" "$(basename "$sk")" "EXISTS_VALID"
        log_qc_metadata "$obsid" "$(basename "$out_sk")" "$out_sk"
        log_qc_metadata "$obsid" "$(basename "$out_ex")" "$out_ex"
        continue
      else
        log_entry "$obsid" "$(basename "$sk")" "EXISTS_INVALID_REDO"
        rm -f "$out_sk" "$out_ex"
      fi
    fi

    tmp_log=$(mktemp)

    if uvotimsum infile="$sk" outfile="$out_sk" clobber=yes > "$tmp_log" 2>&1 && \
       uvotimsum infile="$ex" outfile="$out_ex" expmap=yes clobber=yes >> "$tmp_log" 2>&1
    then
      if validate_imsum_pair "$out_sk" "$out_ex"; then
        log_entry "$obsid" "$(basename "$sk")" "SUMMED_OK"
        log_qc_metadata "$obsid" "$(basename "$out_sk")" "$out_sk"
        log_qc_metadata "$obsid" "$(basename "$out_ex")" "$out_ex"
        echo "  summed: $(basename "$sk")"
      else
        log_entry "$obsid" "$(basename "$sk")" "SUMMED_OUTPUT_INVALID"
        rm -f "$out_sk" "$out_ex"
      fi
    else
      log_entry "$obsid" "$(basename "$sk")" "SUMMED_FAILED"
      rm -f "$out_sk" "$out_ex"
    fi

    rm -f "$tmp_log"
  done
done

shopt -u nullglob