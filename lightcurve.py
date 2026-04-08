#!/usr/bin/env python3
# lightcurve.py

import subprocess, sys
for pkg in ["plotly", "astropy", "pandas", "numpy", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from astropy.io import fits
from astropy.time import Time

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================
# PATH CONFIGURATION
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(SCRIPT_DIR, "../products/uvotsource_results")
TABLE_DIR = os.path.join(SCRIPT_DIR, "../products/tables")
PLOT_DIR  = os.path.join(SCRIPT_DIR, "../products/plots")
LOG_FILE  = os.path.join(SCRIPT_DIR, "../products/pipeline_run_log.txt")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "by_filter"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "by_orbit"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "interactive"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "interactive", "by_filter"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "interactive", "by_orbit"), exist_ok=True)

OUTLIER_FILE     = os.path.join(TABLE_DIR, "outliers.txt")
REVIEW_FILE      = os.path.join(TABLE_DIR, "manual_review.txt")
EXCLUDE_FILE     = os.path.join(TABLE_DIR, "excluded_observations.txt")
FIELD_FLAGS_FILE = os.path.join(TABLE_DIR, "fieldstar_flags.csv")
PRESCAN_EXCLUDED = os.path.join(SCRIPT_DIR, "../products/prescan_excluded.txt")

# ==========================================================
# CONFIGURE FOR YOUR SOURCE
# ==========================================================

SOURCE_NAME = "PSR B1259–63"

PERIASTRON_MJDS = sorted([
    54307.969,   # 2007 periastron
    55544.694,   # 2010 periastron
    56781.418,   # 2014 periastron
    58018.143,   # 2017 periastron
    59254.867,   # 2021 periastron
    60491.592,   # 2024 periastron
])

# ==========================================================
# QC THRESHOLDS
# ==========================================================

MAX_MAG_ERR = 0.5       # science-quality cutoff
MIN_SNR     = 3.0       # only used if RATE and RATE_ERR exist
OUTLIER_SIGMA = 3.0     # report only; do not remove from plots

# ==========================================================
# STYLE CONFIGURATION
# ==========================================================

FILTER_ORDER = ["V", "B", "U", "UVW1", "UVM2", "UVW2", "WHITE", "UNKNOWN"]
MARKERS = {"V": "o", "B": "s", "U": "^", "UVW1": "D", "UVM2": "v", "UVW2": "P", "WHITE": "X", "UNKNOWN": "."}

FILTER_COLORS = {
    "V":       "#4daf4a",
    "B":       "#377eb8",
    "U":       "#984ea3",
    "UVW1":    "#e41a1c",
    "UVM2":    "#ff7f00",
    "UVW2":    "#a65628",
    "WHITE":   "#999999",
    "UNKNOWN": "#cccccc",
}

PLOTLY_SYMBOLS = {
    "V": "circle", "B": "square", "U": "triangle-up",
    "UVW1": "diamond", "UVM2": "triangle-down", "UVW2": "star",
    "WHITE": "x", "UNKNOWN": "circle-open",
}

FILTER_EV = {
    "V":       "2.29 eV",
    "B":       "2.85 eV",
    "U":       "3.52 eV",
    "UVW1":    "4.62 eV",
    "UVM2":    "5.52 eV",
    "UVW2":    "5.95 eV",
    "WHITE":   "",
    "UNKNOWN": "",
}

def filter_label(flt):
    ev = FILTER_EV.get(flt, "")
    return f"{flt} ({ev})" if ev else flt

def orbit_label(orbit_id):
    try:
        year = int(Time(float(PERIASTRON_MJDS[int(orbit_id)]), format="mjd").datetime.year)
        return f"Orbit {int(orbit_id)} ({year})"
    except Exception:
        return f"Orbit {int(orbit_id)}"

def log_entry(obsid, flt, status):
    with open(LOG_FILE, "a") as logf:
        logf.write(f"LIGHTCURVE | {obsid} | {flt} | {status}\n")

# ==========================================================
# HELPERS
# ==========================================================

def infer_obsid_from_name(path: str) -> str:
    m = re.search(r"(\d{11})", path)
    return m.group(1) if m else "unknown"

def infer_filter_from_name(path: str) -> str:
    name = path.lower()

    if name.endswith("_w1.fits"):
        return "UVW1"
    if name.endswith("_w2.fits"):
        return "UVW2"
    if name.endswith("_m2.fits"):
        return "UVM2"

    if name.endswith("_u.fits"):
        return "U"
    if name.endswith("_bb.fits"):
        return "B"
    if name.endswith("_vv.fits"):
        return "V"
    if name.endswith("_wh.fits"):
        return "WHITE"

    return "UNKNOWN"

def scalarize(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    val = arr.ravel()[0]
    try:
        return float(val)
    except Exception:
        return val

def mjd_from_hdul(hdul, data):
    hdr0, hdr1 = hdul[0].header, hdul[1].header

    for h in (hdr0, hdr1):
        if "DATE-OBS" in h:
            try:
                return Time(h["DATE-OBS"], format="isot", scale="utc").mjd
            except Exception:
                try:
                    return Time(h["DATE-OBS"], scale="utc").mjd
                except Exception:
                    pass

    for k in ("MJD-OBS", "MJD_OBS", "MJDSTART", "MJDMEAN", "MJD"):
        if k in hdr0:
            return float(hdr0[k])
        if k in hdr1:
            return float(hdr1[k])

    mjdref = (hdr1.get("MJDREFI", hdr0.get("MJDREFI", None)) or 0.0) + \
             (hdr1.get("MJDREFF", hdr0.get("MJDREFF", None)) or 0.0)

    timezero = hdr1.get("TIMEZERO", hdr0.get("TIMEZERO", 0.0)) or 0.0

    names = set(data.names or [])

    def get_col(col):
        return scalarize(data[col]) if col in names else None

    tstart = get_col("TSTART") or hdr1.get("TSTART", None)
    tstop  = get_col("TSTOP")  or hdr1.get("TSTOP", None)
    expo   = get_col("EXPOSURE") or hdr1.get("EXPOSURE", None)

    if tstart is not None and tstop is not None:
        mid = 0.5 * (tstart + tstop)
    elif tstart is not None and expo is not None:
        mid = tstart + 0.5 * expo
    else:
        return None

    if mjdref == 0.0:
        mjdref = 51910.0

    return mjdref + (timezero + mid) / 86400.0

def mjd_sanity_check(mjd_val, fname):
    if mjd_val is None or not np.isfinite(mjd_val):
        return
    if mjd_val < 40000 or mjd_val > 70000:
        print(f"WARNING: suspicious MJD {mjd_val:.3f} in {os.path.basename(fname)}")
        log_entry(infer_obsid_from_name(fname), infer_filter_from_name(fname), f"SUSPICIOUS_MJD {mjd_val:.3f}")

def assign_orbit(mjd):
    for i, T in enumerate(PERIASTRON_MJDS):
        if i == 0:
            lower = T - 0.5 * (PERIASTRON_MJDS[i + 1] - T)
        else:
            lower = 0.5 * (PERIASTRON_MJDS[i - 1] + T)

        if i == len(PERIASTRON_MJDS) - 1:
            upper = T + 0.5 * (T - PERIASTRON_MJDS[i - 1])
        else:
            upper = 0.5 * (T + PERIASTRON_MJDS[i + 1])

        if lower <= mjd < upper:
            return i, T

    return None, None

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return np.nan

# ==========================================================
# PRESCAN EXCLUSIONS — load specific (obsid, filter) pairs
# flagged by run_uvotimsum.sh.
#
# Expected file format:
#   obsid|filter
# e.g.
#   00030966095|UVW2
#
# This allows exclusion of only the affected product rather
# than the whole obsid.
# ==========================================================

prescan_exclusions = set()   # set of (obsid, filter)

if os.path.exists(PRESCAN_EXCLUDED):
    with open(PRESCAN_EXCLUDED) as _pf:
        for _line in _pf:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue

            parts = [p.strip() for p in _line.split("|")]
            if len(parts) != 2:
                print(f"WARNING: malformed prescan exclusion line skipped: {_line}")
                continue

            obsid, flt = parts
            prescan_exclusions.add((obsid, flt.upper()))

    print(
        f"Loaded {len(prescan_exclusions)} prescan-excluded obsid/filter pair(s) "
        f"from {os.path.basename(PRESCAN_EXCLUDED)}"
    )
    if prescan_exclusions:
        for _obsid, _flt in sorted(prescan_exclusions):
            print(f"  prescan excluded: {_obsid} | {_flt}")
else:
    print(f"No prescan exclusion file found at {PRESCAN_EXCLUDED} — skipping.")

# ==========================================================
# FIELD STAR SYSTEMATIC CHECK — pre-load flagged obsids
# Runs before main data load so flagged observations can be
# excluded from tables, plots, and all downstream products.
# ==========================================================

FIELD_INPUT_DIR  = os.path.join(SCRIPT_DIR, "../products/fieldstar_results")
FIELD_FLAGS_FILE = os.path.join(TABLE_DIR, "fieldstar_flags.csv")
FIELD_SIGMA      = 3.0

# fieldstar_exclusions: set of (obsid, filter) pairs to exclude
fieldstar_exclusions = set()

field_fits = sorted(glob.glob(os.path.join(FIELD_INPUT_DIR, "*.fits")))

if not field_fits:
    print(f"No field star FITS files found in {FIELD_INPUT_DIR} — skipping systematic check.")
else:
    print(f"Running field star systematic check on {len(field_fits)} files...")

    filter_map_fs = {
        "_w2.fits": "UVW2", "_w1.fits": "UVW1", "_m2.fits": "UVM2",
        "_u.fits":  "U",    "_bb.fits": "B",    "_vv.fits": "V",
        "_wh.fits": "WHITE",
    }

    field_rows = []
    for f in field_fits:
        try:
            with fits.open(f, memmap=False) as hdul:
                if len(hdul) < 2 or hdul[1].data is None:
                    continue
                data  = hdul["MAGHIST"].data if "MAGHIST" in hdul else hdul[1].data
                names = set(data.names or [])

                mag     = scalarize(data["MAG"])     if "MAG"     in names else None
                mag_err = scalarize(data["MAG_ERR"]) if "MAG_ERR" in names else None

                if mag is None or mag_err is None:
                    continue
                if not (np.isfinite(mag) and np.isfinite(mag_err)):
                    continue
                if mag_err > MAX_MAG_ERR:
                    continue

                mjd = mjd_from_hdul(hdul, data)
                if mjd is None or not np.isfinite(mjd):
                    continue

                obsid    = infer_obsid_from_name(f)
                flt_norm = "UNKNOWN"
                for suffix, name in filter_map_fs.items():
                    if f.lower().endswith(suffix):
                        flt_norm = name
                        break

                field_rows.append({
                    "obsid":   obsid,
                    "filter":  flt_norm,
                    "mjd":     mjd,
                    "mag":     mag,
                    "mag_err": mag_err,
                })
        except Exception as e:
            print(f"  WARNING: could not read field star file {os.path.basename(f)}: {e}")

    if not field_rows:
        print("  No valid field star measurements extracted — no exclusions applied.")
    else:
        field_df = pd.DataFrame(field_rows)
        flag_rows = []

        for flt, grp in field_df.groupby("filter"):
            if len(grp) < 3:
                print(f"  Field star {flt}: only {len(grp)} measurements — need ≥3 for σ check, skipping.")
                continue

            mean_mag = grp["mag"].mean()
            std_mag  = grp["mag"].std()

            if std_mag == 0 or not np.isfinite(std_mag):
                print(f"  Field star {flt}: zero std — all measurements identical, no flags.")
                continue

            for _, row in grp.iterrows():
                deviation = row["mag"] - mean_mag
                n_sigma   = abs(deviation) / std_mag
                flagged   = n_sigma > FIELD_SIGMA

                if flagged:
                    fieldstar_exclusions.add((row["obsid"], flt))

                flag_rows.append({
                    "obsid":         row["obsid"],
                    "filter":        flt,
                    "mjd":           row["mjd"],
                    "field_mag":     row["mag"],
                    "field_mag_err": row["mag_err"],
                    "filter_mean":   round(mean_mag, 4),
                    "filter_std":    round(std_mag, 4),
                    "deviation_mag": round(deviation, 4),
                    "n_sigma":       round(n_sigma, 2),
                    "flagged":       flagged,
                })

        flags_df = pd.DataFrame(flag_rows)
        flags_df.to_csv(FIELD_FLAGS_FILE, index=False)

        flagged_df = flags_df[flags_df["flagged"]]
        print(f"  Field star check: {len(flagged_df)} observation(s) flagged at >{FIELD_SIGMA}σ "
              f"and will be excluded from light curves.")
        if not flagged_df.empty:
            print(f"  {'ObsID':<15} {'Filter':<8} {'MJD':>10} {'Field mag':>10} "
                  f"{'Mean':>8} {'Δmag':>8} {'Nσ':>6}")
            print("  " + "-" * 65)
            for _, r in flagged_df.sort_values(["filter", "mjd"]).iterrows():
                print(f"  {r['obsid']:<15} {r['filter']:<8} {r['mjd']:>10.3f} "
                      f"{r['field_mag']:>10.3f} {r['filter_mean']:>8.3f} "
                      f"{r['deviation_mag']:>+8.3f} {r['n_sigma']:>6.1f}σ")
        print(f"  Full field star flag table -> {FIELD_FLAGS_FILE}")

# ==========================================================
# LOAD ALL FITS FILES
# ==========================================================

fits_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
if not fits_files:
    print(f"No FITS files found in {INPUT_DIR}")
    raise SystemExit(1)

rows = []
for f in fits_files:
    try:
        with fits.open(f, memmap=False) as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                print(f"Skipped {f}: no BINTABLE in extension +1")
                log_entry(infer_obsid_from_name(f), infer_filter_from_name(f), "SKIPPED_NO_BINTABLE")
                continue

            data = hdul["MAGHIST"].data if "MAGHIST" in hdul else hdul[1].data
            names = set(data.names or [])

            if "MAG" not in names or "MAG_ERR" not in names:
                print(f"Skipped {f}: missing MAG/MAG_ERR columns")
                log_entry(infer_obsid_from_name(f), infer_filter_from_name(f), "SKIPPED_MISSING_MAG_COLUMNS")
                continue

            mjd_val = mjd_from_hdul(hdul, data)
            mjd_sanity_check(mjd_val, f)

            row = {
                "obsid":  infer_obsid_from_name(f),
                "filter": infer_filter_from_name(f),
                "file":   f,
                "mjd":    mjd_val,
            }

            for col in data.names:
                row[col] = scalarize(data[col])

            row["mag_vega"]    = scalarize(data["MAG"])
            row["mag_err"]     = scalarize(data["MAG_ERR"])
            row["mag_lim"]     = scalarize(data["MAG_LIM"]) if "MAG_LIM" in names else np.nan
            row["mag_coi_lim"] = scalarize(data["MAG_COI_LIM"]) if "MAG_COI_LIM" in names else np.nan
            row["rate"]        = scalarize(data["RATE"]) if "RATE" in names else np.nan
            row["rate_err"]    = scalarize(data["RATE_ERR"]) if "RATE_ERR" in names else np.nan
            row["flux"]        = scalarize(data["FLUX"]) if "FLUX" in names else np.nan
            row["exposure"]    = scalarize(data["EXPOSURE"]) if "EXPOSURE" in names else np.nan

            row["snr"] = np.nan
            if np.isfinite(safe_float(row["rate"])) and np.isfinite(safe_float(row["rate_err"])) and row["rate_err"] > 0:
                row["snr"] = row["rate"] / row["rate_err"]

            row["coi_limited"] = False
            if (
                np.isfinite(safe_float(row["mag_vega"]))
                and np.isfinite(safe_float(row["mag_coi_lim"]))
                and 5 < row["mag_coi_lim"] < 25
            ):
                if row["mag_vega"] < row["mag_coi_lim"]:
                    row["coi_limited"] = True

            rows.append(row)

    except Exception as e:
        print(f"Skipped {f}: {e}")
        log_entry(infer_obsid_from_name(f), infer_filter_from_name(f), f"SKIPPED_EXCEPTION {e}")

# ==========================================================
# INITIAL TABLE
# ==========================================================

df = pd.DataFrame(rows)

if df.empty:
    print("No rows collected — check previous skip messages.")
    raise SystemExit(1)

df = df.dropna(subset=["mjd", "mag_vega"]).sort_values(["mjd", "filter"]).reset_index(drop=True)

print("\n--- coincidence summary by filter ---")
print(df.groupby("filter")["coi_limited"].value_counts())

print("\n--- flagged as coincidence-limited ---")
print(df[df["coi_limited"]][["filter", "mag_vega", "mag_coi_lim", "mag_err"]].head(10))

print("\n--- not coincidence-limited ---")
print(df[~df["coi_limited"]][["filter", "mag_vega", "mag_coi_lim", "mag_err"]].head(10))

# ==========================================================
# ORBIT ASSIGNMENT
# ==========================================================

df["orbit_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
df["periastron_mjd"] = np.nan
df["dt_days"] = np.nan

for idx, row in df.iterrows():
    oid, Tperi = assign_orbit(row["mjd"])
    if oid is not None:
        df.at[idx, "orbit_id"] = oid
        df.at[idx, "periastron_mjd"] = float(Tperi)
        df.at[idx, "dt_days"] = float(row["mjd"] - Tperi)

# Convert MJD to dates
t_all = Time(df["mjd"].values, format="mjd")
df["date"] = pd.to_datetime(t_all.to_datetime())

# ==========================================================
# SCIENCE QC FLAGS
# ==========================================================

df["exclude_science"] = False
df["exclude_reason"] = ""
df["review_flag"] = False
df["review_reason"] = ""

exclude_rows = []
review_rows = []

for idx, row in df.iterrows():
    reasons = []
    review_reasons = []

    if row["coi_limited"]:
        reasons.append("COI_LIMITED")

    if not np.isfinite(safe_float(row["mag_vega"])):
        reasons.append("NONFINITE_MAG")

    if not np.isfinite(safe_float(row["mag_err"])):
        reasons.append("NONFINITE_MAG_ERR")

    if np.isfinite(safe_float(row["mag_err"])) and row["mag_err"] > MAX_MAG_ERR:
        reasons.append(f"LARGE_MAG_ERR>{MAX_MAG_ERR}")

    if np.isfinite(safe_float(row["snr"])) and row["snr"] < MIN_SNR:
        reasons.append(f"LOW_SNR<{MIN_SNR}")

    if np.isfinite(safe_float(row["exposure"])) and row["exposure"] <= 0:
        reasons.append("NONPOSITIVE_EXPOSURE")

    if (row["obsid"], row["filter"]) in fieldstar_exclusions:
        reasons.append("FIELDSTAR_VIOLATION")

    if (str(row["obsid"]), str(row["filter"]).upper()) in prescan_exclusions:
        reasons.append("PRESCAN_MAG99")

    # Manual review flags do not exclude by themselves
    if np.isfinite(safe_float(row["mag_err"])) and 0.3 < row["mag_err"] <= MAX_MAG_ERR:
        review_reasons.append("CHECK_MODERATE_MAG_ERR")

    if np.isfinite(safe_float(row["snr"])) and MIN_SNR <= row["snr"] < 5:
        review_reasons.append("CHECK_BORDERLINE_SNR")

    if pd.isna(row["orbit_id"]):
        review_reasons.append("CHECK_ORBIT_UNASSIGNED")

    if reasons:
        df.at[idx, "exclude_science"] = True
        df.at[idx, "exclude_reason"] = ";".join(reasons)
        exclude_rows.append({
            "obsid": row["obsid"],
            "filter": row["filter"],
            "mjd": row["mjd"],
            "date": row["date"],
            "dt_days": row["dt_days"],
            "orbit_id": row["orbit_id"],
            "mag_vega": row["mag_vega"],
            "mag_err": row["mag_err"],
            "snr": row["snr"],
            "reason": ";".join(reasons),
        })
        log_entry(row["obsid"], row["filter"], f"EXCLUDED {df.at[idx, 'exclude_reason']}")

    if review_reasons and not reasons:
        df.at[idx, "review_flag"] = True
        df.at[idx, "review_reason"] = ";".join(review_reasons)
        review_rows.append({
            "obsid": row["obsid"],
            "filter": row["filter"],
            "mjd": row["mjd"],
            "date": row["date"],
            "dt_days": row["dt_days"],
            "orbit_id": row["orbit_id"],
            "mag_vega": row["mag_vega"],
            "mag_err": row["mag_err"],
            "snr": row["snr"],
            "reason": ";".join(review_reasons),
        })
        log_entry(row["obsid"], row["filter"], f"REVIEW {df.at[idx, 'review_reason']}")

# ==========================================================
# TABLE OUTPUTS
# ==========================================================

df.to_csv(os.path.join(TABLE_DIR, "uvot_lightcurve_all.csv"), index=False)

df_lim = df[df["coi_limited"] == True].copy()
df_lim.to_csv(os.path.join(TABLE_DIR, "uvot_lightcurve_limits.csv"), index=False)

# science detections exclude clearly bad points
df_det = df[df["exclude_science"] == False].copy()
df_det.to_csv(os.path.join(TABLE_DIR, "uvot_lightcurve_detections.csv"), index=False)

# save excluded list
with open(EXCLUDE_FILE, "w") as f:
    f.write("# Automatically excluded from science light curves/tables\n")
    f.write(f"# Criteria: coincidence-limited OR mag_err>{MAX_MAG_ERR} OR snr<{MIN_SNR} (if available) OR non-finite values OR FIELDSTAR_VIOLATION OR PRESCAN_MAG99\n")
    f.write("#\n")
    f.write(f"# {'ObsID':<15} {'Filter':<8} {'MJD':>12} {'Date':<20} {'dt_days':>10} {'Orbit':>6} "
            f"{'Mag':>8} {'Err':>7} {'SNR':>8} {'Reason'}\n")
    f.write("# " + "-" * 150 + "\n")
    for r in exclude_rows:
        orbit_str = str(int(r["orbit_id"])) if pd.notna(r["orbit_id"]) else "N/A"
        dt_str = f"{r['dt_days']:+.2f}" if pd.notna(r["dt_days"]) else "N/A"
        snr_str = f"{r['snr']:.2f}" if pd.notna(r["snr"]) else "N/A"
        date_str = str(r["date"])[:19]
        f.write(
            f"  {r['obsid']:<15} {r['filter']:<8} {r['mjd']:>12.4f} {date_str:<20} "
            f"{dt_str:>10} {orbit_str:>6} {r['mag_vega']:>8.3f} {r['mag_err']:>7.3f} "
            f"{snr_str:>8} {r['reason']}\n"
        )

# save manual review list
with open(REVIEW_FILE, "w") as f:
    f.write("# Manual review list for DS9 inspection\n")
    f.write("# These are not automatically excluded, but should be checked if time allows.\n")
    f.write("#\n")
    f.write(f"# {'ObsID':<15} {'Filter':<8} {'MJD':>12} {'Date':<20} {'dt_days':>10} {'Orbit':>6} "
            f"{'Mag':>8} {'Err':>7} {'SNR':>8} {'Reason'}\n")
    f.write("# " + "-" * 150 + "\n")
    for r in review_rows:
        orbit_str = str(int(r["orbit_id"])) if pd.notna(r["orbit_id"]) else "N/A"
        dt_str = f"{r['dt_days']:+.2f}" if pd.notna(r["dt_days"]) else "N/A"
        snr_str = f"{r['snr']:.2f}" if pd.notna(r["snr"]) else "N/A"
        date_str = str(r["date"])[:19]
        f.write(
            f"  {r['obsid']:<15} {r['filter']:<8} {r['mjd']:>12.4f} {date_str:<20} "
            f"{dt_str:>10} {orbit_str:>6} {r['mag_vega']:>8.3f} {r['mag_err']:>7.3f} "
            f"{snr_str:>8} {r['reason']}\n"
        )

# append coincidence-limited points to pipeline log
with open(LOG_FILE, "a") as logf:
    for _, row in df_lim.iterrows():
        logf.write(
            f"LIGHTCURVE | {row['obsid']} | {row['filter']} | "
            f"COI_LIMITED mag={row['mag_vega']:.3f} limit={row['mag_coi_lim']:.3f}\n"
        )

print(f"\nAll rows: {len(df)}")
print(f"Science detections: {len(df_det)}")
print(f"Coincidence-limited: {len(df_lim)}")
print(f"Excluded: {len(exclude_rows)} -> {EXCLUDE_FILE}")
print(f"  of which prescan-excluded: {sum(1 for r in exclude_rows if 'PRESCAN_MAG99' in r['reason'])}")
print(f"Manual review: {len(review_rows)} -> {REVIEW_FILE}")
print("Filters in science detections:", sorted(df_det["filter"].unique()))

# ==========================================================
# OUTLIER DETECTION (report only; do NOT remove from plots)
# ==========================================================

df_det["outlier"] = False
outlier_rows = []

for flt in df_det["filter"].unique():
    sub = df_det[df_det["filter"] == flt].copy()

    valid = (
        sub["mag_vega"].notna()
        & sub["mag_err"].notna()
        & np.isfinite(sub["mag_vega"])
        & np.isfinite(sub["mag_err"])
    )
    sub_valid = sub[valid].copy()

    if len(sub_valid) < 2:
        continue

    mags = sub_valid["mag_vega"].to_numpy(dtype=float)
    errs = sub_valid["mag_err"].to_numpy(dtype=float)
    errs = np.clip(errs, 1e-6, None)

    weights = 1.0 / errs**2
    wmean = np.sum(weights * mags) / np.sum(weights)
    wstd = np.sqrt(np.sum(weights * (mags - wmean)**2) / np.sum(weights))

    if not np.isfinite(wstd) or wstd <= 0:
        continue

    for ix, mag, err in zip(sub_valid.index, mags, errs):
        deviation = abs(mag - wmean)
        sigma = deviation / wstd
        if sigma > OUTLIER_SIGMA:
            df_det.at[ix, "outlier"] = True
            outlier_rows.append({
                "obsid": df_det.at[ix, "obsid"],
                "filter": flt,
                "mjd": df_det.at[ix, "mjd"],
                "date": df_det.at[ix, "date"],
                "dt_days": df_det.at[ix, "dt_days"],
                "orbit_id": df_det.at[ix, "orbit_id"],
                "mag_vega": mag,
                "mag_err": err,
                "wmean": wmean,
                "deviation": deviation,
                "sigma": sigma,
            })
            log_entry(df_det.at[ix, "obsid"], flt, f"OUTLIER {sigma:.2f}SIGMA")

with open(OUTLIER_FILE, "w") as f:
    f.write("# Outlier report — generated each pipeline run\n")
    f.write("# Report only: outliers remain in the plots unless manually excluded elsewhere.\n")
    f.write(f"# Sigma threshold: {OUTLIER_SIGMA}\n")
    f.write(f"# Total outliers flagged: {len(outlier_rows)}\n")
    f.write("#\n")
    f.write(f"# {'ObsID':<15} {'Filter':<8} {'MJD':>12} {'Date':<20} {'dt_days':>10} {'Orbit':>6} "
            f"{'Mag':>8} {'Err':>7} {'WMean':>8} {'Deviation':>10} {'Sigma':>7}\n")
    f.write("# " + "-" * 150 + "\n")
    for r in outlier_rows:
        orbit_str = str(int(r["orbit_id"])) if pd.notna(r["orbit_id"]) else "N/A"
        dt_str = f"{r['dt_days']:+.2f}" if pd.notna(r["dt_days"]) else "N/A"
        date_str = str(r["date"])[:19]
        f.write(
            f"  {r['obsid']:<15} {r['filter']:<8} {r['mjd']:>12.4f} {date_str:<20} "
            f"{dt_str:>10} {orbit_str:>6} {r['mag_vega']:>8.3f} {r['mag_err']:>7.3f} "
            f"{r['wmean']:>8.3f} {r['deviation']:>10.3f} {r['sigma']:>7.2f}\n"
        )

print(f"\nOutlier flagging: {len(outlier_rows)} point(s) flagged at >{OUTLIER_SIGMA}σ -> {OUTLIER_FILE}")

# ==========================================================
# EXCLUSION PLOT — where in the orbit were observations removed and why?
# Shows dt_days vs filter, coloured by exclusion reason.
# Helps identify whether removals cluster around disc crossings.
# ==========================================================

os.makedirs(os.path.join(PLOT_DIR, "quality"), exist_ok=True)

EXCLUDE_REASON_COLORS = {
    "COI_LIMITED":        "#e41a1c",   # red
    "FIELDSTAR_VIOLATION":"#ff7f00",   # orange
    "LARGE_MAG_ERR":      "#984ea3",   # purple
    "LOW_SNR":            "#377eb8",   # blue
    "PRESCAN_MAG99":      "#000000",   # black
    "OTHER":              "#888888",   # grey
}

def simplify_reason(reason_str):
    """Return the primary reason category for colour coding."""
    if "COI_LIMITED" in reason_str:
        return "COI_LIMITED"
    if "FIELDSTAR_VIOLATION" in reason_str:
        return "FIELDSTAR_VIOLATION"
    if "LARGE_MAG_ERR" in reason_str:
        return "LARGE_MAG_ERR"
    if "LOW_SNR" in reason_str:
        return "LOW_SNR"
    if "PRESCAN_MAG99" in reason_str:
        return "PRESCAN_MAG99"
    return "OTHER"

if exclude_rows:
    exc_df = pd.DataFrame(exclude_rows)
    exc_df["dt_days"] = pd.to_numeric(exc_df["dt_days"], errors="coerce")
    exc_df["reason_simple"] = exc_df["reason"].apply(simplify_reason)

    # Filter to orbital window only — unassigned dt_days shown separately
    exc_orbit = exc_df.dropna(subset=["dt_days"]).copy()
    exc_no_orbit = exc_df[exc_df["dt_days"].isna()].copy()

    # --- Static PNG ---
    fig_exc, ax_exc = plt.subplots(figsize=(12, 5))

    plotted_reasons = set()
    for reason_key, color in EXCLUDE_REASON_COLORS.items():
        sub = exc_orbit[exc_orbit["reason_simple"] == reason_key]
        if sub.empty:
            continue
        for flt_idx, flt in enumerate(FILTER_ORDER):
            fsub = sub[sub["filter"] == flt]
            if fsub.empty:
                continue
            label = reason_key if reason_key not in plotted_reasons else "_nolegend_"
            plotted_reasons.add(reason_key)
            ax_exc.scatter(
                fsub["dt_days"],
                [flt] * len(fsub),
                color=color,
                marker="x",
                s=60,
                linewidths=1.5,
                label=label,
                zorder=3,
            )

    ax_exc.axvline(0, linestyle="--", color="grey", alpha=0.5)

    ax_exc.set_xlabel("Days from periastron")
    ax_exc.set_ylabel("Filter")
    ax_exc.set_title(f"{SOURCE_NAME} — Excluded observations by orbital phase and reason")
    ax_exc.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax_exc.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "quality", "excluded_by_orbit_phase.png"), dpi=150)
    plt.close()

    # --- Interactive Plotly HTML ---
    IPLOT_DIR = os.path.join(PLOT_DIR, "interactive")
    fig_iexc = go.Figure()

    for reason_key, color in EXCLUDE_REASON_COLORS.items():
        sub = exc_orbit[exc_orbit["reason_simple"] == reason_key]
        if sub.empty:
            continue
        hover = [
            f"<b>ObsID:</b> {r['obsid']}<br>"
            f"<b>Filter:</b> {r['filter']}<br>"
            f"<b>MJD:</b> {r['mjd']:.2f}<br>"
            f"<b>Δt (days):</b> {r['dt_days']:+.1f}<br>"
            f"<b>Reason:</b> {r['reason']}"
            for _, r in sub.iterrows()
        ]
        fig_iexc.add_trace(go.Scatter(
            x=sub["dt_days"],
            y=sub["filter"],
            mode="markers",
            name=reason_key,
            marker=dict(symbol="x", size=10, color=color,
                        line=dict(width=2, color=color)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    fig_iexc.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)

    fig_iexc.update_layout(
        title=f"{SOURCE_NAME} — Excluded observations by orbital phase",
        xaxis_title="Days from periastron",
        yaxis_title="Filter",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend=dict(itemsizing="constant"),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig_iexc.write_html(
        os.path.join(IPLOT_DIR, "excluded_by_orbit_phase.html"),
        include_plotlyjs="cdn"
    )

    print(f"\nExclusion plot saved -> {os.path.join(PLOT_DIR, 'quality', 'excluded_by_orbit_phase.png')}")
    print(f"Exclusion plot (interactive) -> {os.path.join(IPLOT_DIR, 'excluded_by_orbit_phase.html')}")
    print(f"  {len(exc_orbit)} excluded points with orbital assignment plotted")
    if len(exc_no_orbit):
        print(f"  {len(exc_no_orbit)} excluded points had no orbit assignment (not plotted)")
else:
    print("\nNo excluded observations to plot.")

# ==========================================================
# PLOT DATA
# ==========================================================

# Use science detections only, but do NOT remove statistical outliers automatically
plot_df = df_det.copy()

# ==========================================================
# PLOT 1: FULL MJD LIGHTCURVE
# ==========================================================

plt.figure(figsize=(10, 6))

for flt in FILTER_ORDER:
    sub = plot_df[plot_df["filter"] == flt]
    if sub.empty:
        continue

    plt.errorbar(
        sub["mjd"],
        sub["mag_vega"],
        yerr=sub["mag_err"],
        fmt=MARKERS.get(flt, "o"),
        linestyle="none",
        label=filter_label(flt),
        alpha=0.9,
        capsize=2,
    )

ax = plt.gca()
ax.invert_yaxis()

plt.xlabel("MJD")
plt.ylabel("Vega magnitude")
plt.title(f"{SOURCE_NAME} — Swift/UVOT multi-filter light curve")
plt.legend(ncol=4, frameon=False)
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "uvot_lightcurve_full_mjd.png"), dpi=150)
plt.close()

# ==========================================================
# PLOT SET A: BY FILTER
# ==========================================================

for flt in sorted(plot_df["filter"].unique()):
    sub = plot_df[(plot_df["filter"] == flt) & (plot_df["orbit_id"].notna())].copy()
    if sub.empty:
        continue

    plt.figure(figsize=(10, 6))

    for orbit in sorted(sub["orbit_id"].unique()):
        orbsub = sub[sub["orbit_id"] == orbit].sort_values("dt_days")
        if orbsub.empty:
            continue

        plt.errorbar(
            orbsub["dt_days"],
            orbsub["mag_vega"],
            yerr=orbsub["mag_err"],
            fmt="o",
            linestyle="none",
            capsize=2,
            label=orbit_label(orbit),
            alpha=0.9,
        )
        plt.plot(
            orbsub["dt_days"],
            orbsub["mag_vega"],
            linewidth=1.0,
            alpha=0.25,
        )

    plt.axvline(0, linestyle="--", alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel("Days from periastron")
    plt.ylabel("Vega magnitude")
    plt.title(f"{flt} — orbit comparison")
    plt.legend(frameon=False)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "by_filter", f"{flt}_by_orbit.png"), dpi=150)
    plt.close()

# ==========================================================
# PLOT SET B: BY ORBIT
# ==========================================================

for orbit in sorted(plot_df["orbit_id"].dropna().unique()):
    sub = plot_df[plot_df["orbit_id"] == orbit].copy()
    if sub.empty:
        continue

    try:
        orbit_year = int(Time(float(PERIASTRON_MJDS[int(orbit)]), format="mjd").datetime.year)
    except Exception:
        orbit_year = "?"

    plt.figure(figsize=(10, 6))

    for flt in FILTER_ORDER:
        fltsub = sub[sub["filter"] == flt].sort_values("dt_days")
        if fltsub.empty:
            continue

        plt.errorbar(
            fltsub["dt_days"],
            fltsub["mag_vega"],
            yerr=fltsub["mag_err"],
            fmt=MARKERS.get(flt, "o"),
            linestyle="none",
            capsize=2,
            label=filter_label(flt),
            alpha=0.9,
        )
        plt.plot(
            fltsub["dt_days"],
            fltsub["mag_vega"],
            linewidth=1.0,
            alpha=0.25,
        )

    plt.axvline(0, linestyle="--", alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel("Days from periastron")
    plt.ylabel("Vega magnitude")
    plt.title(f"Orbit {int(orbit)} ({orbit_year}) — multi-filter")
    plt.legend(ncol=3, frameon=False)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "by_orbit", f"orbit_{int(orbit)}.png"), dpi=150)
    plt.close()

# ==========================================================
# INTERACTIVE PLOTS
# ==========================================================

IPLOT_DIR = os.path.join(PLOT_DIR, "interactive")

def _hover_text(row):
    lines = [
        f"<b>ObsID:</b> {row.get('obsid', '?')}",
        f"<b>Filter:</b> {row.get('filter', '?')}",
        f"<b>MJD:</b> {row['mjd']:.4f}",
    ]
    if pd.notna(row.get("date")):
        lines.append(f"<b>Date:</b> {str(row['date'])[:19]}")
    if pd.notna(row.get("dt_days")):
        lines.append(f"<b>Δt (days):</b> {row['dt_days']:+.2f}")
    if pd.notna(row.get("orbit_id")):
        lines.append(f"<b>Orbit:</b> {int(row['orbit_id'])}")

    lines.append(f"<b>Mag (Vega):</b> {row['mag_vega']:.3f} ± {row['mag_err']:.3f}")

    if pd.notna(row.get("snr")):
        lines.append(f"<b>SNR:</b> {row['snr']:.2f}")
    if bool(row.get("outlier", False)):
        lines.append("<b>Outlier flag:</b> yes")
    if bool(row.get("review_flag", False)):
        lines.append(f"<b>Review:</b> {row.get('review_reason', '')}")

    for extra in ("rate", "flux", "exposure", "mag_coi_lim"):
        if extra in row and pd.notna(row.get(extra)):
            lines.append(f"<b>{extra.upper()}:</b> {row[extra]:.4g}")

    return "<br>".join(lines)

def _plotly_layout(title, xlabel, invert_y=True):
    return dict(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title=xlabel, showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(
            title="Vega magnitude",
            showgrid=True,
            gridcolor="#e0e0e0",
            autorange="reversed" if invert_y else True
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(itemsizing="constant", tracegroupgap=4),
        hovermode="closest",
        margin=dict(l=60, r=30, t=60, b=60),
    )

def _add_vline(fig, x=0):
    fig.add_vline(x=x, line_dash="dash", line_color="grey", opacity=0.6)

fig = go.Figure()

for flt in FILTER_ORDER:
    sub = plot_df[plot_df["filter"] == flt]
    if sub.empty:
        continue

    hover = [_hover_text(r) for _, r in sub.iterrows()]
    fig.add_trace(go.Scatter(
        x=sub["mjd"],
        y=sub["mag_vega"],
        mode="markers",
        name=filter_label(flt),
        marker=dict(
            symbol=PLOTLY_SYMBOLS.get(flt, "circle"),
            color=FILTER_COLORS.get(flt, "#888"),
            size=7,
            line=dict(width=0.5, color="white")
        ),
        error_y=dict(
            type="data",
            array=sub["mag_err"].tolist(),
            visible=True,
            thickness=1.2,
            width=3
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

fig.update_layout(**_plotly_layout(
    f"{SOURCE_NAME} — Swift/UVOT multi-filter light curve", "MJD"
))
fig.write_html(
    os.path.join(IPLOT_DIR, "uvot_lightcurve_full_mjd.html"),
    include_plotlyjs="cdn"
)

for flt in sorted(plot_df["filter"].unique()):
    sub = plot_df[(plot_df["filter"] == flt) & (plot_df["orbit_id"].notna())].copy()
    if sub.empty:
        continue

    fig = go.Figure()

    for orbit in sorted(sub["orbit_id"].unique()):
        orbsub = sub[sub["orbit_id"] == orbit].sort_values("dt_days")
        if orbsub.empty:
            continue

        hover = [_hover_text(r) for _, r in orbsub.iterrows()]
        fig.add_trace(go.Scatter(
            x=orbsub["dt_days"],
            y=orbsub["mag_vega"],
            mode="markers+lines",
            name=orbit_label(orbit),
            marker=dict(size=7, line=dict(width=0.5, color="white")),
            line=dict(width=1, dash="dot"),
            error_y=dict(
                type="data",
                array=orbsub["mag_err"].tolist(),
                visible=True,
                thickness=1.2,
                width=3
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    _add_vline(fig)
    fig.update_layout(**_plotly_layout(
        f"{flt} — orbit comparison", "Days from periastron"
    ))
    fig.write_html(
        os.path.join(IPLOT_DIR, "by_filter", f"{flt}_by_orbit.html"),
        include_plotlyjs="cdn",
    )

for orbit in sorted(plot_df["orbit_id"].dropna().unique()):
    sub = plot_df[plot_df["orbit_id"] == orbit].copy()
    if sub.empty:
        continue

    fig = go.Figure()

    for flt in FILTER_ORDER:
        fltsub = sub[sub["filter"] == flt].sort_values("dt_days")
        if fltsub.empty:
            continue

        hover = [_hover_text(r) for _, r in fltsub.iterrows()]
        fig.add_trace(go.Scatter(
            x=fltsub["dt_days"],
            y=fltsub["mag_vega"],
            mode="markers+lines",
            name=filter_label(flt),
            marker=dict(
                symbol=PLOTLY_SYMBOLS.get(flt, "circle"),
                color=FILTER_COLORS.get(flt, "#888"),
                size=7,
                line=dict(width=0.5, color="white")
            ),
            line=dict(color=FILTER_COLORS.get(flt, "#888"), width=1, dash="dot"),
            error_y=dict(
                type="data",
                array=fltsub["mag_err"].tolist(),
                visible=True,
                thickness=1.2,
                width=3
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    _add_vline(fig)
    fig.update_layout(**_plotly_layout(
        f"{orbit_label(orbit)} — multi-filter", "Days from periastron"
    ))
    fig.write_html(
        os.path.join(IPLOT_DIR, "by_orbit", f"orbit_{int(orbit)}.html"),
        include_plotlyjs="cdn",
    )

# ==========================================================
# FIELD STAR SYSTEMATIC CHECK
# (flagging done earlier — results already applied to plot_df)
# ==========================================================

# ==========================================================
# MAG=99 DETECTOR DEGRADATION CHART (PER ORBIT COUNTS)
# Shows total number of prescan MAG=99 exclusions in each
# periastron passage / orbit.
# ==========================================================

if prescan_exclusions:
    # Map obsid -> earliest MJD from the full dataframe
    obsid_mjd = (
        df.groupby("obsid")["mjd"].min().reset_index()
        .rename(columns={"mjd": "obs_mjd"})
    )
    obsid_mjd_map = dict(zip(obsid_mjd["obsid"], obsid_mjd["obs_mjd"]))

    mag99_rows = []
    for obsid, flt in prescan_exclusions:
        mjd_val = obsid_mjd_map.get(obsid, np.nan)
        if pd.isna(mjd_val):
            continue

        orbit_id, peri_mjd = assign_orbit(mjd_val)
        if orbit_id is None:
            continue

        try:
            orbit_year = int(Time(float(PERIASTRON_MJDS[int(orbit_id)]), format="mjd").datetime.year)
        except Exception:
            orbit_year = None

        mag99_rows.append({
            "obsid": obsid,
            "filter": flt,
            "mjd": mjd_val,
            "orbit_id": int(orbit_id),
            "orbit_year": orbit_year,
        })

    mag99_df = pd.DataFrame(mag99_rows)

    if not mag99_df.empty:
        # Create full orbit list FIRST (this is the key fix)
        all_orbits = []
        for i, mjd in enumerate(PERIASTRON_MJDS):
            try:
                year = int(Time(float(mjd), format="mjd").datetime.year)
            except Exception:
                year = None

            all_orbits.append({
                "orbit_id": i,
                "orbit_year": year
            })

        all_orbits_df = pd.DataFrame(all_orbits)

        # Count MAG99 per orbit (as before)
        counts = (
            mag99_df.groupby("orbit_id")
            .size()
            .reset_index(name="mag99_count")
        )

        # Merge → fills missing orbits with NaN → replace with 0
        orbit_counts = all_orbits_df.merge(counts, on="orbit_id", how="left")
        orbit_counts["mag99_count"] = orbit_counts["mag99_count"].fillna(0)

        # Sort properly
        orbit_counts = orbit_counts.sort_values("orbit_id").reset_index(drop=True)

        orbit_counts["x_label"] = orbit_counts.apply(
            lambda r: f"Orbit {int(r['orbit_id'])}\n({int(r['orbit_year'])})"
            if pd.notna(r["orbit_year"]) else f"Orbit {int(r['orbit_id'])}",
            axis=1
        )

        xvals = np.arange(len(orbit_counts))

        # --- Static PNG ---
        fig_m99, ax_m99 = plt.subplots(figsize=(10, 6))

        ax_m99.plot(
            xvals,
            orbit_counts["mag99_count"],
            color="black",
            linewidth=1.5,
            zorder=2,
        )

        ax_m99.scatter(
            xvals,
            orbit_counts["mag99_count"],
            color="red",
            s=45,
            zorder=3,
            label="MAG=99 exclusions",
        )
        ax_m99.set_ylim(bottom=0)
        ax_m99.set_xticks(xvals)
        ax_m99.set_xticklabels(orbit_counts["x_label"])
        ax_m99.set_xlabel("Periastron passage")
        ax_m99.set_ylabel("Number of MAG=99 exclusions")
        ax_m99.set_title(f"{SOURCE_NAME} — MAG=99 exclusions by periastron passage")
        ax_m99.legend(frameon=False)
        ax_m99.grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "quality", "mag99_per_year.png"), dpi=150)
        plt.close()

        # --- Interactive Plotly ---
        fig_m99i = go.Figure()

        hover = [
            f"<b>Orbit:</b> {int(r['orbit_id'])}<br>"
            f"<b>Year:</b> {int(r['orbit_year']) if pd.notna(r['orbit_year']) else 'N/A'}<br>"
            f"<b>MAG=99 count:</b> {int(r['mag99_count'])}"
            for _, r in orbit_counts.iterrows()
        ]

        fig_m99i.add_trace(go.Scatter(
            x=orbit_counts["x_label"].tolist(),
            y=orbit_counts["mag99_count"].tolist(),
            mode="lines+markers",
            name="MAG=99 exclusions",
            line=dict(color="black", width=2),
            marker=dict(color="red", size=9),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

        fig_m99i.update_layout(
            title=f"{SOURCE_NAME} — MAG=99 exclusions by periastron passage",
            xaxis_title="Periastron passage",
            yaxis_title="Number of MAG=99 exclusions",
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            margin=dict(l=60, r=30, t=60, b=60),
        )

        fig_m99i.write_html(
            os.path.join(IPLOT_DIR, "mag99_per_year.html"),
            include_plotlyjs="cdn",
        )

        print(f"\nMAG=99 per-orbit chart -> {os.path.join(PLOT_DIR, 'quality', 'mag99_per_year.png')}")
        print(f"MAG=99 per-orbit chart (interactive) -> {os.path.join(IPLOT_DIR, 'mag99_per_year.html')}")
    else:
        print("\nMAG=99 exclusions found but none could be assigned to an orbit — chart skipped.")
else:
    print("\nNo MAG=99 prescan exclusions recorded — chart not produced.")

print(f"  all table            -> {os.path.join(TABLE_DIR, 'uvot_lightcurve_all.csv')}")
print(f"  detections table     -> {os.path.join(TABLE_DIR, 'uvot_lightcurve_detections.csv')}")
print(f"  limits table         -> {os.path.join(TABLE_DIR, 'uvot_lightcurve_limits.csv')}")
print(f"  excluded list        -> {EXCLUDE_FILE}")
print(f"  manual review list   -> {REVIEW_FILE}")
print(f"  outlier report       -> {OUTLIER_FILE}")
print(f"  field star flags     -> {FIELD_FLAGS_FILE}")
print(f"  plots (png)          -> {PLOT_DIR}")
print(f"  plots (html)         -> {IPLOT_DIR}")
print(f"  exclusion plot (png) -> {os.path.join(PLOT_DIR, 'quality', 'excluded_by_orbit_phase.png')}")
print(f"  exclusion plot (html)-> {os.path.join(IPLOT_DIR, 'excluded_by_orbit_phase.html')}")
print(f"  mag99 chart (png)    -> {os.path.join(PLOT_DIR, 'quality', 'mag99_per_year.png')}")
print(f"  mag99 chart (html)   -> {os.path.join(IPLOT_DIR, 'mag99_per_year.html')}")
print("Done.")