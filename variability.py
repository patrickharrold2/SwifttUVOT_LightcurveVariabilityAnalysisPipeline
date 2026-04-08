#!/usr/bin/env python3
# variability.py
#
# Variability analysis for Swift/UVOT light curves of PSR B1259-63.
# Reads from products/tables/uvot_lightcurve_detections.csv (produced by lightcurve.py).
#
# Analyses:
#   1. Within-orbit chi-squared — variability within each orbital passage per filter
#   2. Orbit-to-orbit chi-squared — is the mean brightness consistent across orbits per filter
#   3. Disc-crossing segment chi-squared — variability within each physical phase of the orbit
#   4. Active-phase orbit-to-orbit chi-squared — compare mean brightness within ±100 d across years
#   5. Active-phase within-orbit chi-squared — variability within ±100 d of periastron for each orbit/filter
#
# Outputs (all in products/tables/, all overwritten each run):
#   variability_within_orbit.csv
#   variability_orbit_to_orbit.csv
#   variability_disc_crossing.csv
#   variability_active_phase.csv
#   variability_active_phase_within_orbit.csv
#   variability_summary.txt

import subprocess
import sys

for pkg in ["numpy", "pandas", "scipy", "astropy"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import os
import warnings
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy import stats

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURE FOR YOUR SOURCE
# Keep in sync with lightcurve.py
# ==========================================================

SOURCE_NAME = "PSR B1259–63"

PERIASTRON_MJDS = sorted([
    54307.969,   # 2007 periastron (Johnston et al. orbital period extrapolation)
    55544.694,   # 2010 periastron (Chang et al. 2019)
    56781.418,   # 2014 periastron (Chang et al. 2019)
    58018.143,   # 2017 periastron (Chang et al. 2019)
    59254.867,   # 2021 periastron (Chernyakova et al. 2021)
    60491.592,   # 2024 periastron (Chernyakova et al. 2024)
])

# ==========================================================
# DISC CROSSING SEGMENTS
# Physically motivated boundaries from the literature.
# ==========================================================

SEGMENTS = [
    ("pre_approach",     r"dt < -30 d",           None,   -30.0),
    ("first_crossing",   r"-30 to -16 d",         -30.0,  -16.0),
    ("periastron",       r"-16 to +15 d",         -16.0,  +15.0),
    ("second_crossing",  r"+15 to +30 d",         +15.0,  +30.0),
    ("flare_window",     r"+30 to +100 d",        +30.0, +100.0),
    ("quiescence",       r"dt > +100 d",          +100.0,  None),
]

# ==========================================================
# CUSTOM PERIASTRON-RELATIVE RANGES
# User-editable windows for source-specific tests
# Bounds are applied inclusively: dt_days >= start and dt_days <= end
# ==========================================================

CUSTOM_RANGES = [
    ("custom_m30_p40", "-30 to +40 d", -30.0, +40.0),
    ("custom_0_p40",   "17 to +40 d",     17.0, +40.0),
    ("custom_m30_p41", "-30 to +41 d", -30.0, +41.0),
    ("custom_0_p41",   "0 to +41 d",     0.0, +41.0),
    ("custom_0_p41",   "-9 to +70 d",     -9.0, +70.0),
]

MIN_POINTS = 3
ACTIVE_PHASE_DAYS = 100.0  # ±100 days from periastron

# ==========================================================
# PATHS
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR  = os.path.join(SCRIPT_DIR, "../products/tables")
DET_CSV    = os.path.join(TABLE_DIR, "uvot_lightcurve_detections.csv")

os.makedirs(TABLE_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

if not os.path.exists(DET_CSV):
    print(f"ERROR: detections CSV not found at {DET_CSV}")
    print("       Run lightcurve.py first.")
    sys.exit(1)

df = pd.read_csv(DET_CSV)

required = {"filter", "mjd", "mag_vega", "mag_err", "orbit_id", "dt_days"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: detections CSV is missing columns: {missing}")
    sys.exit(1)

df["orbit_id"] = pd.to_numeric(df["orbit_id"], errors="coerce")
df["dt_days"]  = pd.to_numeric(df["dt_days"],  errors="coerce")
df["mag_vega"] = pd.to_numeric(df["mag_vega"], errors="coerce")
df["mag_err"]  = pd.to_numeric(df["mag_err"],  errors="coerce").clip(lower=1e-6)

df = df.dropna(subset=["mag_vega", "mag_err", "mjd"]).reset_index(drop=True)

FILTERS = sorted(df["filter"].unique())

def periastron_year(orbit_id):
    try:
        return int(Time(float(PERIASTRON_MJDS[int(orbit_id)]), format="mjd").datetime.year)
    except Exception:
        return None

# ==========================================================
# CHI-SQUARED HELPER
# ==========================================================

def chi_squared(mags, errs):
    """
    Compute chi-squared against a weighted mean (constant source hypothesis).

    Returns:
        chi2       : raw chi-squared value
        dof        : degrees of freedom (N - 1)
        reduced    : chi2 / dof
        p_value    : probability of obtaining this chi2 if source is constant
        n          : number of points used
        wmean      : weighted mean magnitude
        verdict    : 'VARIABLE', 'POSSIBLY VARIABLE', 'CONSTANT', or 'INSUFFICIENT DATA'
    """
    mags = np.asarray(mags, dtype=float)
    errs = np.asarray(errs, dtype=float).clip(min=1e-6)
    n = len(mags)

    if n < MIN_POINTS:
        return dict(
            chi2=np.nan,
            dof=np.nan,
            reduced=np.nan,
            p_value=np.nan,
            n=n,
            wmean=np.nan,
            verdict="INSUFFICIENT DATA"
        )

    weights = 1.0 / errs**2
    wmean   = np.sum(weights * mags) / np.sum(weights)
    chi2    = np.sum(((mags - wmean) / errs) ** 2)
    dof     = n - 1
    reduced = chi2 / dof
    p_value = 1.0 - stats.chi2.cdf(chi2, dof)

    if p_value < 0.01 and reduced > 2.0:
        verdict = "VARIABLE"
    elif p_value < 0.05 or reduced > 1.5:
        verdict = "POSSIBLY VARIABLE"
    else:
        verdict = "CONSTANT"

    return dict(
        chi2=chi2,
        dof=dof,
        reduced=reduced,
        p_value=p_value,
        n=n,
        wmean=wmean,
        verdict=verdict
    )

# ==========================================================
# ANALYSIS 1: WITHIN-ORBIT VARIABILITY
# ==========================================================

print("\n=== Analysis 1: Within-orbit variability ===")
within_rows = []

orb_df = df.dropna(subset=["orbit_id", "dt_days"])

for flt in FILTERS:
    for orbit in sorted(orb_df["orbit_id"].dropna().unique()):
        sub = orb_df[(orb_df["filter"] == flt) & (orb_df["orbit_id"] == orbit)]
        if sub.empty:
            continue

        year = periastron_year(orbit)
        res = chi_squared(sub["mag_vega"].values, sub["mag_err"].values)

        within_rows.append({
            "filter":       flt,
            "orbit_id":     int(orbit),
            "year":         year,
            "n_points":     res["n"],
            "wmean_mag":    round(res["wmean"], 4) if np.isfinite(res.get("wmean", np.nan)) else np.nan,
            "chi2":         round(res["chi2"], 4) if np.isfinite(res.get("chi2", np.nan)) else np.nan,
            "dof":          res["dof"],
            "reduced_chi2": round(res["reduced"], 4) if np.isfinite(res.get("reduced", np.nan)) else np.nan,
            "p_value":      round(res["p_value"], 6) if np.isfinite(res.get("p_value", np.nan)) else np.nan,
            "verdict":      res["verdict"],
        })

        print(
            f"  {flt:6s} | Orbit {int(orbit)} ({year}) | n={res['n']:3d} | "
            f"χ²/dof={res['reduced']:.2f} | p={res['p_value']:.4f} | {res['verdict']}"
        )

df_within = pd.DataFrame(within_rows)
df_within.to_csv(os.path.join(TABLE_DIR, "variability_within_orbit.csv"), index=False)
print(f"  -> saved variability_within_orbit.csv ({len(df_within)} rows)")

# ==========================================================
# ANALYSIS 2: ORBIT-TO-ORBIT VARIABILITY
# ==========================================================

print("\n=== Analysis 2: Orbit-to-orbit variability ===")
orbit_rows = []

orbit_means = []
for flt in FILTERS:
    for orbit in sorted(orb_df["orbit_id"].dropna().unique()):
        sub = orb_df[(orb_df["filter"] == flt) & (orb_df["orbit_id"] == orbit)]
        if len(sub) < MIN_POINTS:
            print(f"  {flt} | Orbit {int(orbit)}: skipped (n={len(sub)} < MIN_POINTS={MIN_POINTS})")
            continue

        weights   = 1.0 / sub["mag_err"].values**2
        wmean     = np.sum(weights * sub["mag_vega"].values) / np.sum(weights)
        wmean_err = 1.0 / np.sqrt(np.sum(weights))

        orbit_means.append({
            "filter":    flt,
            "orbit_id":  int(orbit),
            "year":      periastron_year(orbit),
            "wmean":     wmean,
            "wmean_err": wmean_err,
            "n_points":  len(sub),
        })

df_omeans = pd.DataFrame(orbit_means)
df_omeans.to_csv(os.path.join(TABLE_DIR, "variability_orbit_means.csv"), index=False)
print(f"  -> saved variability_orbit_means.csv ({len(df_omeans)} rows)")

print(f"\n  {'Filter':<8} {'Orbit':>6} {'Year':>6} {'N':>4} {'Wmean mag':>11} {'Wmean err':>11}")
print("  " + "-" * 52)
for _, r in df_omeans.iterrows():
    print(
        f"  {r['filter']:<8} {int(r['orbit_id']):>6} {int(r['year']) if r['year'] else 'N/A':>6} "
        f"{int(r['n_points']):>4} {r['wmean']:>11.4f} {r['wmean_err']:>11.4f}"
    )

for flt in FILTERS:
    sub = df_omeans[df_omeans["filter"] == flt]
    if len(sub) < 2:
        continue

    res = chi_squared(sub["wmean"].values, sub["wmean_err"].values)

    orbit_rows.append({
        "filter":           flt,
        "n_orbits":         len(sub),
        "orbits_included":  ", ".join(str(int(o)) for o in sub["orbit_id"]),
        "years_included":   ", ".join(str(y) for y in sub["year"]),
        "chi2":             round(res["chi2"], 4) if np.isfinite(res.get("chi2", np.nan)) else np.nan,
        "dof":              res["dof"],
        "reduced_chi2":     round(res["reduced"], 4) if np.isfinite(res.get("reduced", np.nan)) else np.nan,
        "p_value":          round(res["p_value"], 6) if np.isfinite(res.get("p_value", np.nan)) else np.nan,
        "verdict":          res["verdict"],
    })

    print(
        f"  {flt:6s} | {len(sub)} orbits | χ²/dof={res['reduced']:.2f} | "
        f"p={res['p_value']:.4f} | {res['verdict']}"
    )

df_orbit = pd.DataFrame(orbit_rows)
df_orbit.to_csv(os.path.join(TABLE_DIR, "variability_orbit_to_orbit.csv"), index=False)
print(f"  -> saved variability_orbit_to_orbit.csv ({len(df_orbit)} rows)")

# ==========================================================
# ANALYSIS 3: DISC-CROSSING SEGMENT VARIABILITY
# ==========================================================

print("\n=== Analysis 3: Disc-crossing segment variability ===")
seg_rows = []

for flt in FILTERS:
    for seg_name, seg_label, seg_lo, seg_hi in SEGMENTS:
        for orbit in sorted(orb_df["orbit_id"].dropna().unique()):
            sub = orb_df[(orb_df["filter"] == flt) & (orb_df["orbit_id"] == orbit)].copy()
            if sub.empty:
                continue

            if seg_lo is not None:
                sub = sub[sub["dt_days"] >= seg_lo]
            if seg_hi is not None:
                sub = sub[sub["dt_days"] < seg_hi]

            if sub.empty:
                continue

            year = periastron_year(orbit)
            res = chi_squared(sub["mag_vega"].values, sub["mag_err"].values)

            seg_rows.append({
                "filter":        flt,
                "segment":       seg_name,
                "segment_label": seg_label,
                "orbit_id":      int(orbit),
                "year":          year,
                "n_points":      res["n"],
                "wmean_mag":     round(res["wmean"], 4) if np.isfinite(res.get("wmean", np.nan)) else np.nan,
                "chi2":          round(res["chi2"], 4) if np.isfinite(res.get("chi2", np.nan)) else np.nan,
                "dof":           res["dof"],
                "reduced_chi2":  round(res["reduced"], 4) if np.isfinite(res.get("reduced", np.nan)) else np.nan,
                "p_value":       round(res["p_value"], 6) if np.isfinite(res.get("p_value", np.nan)) else np.nan,
                "verdict":       res["verdict"],
            })

df_seg = pd.DataFrame(seg_rows)
df_seg.to_csv(os.path.join(TABLE_DIR, "variability_disc_crossing.csv"), index=False)
print(f"  -> saved variability_disc_crossing.csv ({len(df_seg)} rows)")

for seg_name, seg_label, _, _ in SEGMENTS:
    print(f"\n  Segment: {seg_label}")
    sub = df_seg[df_seg["segment"] == seg_name]
    for flt in FILTERS:
        fsub = sub[sub["filter"] == flt]
        if fsub.empty:
            continue
        variable = (fsub["verdict"] == "VARIABLE").sum()
        possible = (fsub["verdict"] == "POSSIBLY VARIABLE").sum()
        constant = (fsub["verdict"] == "CONSTANT").sum()
        insuff   = (fsub["verdict"] == "INSUFFICIENT DATA").sum()
        print(
            f"    {flt:6s} | {len(fsub)} orbits | "
            f"VARIABLE={variable} POSSIBLY={possible} CONSTANT={constant} INSUFF={insuff}"
        )

# ==========================================================
# ANALYSIS 4: ACTIVE PHASE (±100 DAYS) ORBIT-TO-ORBIT VARIABILITY
# ==========================================================

print(f"\n=== Analysis 4: Active phase (±{int(ACTIVE_PHASE_DAYS)} days) orbit-to-orbit variability ===")
active_rows = []

active_df = orb_df[
    orb_df["dt_days"].notna() &
    (orb_df["dt_days"].abs() <= ACTIVE_PHASE_DAYS)
].copy()

active_means = []
for flt in FILTERS:
    for orbit in sorted(active_df["orbit_id"].dropna().unique()):
        sub = active_df[(active_df["filter"] == flt) & (active_df["orbit_id"] == orbit)]
        if len(sub) < MIN_POINTS:
            print(f"  {flt} | Orbit {int(orbit)}: skipped (n={len(sub)} < MIN_POINTS={MIN_POINTS})")
            continue

        weights   = 1.0 / sub["mag_err"].values**2
        wmean     = np.sum(weights * sub["mag_vega"].values) / np.sum(weights)
        wmean_err = 1.0 / np.sqrt(np.sum(weights))

        active_means.append({
            "filter":    flt,
            "orbit_id":  int(orbit),
            "year":      periastron_year(orbit),
            "wmean":     wmean,
            "wmean_err": wmean_err,
            "n_points":  len(sub),
        })

df_active_means = pd.DataFrame(active_means)
df_active_means.to_csv(os.path.join(TABLE_DIR, "variability_active_phase_orbit_means.csv"), index=False)
print(f"  -> saved variability_active_phase_orbit_means.csv ({len(df_active_means)} rows)")

print(f"\n  Per-orbit weighted means within ±{int(ACTIVE_PHASE_DAYS)} days of periastron:")
print(f"  {'Filter':<8} {'Orbit':>6} {'Year':>6} {'N':>4} {'Wmean mag':>11} {'Wmean err':>11}")
print("  " + "-" * 52)
for _, r in df_active_means.iterrows():
    print(
        f"  {r['filter']:<8} {int(r['orbit_id']):>6} {int(r['year']) if r['year'] else 'N/A':>6} "
        f"{int(r['n_points']):>4} {r['wmean']:>11.4f} {r['wmean_err']:>11.4f}"
    )

for flt in FILTERS:
    sub = df_active_means[df_active_means["filter"] == flt]
    if len(sub) < 2:
        continue

    res = chi_squared(sub["wmean"].values, sub["wmean_err"].values)

    active_rows.append({
        "filter":          flt,
        "n_orbits":        len(sub),
        "orbits_included": ", ".join(str(int(o)) for o in sub["orbit_id"]),
        "years_included":  ", ".join(str(y) for y in sub["year"]),
        "chi2":            round(res["chi2"], 4) if np.isfinite(res.get("chi2", np.nan)) else np.nan,
        "dof":             res["dof"],
        "reduced_chi2":    round(res["reduced"], 4) if np.isfinite(res.get("reduced", np.nan)) else np.nan,
        "p_value":         round(res["p_value"], 6) if np.isfinite(res.get("p_value", np.nan)) else np.nan,
        "verdict":         res["verdict"],
    })

    print(
        f"  {flt:6s} | {len(sub)} orbits | χ²/dof={res['reduced']:.2f} | "
        f"p={res['p_value']:.4f} | {res['verdict']}"
    )

df_active = pd.DataFrame(active_rows)
df_active.to_csv(os.path.join(TABLE_DIR, "variability_active_phase.csv"), index=False)
print(f"  -> saved variability_active_phase.csv ({len(df_active)} rows)")

# ==========================================================
# ANALYSIS 5: ACTIVE PHASE (±100 DAYS) WITHIN-ORBIT VARIABILITY
# ==========================================================

print(f"\n=== Analysis 5: Active phase (±{int(ACTIVE_PHASE_DAYS)} days) within-orbit variability ===")
active_within_rows = []

active_within_df = orb_df[
    orb_df["dt_days"].notna() &
    (orb_df["dt_days"].abs() <= ACTIVE_PHASE_DAYS)
].copy()

for flt in FILTERS:
    for orbit in sorted(active_within_df["orbit_id"].dropna().unique()):
        sub = active_within_df[
            (active_within_df["filter"] == flt) &
            (active_within_df["orbit_id"] == orbit)
        ].copy()

        if sub.empty:
            continue

        year = periastron_year(orbit)
        res = chi_squared(sub["mag_vega"].values, sub["mag_err"].values)

        active_within_rows.append({
            "filter":       flt,
            "orbit_id":     int(orbit),
            "year":         year,
            "n_points":     res["n"],
            "wmean_mag":    round(res["wmean"], 4) if np.isfinite(res.get("wmean", np.nan)) else np.nan,
            "chi2":         round(res["chi2"], 4) if np.isfinite(res.get("chi2", np.nan)) else np.nan,
            "dof":          res["dof"],
            "reduced_chi2": round(res["reduced"], 4) if np.isfinite(res.get("reduced", np.nan)) else np.nan,
            "p_value":      round(res["p_value"], 6) if np.isfinite(res.get("p_value", np.nan)) else np.nan,
            "verdict":      res["verdict"],
        })

        print(
            f"  {flt:6s} | Orbit {int(orbit)} ({year}) | n={res['n']:3d} | "
            f"χ²/dof={res['reduced']:.2f} | p={res['p_value']:.4f} | {res['verdict']}"
        )

df_active_within = pd.DataFrame(active_within_rows)
df_active_within.to_csv(os.path.join(TABLE_DIR, "variability_active_phase_within_orbit.csv"), index=False)
print(f"  -> saved variability_active_phase_within_orbit.csv ({len(df_active_within)} rows)")

# ==========================================================
# ANALYSIS 6: CUSTOM PERIASTRON-RELATIVE RANGE VARIABILITY
# ==========================================================

print("\n=== Analysis 6: Custom periastron-relative range variability ===")
custom_rows = []

for flt in FILTERS:
    for range_name, range_label, range_lo, range_hi in CUSTOM_RANGES:
        for orbit in sorted(orb_df["orbit_id"].dropna().unique()):
            sub = orb_df[
                (orb_df["filter"] == flt) &
                (orb_df["orbit_id"] == orbit)
            ].copy()

            if sub.empty:
                continue

            # Inclusive bounds so requested endpoints are included
            if range_lo is not None:
                sub = sub[sub["dt_days"] >= range_lo]
            if range_hi is not None:
                sub = sub[sub["dt_days"] <= range_hi]

            if sub.empty:
                continue

            year = periastron_year(orbit)
            res = chi_squared(sub["mag_vega"].values, sub["mag_err"].values)

            custom_rows.append({
                "filter":       flt,
                "custom_range": range_name,
                "range_label":  range_label,
                "range_start":  range_lo,
                "range_end":    range_hi,
                "orbit_id":     int(orbit),
                "year":         year,
                "n_points":     res["n"],
                "wmean_mag":    round(res["wmean"], 4) if np.isfinite(res.get("wmean", np.nan)) else np.nan,
                "chi2":         round(res["chi2"], 4) if np.isfinite(res.get("chi2", np.nan)) else np.nan,
                "dof":          res["dof"],
                "reduced_chi2": round(res["reduced"], 4) if np.isfinite(res.get("reduced", np.nan)) else np.nan,
                "p_value":      round(res["p_value"], 6) if np.isfinite(res.get("p_value", np.nan)) else np.nan,
                "verdict":      res["verdict"],
            })

            print(
                f"  {flt:6s} | {range_label:12s} | Orbit {int(orbit)} ({year}) | "
                f"n={res['n']:3d} | χ²/dof={res['reduced']:.2f} | "
                f"p={res['p_value']:.4f} | {res['verdict']}"
            )

df_custom = pd.DataFrame(custom_rows)
df_custom.to_csv(os.path.join(TABLE_DIR, "variability_custom_ranges.csv"), index=False)
print(f"  -> saved variability_custom_ranges.csv ({len(df_custom)} rows)")

# ==========================================================
# SUMMARY REPORT
# ==========================================================

SUMMARY_FILE = os.path.join(TABLE_DIR, "variability_summary.txt")

with open(SUMMARY_FILE, "w") as f:
    def w(line=""):
        f.write(line + "\n")

    w(f"# Variability Analysis Summary — {SOURCE_NAME}")
    w(f"# Generated from: {os.path.basename(DET_CSV)}")
    w(f"# Chi-squared verdict thresholds: VARIABLE = p<0.01 AND χ²/dof>2.0")
    w(f"#                                 POSSIBLY VARIABLE = p<0.05 OR χ²/dof>1.5")
    w(f"#                                 CONSTANT = otherwise")
    w(f"# Minimum points per test: {MIN_POINTS}")
    w()

    # --- Analysis 1 ---
    w("=" * 70)
    w("ANALYSIS 1: WITHIN-ORBIT VARIABILITY")
    w("Is the source varying significantly within each orbital passage?")
    w("=" * 70)
    for flt in FILTERS:
        w(f"\n  Filter: {flt}")
        sub = df_within[df_within["filter"] == flt]
        if sub.empty:
            w("    No data.")
            continue
        for _, row in sub.iterrows():
            if pd.isna(row["chi2"]):
                w(f"    Orbit {int(row['orbit_id'])} ({row['year']}): INSUFFICIENT DATA (n={int(row['n_points'])})")
            else:
                w(
                    f"    Orbit {int(row['orbit_id'])} ({row['year']}): "
                    f"n={int(row['n_points'])} | χ²/dof={row['reduced_chi2']:.2f} | "
                    f"p={row['p_value']:.4f} | {row['verdict']}"
                )

    w()

    # --- Analysis 2 ---
    w("=" * 70)
    w("ANALYSIS 2: ORBIT-TO-ORBIT VARIABILITY")
    w("Is the mean brightness consistent across orbital passages?")
    w("=" * 70)
    for _, row in df_orbit.iterrows():
        if pd.isna(row["chi2"]):
            w(f"  {row['filter']:6s}: INSUFFICIENT DATA")
        else:
            w(
                f"  {row['filter']:6s}: {row['n_orbits']} orbits ({row['years_included']}) | "
                f"χ²/dof={row['reduced_chi2']:.2f} | p={row['p_value']:.4f} | {row['verdict']}"
            )

    w()

    # --- Analysis 3 ---
    w("=" * 70)
    w("ANALYSIS 3: DISC-CROSSING SEGMENT VARIABILITY")
    w("Variability within each physical phase of the orbital passage.")
    w()
    w("Segment boundaries (days from periastron):")
    for seg_name, seg_label, lo, hi in SEGMENTS:
        w(f"  {seg_name:<20} {seg_label}")
    w("=" * 70)

    for seg_name, seg_label, _, _ in SEGMENTS:
        w(f"\n  --- {seg_label} ({seg_name}) ---")
        sub = df_seg[df_seg["segment"] == seg_name]
        for flt in FILTERS:
            fsub = sub[sub["filter"] == flt].sort_values("orbit_id")
            if fsub.empty:
                continue
            w(f"\n    Filter: {flt}")
            for _, row in fsub.iterrows():
                if pd.isna(row["chi2"]):
                    w(
                        f"      Orbit {int(row['orbit_id'])} ({row['year']}): "
                        f"INSUFFICIENT DATA (n={int(row['n_points'])})"
                    )
                else:
                    w(
                        f"      Orbit {int(row['orbit_id'])} ({row['year']}): "
                        f"n={int(row['n_points'])} | χ²/dof={row['reduced_chi2']:.2f} | "
                        f"p={row['p_value']:.4f} | {row['verdict']}"
                    )

    w()

    # --- Analysis 4 ---
    w("=" * 70)
    w("ANALYSIS 4: ACTIVE PHASE (±100 DAYS) ORBIT-TO-ORBIT VARIABILITY")
    w("Is the mean brightness in the active orbital phase consistent across years?")
    w("Restricted to ±100 days from periastron to avoid quiescence dilution.")
    w("=" * 70)
    if df_active.empty:
        w("  No filters had sufficient data across multiple orbits.")
    else:
        for _, row in df_active.iterrows():
            if pd.isna(row["chi2"]):
                w(f"  {row['filter']:6s}: INSUFFICIENT DATA")
            else:
                w(
                    f"  {row['filter']:6s}: {row['n_orbits']} orbits ({row['years_included']}) | "
                    f"χ²/dof={row['reduced_chi2']:.2f} | p={row['p_value']:.4f} | {row['verdict']}"
                )

    w()

    # --- Analysis 4 orbit means ---
    w("=" * 70)
    w("ANALYSIS 4b: ACTIVE PHASE PER-ORBIT WEIGHTED MEANS")
    w(f"Weighted mean magnitude within ±{int(ACTIVE_PHASE_DAYS)} days per orbit/filter.")
    w("=" * 70)
    for flt in FILTERS:
        w(f"\n  Filter: {flt}")
        sub = df_active_means[df_active_means["filter"] == flt]
        if sub.empty:
            w("    No data.")
            continue
        w(f"  {'Orbit':>6} {'Year':>6} {'N':>4} {'Wmean mag':>11} {'Wmean err':>11}")
        w("  " + "-" * 44)
        for _, row in sub.iterrows():
            yr = int(row["year"]) if row["year"] else "N/A"
            w(f"  {int(row['orbit_id']):>6} {str(yr):>6} {int(row['n_points']):>4} "
              f"{row['wmean']:>11.4f} {row['wmean_err']:>11.4f}")

    w()

    # --- Analysis 5 ---
    w("=" * 70)
    w("ANALYSIS 5: ACTIVE PHASE (±100 DAYS) WITHIN-ORBIT VARIABILITY")
    w("Is the source variable within the ±100 day active window of each orbit?")
    w("=" * 70)
    for flt in FILTERS:
        w(f"\n  Filter: {flt}")
        sub = df_active_within[df_active_within["filter"] == flt]
        if sub.empty:
            w("    No data.")
            continue
        for _, row in sub.iterrows():
            if pd.isna(row["chi2"]):
                w(f"    Orbit {int(row['orbit_id'])} ({row['year']}): INSUFFICIENT DATA (n={int(row['n_points'])})")
            else:
                w(
                    f"    Orbit {int(row['orbit_id'])} ({row['year']}): "
                    f"n={int(row['n_points'])} | χ²/dof={row['reduced_chi2']:.2f} | "
                    f"p={row['p_value']:.4f} | {row['verdict']}"
                )
    w()           
    # --- Analysis 6 ---
    w("=" * 70)
    w("ANALYSIS 6: CUSTOM PERIASTRON-RELATIVE RANGE VARIABILITY")
    w("User-defined dt windows for source-specific tests.")
    w()
    w("Custom ranges (days from periastron):")
    for range_name, range_label, lo, hi in CUSTOM_RANGES:
        w(f"  {range_name:<18} {range_label}")
    w("=" * 70)

    for range_name, range_label, _, _ in CUSTOM_RANGES:
        w(f"\n  --- {range_label} ({range_name}) ---")
        sub = df_custom[df_custom["custom_range"] == range_name]
        for flt in FILTERS:
            fsub = sub[sub["filter"] == flt].sort_values("orbit_id")
            if fsub.empty:
                continue
            w(f"\n    Filter: {flt}")
            for _, row in fsub.iterrows():
                if pd.isna(row["chi2"]):
                    w(
                        f"      Orbit {int(row['orbit_id'])} ({row['year']}): "
                        f"INSUFFICIENT DATA (n={int(row['n_points'])})"
                    )
                else:
                    w(
                        f"      Orbit {int(row['orbit_id'])} ({row['year']}): "
                        f"n={int(row['n_points'])} | χ²/dof={row['reduced_chi2']:.2f} | "
                        f"p={row['p_value']:.4f} | {row['verdict']}"
                    )
    w()
    w("=" * 70)
    w("OVERALL VARIABILITY FLAGS (any orbit showing VARIABLE or POSSIBLY VARIABLE)")
    w("=" * 70)
    w()
    w(f"  {'Filter':<8} {'Within-orbit':<20} {'Orbit-to-orbit':<20} {'Most variable segment':<35} {'Active wmean spread'}")
    w("  " + "-" * 95)

    for flt in FILTERS:
        wo = df_within[df_within["filter"] == flt]
        wo_flag = (
            "VARIABLE" if (wo["verdict"] == "VARIABLE").any()
            else "POSSIBLY" if (wo["verdict"] == "POSSIBLY VARIABLE").any()
            else "CONSTANT"
        )

        oo = df_orbit[df_orbit["filter"] == flt]
        oo_flag = oo["verdict"].values[0] if len(oo) else "NO DATA"

        sg = df_seg[df_seg["filter"] == flt].dropna(subset=["reduced_chi2"])
        if not sg.empty:
            best_seg = sg.groupby("segment")["reduced_chi2"].mean().idxmax()
            best_val = sg.groupby("segment")["reduced_chi2"].mean().max()
            best_str = f"{best_seg} (χ²/dof={best_val:.2f})"
        else:
            best_str = "NO DATA"

        am = df_active_means[df_active_means["filter"] == flt]
        if len(am) >= 2 and am["wmean"].notna().sum() >= 2:
            spread = am["wmean"].max() - am["wmean"].min()
            spread_str = f"{spread:.3f} mag ({len(am)} orbits)"
        else:
            spread_str = "NO DATA"

        w(f"  {flt:<8} {wo_flag:<20} {oo_flag:<20} {best_str:<35} {spread_str}")

    w()
    w("# End of summary")

print(f"\n  -> saved variability_summary.txt")
print("\n=== Variability analysis complete ===")
print(f"Outputs in: {TABLE_DIR}")
print("  variability_within_orbit.csv")
print("  variability_orbit_to_orbit.csv")
print("  variability_orbit_means.csv")
print("  variability_disc_crossing.csv")
print("  variability_active_phase.csv")
print("  variability_active_phase_orbit_means.csv")
print("  variability_active_phase_within_orbit.csv")
print("  variability_custom_ranges.csv")
print("  variability_summary.txt")