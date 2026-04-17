#!/usr/bin/env python3
"""Run statistical analyses on flattened user-study CSVs.

Outputs (saved to the same folder as the input CSVs):
 - statistical_results.json  (detailed results)
 - statistical_results.csv   (tabular summary)
 - statistical_summary.md    (human readable markdown)

This script is defensive: it warns (doesn't crash) when data are missing
or a test cannot be computed.
"""
import json
import math
from pathlib import Path
from collections import defaultdict
import warnings

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except Exception as e:
    raise SystemExit(
        "Missing dependency when importing libraries: {}.\nPlease install required packages: pandas, numpy, scipy".format(e)
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "marked sessions"
OUT_JSON = DATA_DIR / "statistical_results.json"
OUT_CSV = DATA_DIR / "statistical_results.csv"
OUT_MD = DATA_DIR / "statistical_summary.md"


def safe_load_csv(name: str):
    path = DATA_DIR / name
    if not path.exists():
        warnings.warn(f"Missing expected file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Error reading {path}: {e}")
        return None


def pivot_condition_matrix(df: pd.DataFrame, metric: str, cond_order=None):
    # Create participant x condition matrix; aggregate with mean if multiple
    if metric not in df.columns:
        raise KeyError(metric)
    mat = df.pivot_table(index="participantId", columns="condition", values=metric, aggfunc="mean")
    # Reorder columns if requested and available
    if cond_order:
        available = [c for c in cond_order if c in mat.columns]
        mat = mat[available]
    return mat


def run_friedman(matrix: pd.DataFrame):
    # matrix: participants x conditions
    k = matrix.shape[1]
    n = matrix.shape[0]
    if k < 2 or n < 2:
        return {"ok": False, "reason": f"Need >=2 conditions and >=2 participants (have {k} conditions, {n} participants)"}
    try:
        arrays = [matrix.iloc[:, i].values for i in range(k)]
        # scipy expects each argument as a sample
        res = stats.friedmanchisquare(*arrays)
        chi2 = float(res.statistic)
        p = float(res.pvalue)
        # Kendall's W from chi2
        W = chi2 / (n * (k - 1)) if (n * (k - 1)) else float('nan')
        return {"ok": True, "chi2": chi2, "p": p, "kendalls_w": W, "n": int(n), "k": int(k)}
    except Exception as e:
        return {"ok": False, "reason": f"friedman error: {e}"}


def wilcoxon_posthoc(matrix: pd.DataFrame):
    # pairwise Wilcoxon signed-rank tests with Bonferroni correction
    cols = list(matrix.columns)
    n = matrix.shape[0]
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = matrix[cols[i]].values
            b = matrix[cols[j]].values
            # Drop NaNs pairwise
            mask = ~np.isnan(a) & ~np.isnan(b)
            a2 = a[mask]
            b2 = b[mask]
            entry = {"pair": (cols[i], cols[j]), "n_pairs": int(mask.sum())}
            if len(a2) < 2:
                entry.update({"ok": False, "reason": "not enough paired observations"})
            else:
                try:
                    stat, p = stats.wilcoxon(a2, b2)
                    # paired effect size d (Cohen's d for paired samples)
                    diff = a2 - b2
                    d = float(np.nanmean(diff) / (np.nanstd(diff, ddof=1) if np.nanstd(diff, ddof=1) != 0 else float('nan')))
                    entry.update({"ok": True, "stat": float(stat), "p": float(p), "d_paired": d})
                except Exception as e:
                    entry.update({"ok": False, "reason": f"wilcoxon error: {e}"})
            pairs.append(entry)
    # Bonferroni
    m = len(pairs)
    for e in pairs:
        if e.get("ok"):
            e["p_bonf"] = min(1.0, e["p"] * m)
    return pairs


def descriptive_stats(series: pd.Series):
    # Coerce to numeric where possible; non-numeric values become NaN and are
    # treated as missing for numeric summaries. Keep original missing count.
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        numeric = series
    n = int(numeric.count())
    return {
        "n": n,
        "mean": float(numeric.mean()) if n else None,
        "median": float(numeric.median()) if n else None,
        "std": float(numeric.std(ddof=1)) if n > 1 else None,
        "min": (float(numeric.min()) if n else None) if pd.api.types.is_numeric_dtype(numeric) else None,
        "max": (float(numeric.max()) if n else None) if pd.api.types.is_numeric_dtype(numeric) else None,
        "missing": int(series.isna().sum()) + int(pd.isna(numeric).sum() - series.isna().sum()),
    }


def analyze_paper_level(df_paper: pd.DataFrame, metrics=None):
    if metrics is None:
        metrics = ["paper_total_score", "free_text_score", "mc_first_attempt_accuracy", "paper_duration_seconds"]
    results = {}
    cond_order = ["no_highlights", "all_highlights", "contextual_highlights"]
    for metric in metrics:
        if metric not in df_paper.columns:
            warnings.warn(f"Metric {metric} not found in paper_level.csv; skipping")
            continue
        mat = pivot_condition_matrix(df_paper, metric, cond_order=cond_order)
        mat_clean = mat.dropna(axis=0, how='any')
        desc = {c: descriptive_stats(mat[c]) for c in mat.columns}
        fried = run_friedman(mat_clean) if not mat_clean.empty else {"ok": False, "reason": "no complete rows after dropping missing"}
        post = wilcoxon_posthoc(mat_clean) if fried.get("ok") else []
        # order-effects: pivot by orderPosition
        if "orderPosition" in df_paper.columns:
            op_mat = df_paper.pivot_table(index="participantId", columns="orderPosition", values=metric, aggfunc="mean")
            op_clean = op_mat.dropna(axis=0, how='any')
            order_test = run_friedman(op_clean) if not op_clean.empty else {"ok": False, "reason": "orderPosition test: insufficient data"}
        else:
            order_test = {"ok": False, "reason": "orderPosition column not present"}

        results[metric] = {"descriptive_by_condition": desc, "friedman": fried, "posthoc_wilcoxon": post, "order_effects": order_test}
    return results


def analyze_cross_level(df_cross: pd.DataFrame):
    res = {}
    if "cross_total_score" in df_cross.columns:
        res["by_contextual_paper"] = {}
        for grp, gdf in df_cross.groupby("contextual_paper"):
            res["by_contextual_paper"][str(grp)] = descriptive_stats(gdf["cross_total_score"])
    else:
        warnings.warn("cross_total_score not found in cross_level.csv")
    return res


def analyze_final_survey(df_final: pd.DataFrame):
    res = {}
    # Likert/frequency items start with fq_ in this dataset
    likers = [c for c in df_final.columns if c.startswith("fq_")]
    for c in likers:
        res[c] = descriptive_stats(df_final[c])
    # demographics: age, role, field, reading_freq, reading_style
    demos = {}
    for d in ["age", "role", "field", "reading_freq", "reading_style"]:
        if d in df_final.columns:
            demos[d] = df_final[d].value_counts(dropna=False).to_dict()
    res["demographics"] = demos
    return res


def summarize_to_markdown(results: dict) -> str:
    lines = ["# Statistical analysis summary", ""]
    # Paper-level metrics
    pl = results.get("paper_level", {})
    lines.append("## Paper-level analyses")
    for metric, r in pl.items():
        lines.append(f"### Metric: {metric}")
        fried = r.get("friedman", {})
        if fried.get("ok"):
            lines.append(f"Friedman chi2 = {fried['chi2']:.4f}, p = {fried['p']:.4g}, Kendall's W = {fried['kendalls_w']:.4f} (n={fried['n']}, k={fried['k']})")
        else:
            lines.append(f"Friedman test not run: {fried.get('reason')}")
        lines.append("Descriptives by condition:")
        for cond, d in (r.get('descriptive_by_condition') or {}).items():
            lines.append(f"- {cond}: n={d['n']} mean={d['mean']} sd={d['std']}")
        lines.append("Post-hoc (Wilcoxon, Bonferroni-corrected):")
        for p in r.get("posthoc_wilcoxon", []):
            if p.get("ok"):
                lines.append(f"- {p['pair'][0]} vs {p['pair'][1]}: stat={p['stat']:.4f}, p={p['p']:.4g}, p_bonf={p.get('p_bonf'):.4g}, d_paired={p.get('d_paired')}")
            else:
                lines.append(f"- {p['pair'][0]} vs {p['pair'][1]}: skipped ({p.get('reason')})")
        lines.append("")

    # Cross-level
    lines.append("## Cross-level analyses")
    for k, v in (results.get("cross_level") or {}).items():
        lines.append(f"### {k}")
        if isinstance(v, dict):
            for grp, d in v.items():
                lines.append(f"- {grp}: n={d['n']} mean={d['mean']} sd={d['std']}")
    lines.append("")

    # Final survey
    lines.append("## Final survey descriptives")
    fs = results.get("final_survey") or {}
    for k, v in fs.items():
        if k == "demographics":
            lines.append("### Demographics")
            for d, counts in v.items():
                lines.append(f"- {d}: {counts}")
        else:
            lines.append(f"- {k}: n={v['n']} mean={v['mean']} sd={v['std']}")

    return "\n".join(lines)


def main():
    df_paper = safe_load_csv("paper_level.csv")
    df_question = safe_load_csv("question_level.csv")
    df_cross = safe_load_csv("cross_level.csv")
    df_final = safe_load_csv("final_survey.csv")

    results = {}
    if df_paper is not None:
        results["paper_level"] = analyze_paper_level(df_paper)
    else:
        results["paper_level"] = {}

    if df_cross is not None:
        results["cross_level"] = analyze_cross_level(df_cross)
    else:
        results["cross_level"] = {}

    if df_final is not None:
        results["final_survey"] = analyze_final_survey(df_final)
    else:
        results["final_survey"] = {}

    # Write outputs
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Flatten JSON to CSV summary
    rows = []
    for metric, r in results.get("paper_level", {}).items():
        fried = r.get("friedman", {})
        row = {"metric": metric}
        row.update({"friedman_ok": fried.get("ok", False)})
        if fried.get("ok"):
            row.update({"chi2": fried.get("chi2"), "p": fried.get("p"), "kendalls_w": fried.get("kendalls_w")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    md = summarize_to_markdown(results)
    OUT_MD.write_text(md)

    print(f"Wrote: {OUT_JSON}, {OUT_CSV}, {OUT_MD}")


if __name__ == "__main__":
    main()
