#!/usr/bin/env python3

import csv
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binomtest


INDIVIDUAL_PERSPECTIVES = ["risk", "relationship", "financial", "competitive"]
EXCLUDED_ENCODER = "gpt-5-mini"


def load_csv_files(results_dir):
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory '{results_dir}' does not exist.")
        sys.exit(1)

    csv_files = list(results_path.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{results_dir}'.")
        sys.exit(1)

    records = []
    for csv_file in csv_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                composite = (
                    float(row.get("judgment_relevance", 0))
                    + float(row.get("judgment_completeness", 0))
                    + float(row.get("judgment_accuracy", 0))
                    + float(row.get("judgment_clarity", 0))
                ) / 4
                records.append(
                    {
                        "model": row.get("model", "unknown"),
                        "scenario": Path(row.get("scenario", "")).stem,
                        "query": row.get("query", ""),
                        "perspective": row.get("perspective", "unknown"),
                        "score": composite,
                    }
                )
    return records


def group_by_query(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["model"], r["scenario"], r["query"])].append(r)
    return dict(grouped)


def average_by_perspective(query_records):
    by_persp = defaultdict(list)
    for r in query_records:
        by_persp[r["perspective"]].append(r["score"])
    return {p: statistics.mean(scores) for p, scores in by_persp.items()}


def compute_grouped_scores(records):
    grouped = group_by_query(records)
    scores = {
        "max_perspective": [],
        "unified": [],
        "unified_interpretive": [],
        "arbiter": [],
    }

    for query_records in grouped.values():
        by_persp = average_by_perspective(query_records)
        individual = [by_persp[p] for p in INDIVIDUAL_PERSPECTIVES if p in by_persp]
        if individual:
            scores["max_perspective"].append(max(individual))
        if "unified" in by_persp:
            scores["unified"].append(by_persp["unified"])
        if "unified-interpretive" in by_persp:
            scores["unified_interpretive"].append(by_persp["unified-interpretive"])
        if "arbiter" in by_persp:
            scores["arbiter"].append(by_persp["arbiter"])
    return scores


def compute_stats(values, name):
    if len(values) < 2:
        return {"name": name, "n": len(values), "mean": 0, "std": 0, "ci_95": 0}
    m = statistics.mean(values)
    sd = statistics.stdev(values)
    ci = 1.96 * sd / np.sqrt(len(values))
    return {"name": name, "n": len(values), "mean": m, "std": sd, "ci_95": ci}


def statistical_test(g1, g2):
    if len(g1) < 2 or len(g2) < 2:
        return None
    t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    pooled_std = np.sqrt(
        ((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1))
        / (len(g1) + len(g2) - 2)
    )
    d = (
        (statistics.mean(g1) - statistics.mean(g2)) / pooled_std
        if pooled_std > 0
        else 0
    )
    return {"t": t_stat, "t_p": t_p, "U": u_stat, "U_p": u_p, "d": d}


def compute_win_rates(records):
    grouped = group_by_query(records)
    comparisons = {
        "max_vs_unified": [0, 0, 0],
        "max_vs_unified_interp": [0, 0, 0],
        "arbiter_vs_unified": [0, 0, 0],
        "arbiter_vs_unified_interp": [0, 0, 0],
        "arbiter_vs_max": [0, 0, 0],
    }

    for query_records in grouped.values():
        by_persp = average_by_perspective(query_records)
        individual = [by_persp[p] for p in INDIVIDUAL_PERSPECTIVES if p in by_persp]
        max_p = max(individual) if individual else None
        unified = by_persp.get("unified")
        unified_interp = by_persp.get("unified-interpretive")
        arbiter = by_persp.get("arbiter")

        def compare(a, b, key):
            if a is not None and b is not None:
                if a > b:
                    comparisons[key][0] += 1
                elif b > a:
                    comparisons[key][1] += 1
                else:
                    comparisons[key][2] += 1

        compare(max_p, unified, "max_vs_unified")
        compare(max_p, unified_interp, "max_vs_unified_interp")
        compare(arbiter, unified, "arbiter_vs_unified")
        compare(arbiter, unified_interp, "arbiter_vs_unified_interp")
        compare(arbiter, max_p, "arbiter_vs_max")

    return comparisons


def print_win_rate(name1, name2, wins, losses, ties):
    total = wins + losses + ties
    if total == 0:
        return
    print(f"\n{name1} vs {name2}:")
    print(f"  {name1} wins: {wins} ({100 * wins / total:.1f}%)")
    print(f"  {name2} wins: {losses} ({100 * losses / total:.1f}%)")
    print(f"  Ties: {ties} ({100 * ties / total:.1f}%)")
    if wins + losses > 0:
        p = binomtest(wins, wins + losses, p=0.5).pvalue
        sig = "*" if p < 0.05 else "ns"
        print(f"  Binomial test (excl. ties): p={p:.4f} {sig}")


def print_condition_stats(scores, header):
    print("=" * 70)
    print(header)
    print("=" * 70)
    print(f"{'Condition':<22} {'M':>6} {'SD':>6} {'95% CI':>14}")
    print("-" * 70)
    for key, label in [
        ("arbiter", "Arbiter"),
        ("max_perspective", "Max Perspective"),
        ("unified", "Unified"),
        ("unified_interpretive", "Unified-Interpretive"),
    ]:
        s = compute_stats(scores[key], label)
        lo, hi = s["mean"] - s["ci_95"], s["mean"] + s["ci_95"]
        print(
            f"{label:<22} {s['mean']:>6.2f} {s['std']:>6.2f} [{lo:>5.2f}, {hi:>5.2f}]"
        )


def print_statistical_tests(scores):
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)
    test = statistical_test(scores["arbiter"], scores["unified_interpretive"])
    if test is None:
        return
    n1, n2 = len(scores["arbiter"]), len(scores["unified_interpretive"])
    print(f"\nArbiter vs Unified-Interpretive:")
    print(f"  t({n1 + n2 - 2}) = {test['t']:.2f}, p = {test['t_p']:.3f}")
    print(f"  Mann-Whitney U = {test['U']:.1f}, p = {test['U_p']:.3f}")
    print(f"  Cohen's d = {test['d']:.2f}")
    for g1, g2, label in [
        ("arbiter", "unified", "Arbiter vs Unified"),
        ("max_perspective", "unified", "Max Perspective vs Unified"),
    ]:
        t = statistical_test(scores[g1], scores[g2])
        if t is not None:
            print(f"\n{label}: d = {t['d']:.2f}")


def print_win_rates_block(records):
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD WIN RATES")
    print("=" * 70)
    win_rates = compute_win_rates(records)
    for key, (n1, n2) in [
        ("max_vs_unified", ("Max Perspective", "Unified")),
        ("max_vs_unified_interp", ("Max Perspective", "Unified-Interp")),
        ("arbiter_vs_unified", ("Arbiter", "Unified")),
        ("arbiter_vs_unified_interp", ("Arbiter", "Unified-Interp")),
        ("arbiter_vs_max", ("Arbiter", "Max Perspective")),
    ]:
        w, l, t = win_rates[key]
        print_win_rate(n1, n2, w, l, t)


def print_per_model_table(records, header):
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    print(f"{'Model':<18} {'Arbiter':>8} {'Unified-Interp':>15} {'Diff':>8}")
    print("-" * 70)
    for model in sorted(by_model.keys()):
        model_scores = compute_grouped_scores(by_model[model])
        arb = statistics.mean(model_scores["arbiter"]) if model_scores["arbiter"] else 0
        ui = (
            statistics.mean(model_scores["unified_interpretive"])
            if model_scores["unified_interpretive"]
            else 0
        )
        print(f"{model:<18} {arb:>8.2f} {ui:>15.2f} {arb - ui:>+8.2f}")


def print_summary_block(records, table2_header, per_model_header):
    scores = compute_grouped_scores(records)
    print_condition_stats(scores, table2_header)
    print_statistical_tests(scores)
    print_win_rates_block(records)
    print_per_model_table(records, per_model_header)


def build_mixed_effects_rows(records):
    rows = []
    grouped = group_by_query(records)
    for (model, scenario, query), query_records in grouped.items():
        by_persp = average_by_perspective(query_records)
        individual = [by_persp[p] for p in INDIVIDUAL_PERSPECTIVES if p in by_persp]
        conds = {}
        if individual:
            conds["max_perspective"] = max(individual)
        if "unified" in by_persp:
            conds["unified"] = by_persp["unified"]
        if "unified-interpretive" in by_persp:
            conds["unified_interpretive"] = by_persp["unified-interpretive"]
        if "arbiter" in by_persp:
            conds["arbiter"] = by_persp["arbiter"]
        for cond, score in conds.items():
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "query": f"{scenario}::{query}",
                    "condition": cond,
                    "score": score,
                }
            )
    return rows


def _fit_mixedlm(df, formula_label, crossed):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if crossed:
            df = df.assign(_all=1)
            vc = {"query_re": "0 + C(query)", "model_re": "0 + C(model)"}
            md = smf.mixedlm(
                "score ~ C(condition)",
                df,
                groups="_all",
                vc_formula=vc,
                re_formula="0",
            )
        else:
            md = smf.mixedlm(
                "score ~ C(condition)", df, groups="query", re_formula="~1"
            )
        return md.fit(reml=True), formula_label


def _print_mixed_fit(fit, formula_label, reference):
    print(f"\n{formula_label}")
    print(f"[reference level for condition: {reference}]")
    print("-" * 70)
    print(fit.summary())
    print("\nKey contrasts (arbiter effect is primary):")
    for term in fit.params.index:
        if not term.startswith("C(condition)[T."):
            continue
        name = term[len("C(condition)[T.") : -1]
        est = fit.params[term]
        se = fit.bse[term]
        lo, hi = fit.conf_int().loc[term, 0], fit.conf_int().loc[term, 1]
        p = fit.pvalues[term]
        p_str = f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"
        print(
            f"  {name} vs {reference}: "
            f"beta = {est:+.3f}, SE = {se:.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], p = {p_str}"
        )


def run_mixed_effects(records, header):
    import pandas as pd

    rows = build_mixed_effects_rows(records)
    if not rows:
        return
    df = pd.DataFrame(rows)
    reference = "unified_interpretive"
    conds_present = list(df["condition"].unique())
    if reference not in conds_present:
        reference = conds_present[0]
    others = [c for c in ["arbiter", "max_perspective", "unified"] if c in conds_present]
    df["condition"] = pd.Categorical(df["condition"], categories=[reference] + others)

    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)
    print(
        f"N = {len(df)} obs | {df['query'].nunique()} queries | "
        f"{df['model'].nunique()} models | {df['condition'].nunique()} conditions"
    )

    fit, label = _fit_mixedlm(
        df, "Model A: score ~ condition + (1|query)  [matches reviewer snippet]", crossed=False
    )
    _print_mixed_fit(fit, label, reference)

    fit, label = _fit_mixedlm(
        df,
        "Model B: score ~ condition + (1|query) + (1|model)  [crossed random effects]",
        crossed=True,
    )
    _print_mixed_fit(fit, label, reference)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Statistics for Goal-Conditioned Encoding paper"
    )
    parser.add_argument("results_dir", nargs="?", default="./results/")
    parser.add_argument("--skip-scenarios", default=None)
    parser.add_argument("--skip-models", default=None)
    args = parser.parse_args()

    records = load_csv_files(args.results_dir)
    print(f"Loaded {len(records)} records from {args.results_dir}\n")

    print_summary_block(
        records,
        "TABLE 2: CONDITION STATISTICS (n=50 per condition)",
        "TABLE 3: PER-MODEL RESULTS",
    )

    print("\n" + "=" * 70)
    print("PER-SCENARIO KEY STATISTICS")
    print("=" * 70)
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r["scenario"]].append(r)

    for scenario in sorted(by_scenario.keys()):
        sc_scores = compute_grouped_scores(by_scenario[scenario])
        print(f"\n{scenario.capitalize()}:")
        for key, label in [
            ("max_perspective", "Max Perspective"),
            ("arbiter", "Arbiter"),
            ("unified_interpretive", "Unified-Interpretive"),
        ]:
            if sc_scores[key]:
                print(f"  {label}: M = {statistics.mean(sc_scores[key]):.2f}")

    print("\n" + "=" * 70)
    print("INDIVIDUAL PERSPECTIVE MEANS")
    print("=" * 70)
    by_persp = defaultdict(list)
    for r in records:
        by_persp[r["perspective"]].append(r["score"])
    pooled = []
    for p in INDIVIDUAL_PERSPECTIVES:
        if p in by_persp:
            print(f"{p}: M = {statistics.mean(by_persp[p]):.2f} (n={len(by_persp[p])})")
            pooled.extend(by_persp[p])
    if pooled:
        print(
            f"\nPooled across all individual perspectives: M = {statistics.mean(pooled):.2f} (n={len(pooled)})"
        )
        persp_means = [
            statistics.mean(by_persp[p])
            for p in INDIVIDUAL_PERSPECTIVES
            if p in by_persp
        ]
        print(
            f"Unweighted mean of perspective means: M = {statistics.mean(persp_means):.2f}"
        )

    filtered = [r for r in records if r["model"] != EXCLUDED_ENCODER]
    if filtered and len(filtered) < len(records):
        n_removed = len(records) - len(filtered)
        print("\n\n" + "#" * 70)
        print(
            f"# ROBUSTNESS CHECK: {EXCLUDED_ENCODER} EXCLUDED AS ENCODER  "
            f"(dropped {n_removed} rows; judge unchanged)"
        )
        print("#" * 70)
        print_summary_block(
            filtered,
            f"TABLE 2 (robustness): CONDITIONS WITH {EXCLUDED_ENCODER} EXCLUDED",
            f"TABLE 3 (robustness): PER-MODEL RESULTS WITH {EXCLUDED_ENCODER} EXCLUDED",
        )

    print("\n\n" + "#" * 70)
    print("# MIXED-EFFECTS MODEL")
    print("#" * 70)
    try:
        run_mixed_effects(records, "MIXED-EFFECTS: FULL DATA")
        if filtered and len(filtered) < len(records):
            run_mixed_effects(
                filtered,
                f"MIXED-EFFECTS: {EXCLUDED_ENCODER} EXCLUDED AS ENCODER",
            )
    except ImportError:
        print("(skipped: statsmodels not installed; run `uv sync`)")


if __name__ == "__main__":
    main()
