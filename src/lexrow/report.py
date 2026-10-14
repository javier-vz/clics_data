"""Descriptive report generated only from actually computed outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def create_report(tables: Path, target: Path) -> None:
    target.mkdir(parents=True,exist_ok=True)
    metrics = pd.read_csv(tables/"language_metrics.csv")
    config = json.loads((tables/"analysis_config.json").read_text(encoding="utf-8"))
    min_c = int(config["min_concepts"])
    x = metrics[metrics["M_concepts"]>=min_c].copy()
    if x.empty:
        raise RuntimeError("No languages passed the minimum-coverage threshold")
    def median(col):
        return float(x[col].median())
    def fmt(v):
        return f"{v:.4g}" if np.isfinite(v) else "N/A"
    ratio = x["h_ratio"].replace([np.inf,-np.inf],np.nan).dropna()
    frac_below = float((x["h_A"] < x["h_null_mean"]).mean())
    tail_count = int((x["p_h_two_sided"]<0.05).sum())
    fdr_count = int((x["q_h_fdr"]<0.05).sum())
    text = f"""# CLICS row-occupancy: automated descriptive report

Generated from `results/tables/`, not from predetermined conclusions.

## Data coverage

- Languages with analyzed associations: **{len(metrics)}**.
- Languages with at least {min_c} concepts: **{len(x)}**.
- Median concepts per filtered language: **{fmt(median('M_concepts'))}**.
- Median unique form--concept associations A: **{fmt(median('A_associations'))}**.
- Median distinct active forms V: **{fmt(median('V_forms'))}**.

## Conditional occupancy null (fixed A and V)

- Median observed concentration h_A: **{fmt(median('h_A'))}**.
- Median expected conditional-null concentration: **{fmt(median('h_null_mean'))}**.
- Median per-language observed/conditional-null concentration ratio: **{fmt(float(ratio.median()) if len(ratio) else float('nan'))}**.
- Languages with observed concentration below their conditional-null mean: **{frac_below:.1%}**.
- Nominal two-sided Monte Carlo p<0.05: **{tail_count}/{len(x)}**.
- BH-FDR q<0.05: **{fdr_count}/{len(x)}**.

Each randomization assigns **A labeled associations to V labeled forms, conditioned on every form being occupied**. Therefore, observed and null values of V/A are *identical by construction* and there is no vocabulary-dispersion test against this null. The comparison concerns **how unevenly associations are distributed over those V occupied forms**.

Monte Carlo p-values use independent conditional-surjection samples with the plus-one correction, so their minimum attainable value is **1/({config['occupancy_null_samples']}+1)**. Significance claims require adequate sample resolution and consideration of dependence among related languages.

## Degree-preserving colexification null

- Median observed largest-component fraction: **{fmt(median('largest_component_frac'))}**.
- Median degree-preserving-null mean largest-component fraction: **{fmt(median('largest_component_frac_null_mean'))}**.
- Median observed isolated-concept fraction: **{fmt(median('isolated_concepts_frac'))}**.
- Median null mean isolated-concept fraction: **{fmt(median('isolated_concepts_frac_null_mean'))}**.

The bipartite edge-swap chain preserves all individual concept degrees and form occupancies *exactly* while avoiding duplicate binary edges. It is an MCMC null, so its finite-run convergence requires sensitivity analyses to burn-in, thinning, and independent chains.

## Interpretation cautions

1. Sampling coverage differs among languages and sources. A ≥500 concept filter improves comparability but cannot eliminate documentation bias.
2. Global form normalization is conservative: NFC + casefold + whitespace cleanup, with diacritics preserved. Homography/segmentation can still influence colexification.
3. The conditional-null comparison controls A and V, but deliberately does not conserve concept degrees. The graph null tests a different structural property by fixing both degree sequences.
4. Languages are phylogenetically related; language-wise p-values or global median bootstrap are not independent phylogenetic replicates. Family-level sensitivity analyses are advisable.
5. Results should be presented as network occupancy patterns, not direct inference of lexical change or actual language-game dynamics.

## Reproduction

- Seed: `{config['seed']}`.
- Conditional-null independent draws per language: `{config['occupancy_null_samples']}`.
- Degree-preserving samples per language: `{config['component_null_samples']}`.
- Edge-swap burn-in: `{config['swaps_burn']}` attempted swaps per association.
- Edge-swap thinning: `{config['swaps_thin']}` attempted swaps per association.
- Fingerprint: `{config['input_fingerprint']}`.

Detailed input provenance is in `results/provenance.json`; the CSV files retain observed and null replicate-level quantities for independent figure regeneration.
"""
    (target/"summary.md").write_text(text,encoding="utf-8")
    print("Report saved:",target/"summary.md")
