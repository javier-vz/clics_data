"""Language-wise observed statistics, conditional-null replicates, exact-degree graph null."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .components import bipartite_edges, bipartite_swap_chain, component_stats
from .occupancy import (
    concentration, minimal_concentration, monte_carlo_tail,
    occupancy_draw_metrics, sample_surjection_counts,
)


def _seed(seed: int, language: str, stage: str) -> int:
    b = f"{seed}:{language}:{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(b).digest()[:4], "little")


def _bh_fdr(p: pd.Series) -> np.ndarray:
    vals = p.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    valid = np.isfinite(vals)
    z = vals[valid]
    n = len(z)
    if n == 0:
        return out
    order = np.argsort(z)
    sorted_q = z[order] * n / np.arange(1, n+1)
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    reord = np.empty(n)
    reord[order] = np.clip(sorted_q, 0, 1)
    out[valid] = reord
    return out


def analyze_language(g: pd.DataFrame, samples: int, component_samples: int,
                     seed: int, swaps_burn: int, swaps_thin: int) -> tuple[dict,list[dict],list[dict]]:
    language = str(g["Language_ID"].iloc[0])
    forms_counts = g.groupby("form_norm", sort=False)["Parameter_ID"].nunique().to_numpy(dtype=int)
    a = int(len(g))
    v = int(len(forms_counts))
    m = int(g["Parameter_ID"].nunique())
    h_obs = concentration(forms_counts)
    rho_max = int(forms_counts.max())
    forms_ge2 = int((forms_counts >= 2).sum())
    forms_ge3 = int((forms_counts >= 3).sum())
    null_counts = sample_surjection_counts(a, v, samples, _seed(seed, language, "occupancy"))
    metrics = occupancy_draw_metrics(null_counts)
    h_null = metrics["h_null"]
    h_null_sd = float(np.std(h_null, ddof=1)) if samples > 1 else 0.0

    edges, concepts, forms = bipartite_edges(g)
    observed_comp = component_stats(edges, concepts, forms)
    comp_null = bipartite_swap_chain(
        edges, concepts, forms, component_samples,
        seed=_seed(seed, language, "component"),
        burnin_multiplier=swaps_burn, thinning_multiplier=swaps_thin,
    )
    component_draw_rows = [dict(Language_ID=language, draw=j, **d) for j,d in enumerate(comp_null)]
    comp_df = pd.DataFrame(comp_null)

    obs = {
        "Language_ID":language,
        "A_associations":a,
        "M_concepts":m,
        "V_forms":v,
        "sV_star": v/a,  # equal to every conditional-null value by construction
        "h_A":h_obs,
        "h_min_fixed_A_V":minimal_concentration(a, v),
        "h_null_mean":float(np.mean(h_null)),
        "h_null_sd":h_null_sd,
        "h_null_median":float(np.median(h_null)),
        "h_ratio": h_obs / float(np.mean(h_null)) if np.mean(h_null) else np.nan,
        "h_z": (h_obs-float(np.mean(h_null))) / h_null_sd if h_null_sd else np.nan,
        "p_h_lower":monte_carlo_tail(h_obs, h_null, "lower"),
        "p_h_upper":monte_carlo_tail(h_obs, h_null, "upper"),
        "rho_max":rho_max,
        "rho_max_null_mean":float(np.mean(metrics["max_rho_null"])),
        "forms_ge2":forms_ge2,
        "forms_ge2_null_mean":float(np.mean(metrics["forms_ge2_null"])),
        "forms_ge3":forms_ge3,
        "forms_ge3_null_mean":float(np.mean(metrics["forms_ge3_null"])),
        "largest_component_frac":observed_comp["largest_component_frac"],
        "largest_component_size":observed_comp["largest_component_size"],
        "largest_component_frac_null_mean":float(comp_df["largest_component_frac"].mean()),
        "largest_component_frac_null_sd":float(comp_df["largest_component_frac"].std(ddof=1)) if component_samples>1 else 0.0,
        "isolated_concepts_frac":observed_comp["isolated_concepts_frac"],
        "isolated_concepts_frac_null_mean":float(comp_df["isolated_concepts_frac"].mean()),
        "nontrivial_components":observed_comp["nontrivial_components"],
        "nontrivial_components_null_mean":float(comp_df["nontrivial_components"].mean()),
    }
    obs["p_h_two_sided"] = min(1.0, 2*min(obs["p_h_lower"],obs["p_h_upper"]))
    occ_draw_rows = [
        {"Language_ID":language, "draw":i,
         "h_null":float(metrics["h_null"][i]),
         "max_rho_null":int(metrics["max_rho_null"][i]),
         "forms_ge2_null":int(metrics["forms_ge2_null"][i]),
         "forms_ge3_null":int(metrics["forms_ge3_null"][i])}
        for i in range(samples)
    ]
    return obs, occ_draw_rows, component_draw_rows


def run_analysis(pairs: pd.DataFrame, metadata: pd.DataFrame, out: Path,
                 *, samples: int = 199, component_samples: int = 49,
                 seed: int = 2026, swaps_burn: int = 10,
                 swaps_thin: int = 3, min_concepts: int = 500,
                 input_fingerprint: str = "", resume: bool = True) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out.parent / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    run_key = hashlib.sha256(json.dumps({
        "fingerprint":input_fingerprint, "samples": samples,
        "component_samples":component_samples, "seed":seed,
        "burn":swaps_burn, "thin":swaps_thin,
        "version":"0.1.0",
    }, sort_keys=True).encode()).hexdigest()[:16]
    results, occupancy_draws, component_draws = [], [], []
    grouped = pairs.groupby("Language_ID", sort=True)
    n_total = grouped.ngroups
    for k, (lang, g) in enumerate(grouped, 1):
        file_key = hashlib.sha1(str(lang).encode()).hexdigest()[:16]
        cp = checkpoint / f"{file_key}_{run_key}.json"
        if resume and cp.exists():
            payload = json.loads(cp.read_text(encoding="utf-8"))
        else:
            obs, occ, comp = analyze_language(g, samples, component_samples, seed, swaps_burn, swaps_thin)
            payload = {"row":obs,"occupancy_draws":occ,"component_draws":comp}
            cp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=True), encoding="utf-8")
        results.append(payload["row"])
        occupancy_draws.extend(payload["occupancy_draws"])
        component_draws.extend(payload["component_draws"])
        if k==1 or k % 10 == 0 or k==n_total:
            print(f"[{k}/{n_total}] {lang} — {payload['row']['M_concepts']} concepts", flush=True)

    df = pd.DataFrame(results).merge(metadata, on="Language_ID", how="left", validate="one_to_one")
    df["q_h_fdr"] = _bh_fdr(df["p_h_two_sided"])
    df = df.sort_values("Language_ID")
    df.to_csv(out / "language_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(occupancy_draws).to_csv(out/"occupancy_null_draws.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(component_draws).to_csv(out/"component_null_draws.csv", index=False, encoding="utf-8-sig")
    rows = []
    thresholds = sorted(set([0, 250, 500, 750, 1000, 1250, 1500, min_concepts]))
    for threshold in thresholds:
        x = df[df["M_concepts"] >= threshold]
        if len(x)==0:
            continue
        rows.append({
            "min_concepts":threshold,
            "n_languages":len(x),
            "median_h_ratio":float(x["h_ratio"].median()),
            "median_h_empirical":float(x["h_A"].median()),
            "median_h_null":float(x["h_null_mean"].median()),
            "fraction_h_below_conditional_null":float((x["h_A"] < x["h_null_mean"]).mean()),
            "fraction_h_two_sided_p_lt_005":float((x["p_h_two_sided"]<0.05).mean()),
        })
    pd.DataFrame(rows).to_csv(out / "robustness_thresholds.csv", index=False, encoding="utf-8-sig")
    if "Family" in df:
        group = df[df["M_concepts"]>=min_concepts].groupby(["Macroarea","Family"], dropna=False)
        fam = group.agg(n_languages=("Language_ID","size"),median_h_ratio=("h_ratio","median"),
                        median_h=("h_A","median"),median_null=("h_null_mean","median")).reset_index()
        fam.to_csv(out/"family_summary.csv", index=False, encoding="utf-8-sig")
    (out/"analysis_config.json").write_text(json.dumps({
        "n_languages":len(df), "n_filtered":int((df["M_concepts"] >= min_concepts).sum()),
        "min_concepts":min_concepts, "occupancy_null_samples":samples,
        "component_null_samples":component_samples, "seed":seed,
        "swaps_burn":swaps_burn, "swaps_thin":swaps_thin,
        "input_fingerprint":input_fingerprint,
        "null_occupancy":"iid exact uniform surjections conditional on A and V",
        "null_components":"degree-preserving bipartite edge-switch Markov chain",
    }, indent=2),encoding="utf-8")
    return df
