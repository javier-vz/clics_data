"""Publication-style graphics (PDF and PNG) from CSV results."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AMERICAN_MACROAREAS = {"North America", "South America"}
FORCE_AMERICAN_FAMILIES = {"Quechuan", "Quechua", "Aymaran"}
EXCLUDED_IMPORTED_FAMILIES = {
    "Indo-European", "Afro-Asiatic", "Austronesian", "Sino-Tibetan", "Japonic",
    "Koreanic", "Turkic", "Uralic", "Austroasiatic", "Dravidian", "Niger-Congo",
    "Nilo-Saharan",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family":"DejaVu Sans", "font.size":8.5, "axes.labelsize":9,
        "xtick.labelsize":8, "ytick.labelsize":8, "legend.fontsize":7.8,
        "axes.linewidth":0.8, "pdf.fonttype":42, "ps.fonttype":42,
        "savefig.dpi":350, "figure.dpi":130,
    })


def _clean(ax, label: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.text(-0.10,1.04,label,transform=ax.transAxes,fontweight="bold",fontsize=9,va="top")


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out/f"{name}.png", dpi=350,bbox_inches="tight")
    plt.close(fig)


def plot_all(tables: Path, dest: Path, min_concepts: int = 500) -> None:
    _style()
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(tables/"language_metrics.csv")
    x = df[df["M_concepts"] >= min_concepts].copy()
    if x.empty:
        print("No languages above coverage threshold; skipping figures.")
        return

    # Fig 1: this supersedes the old comparison of V/A with an unconditioned null.
    fig, axes = plt.subplots(1,3,figsize=(10.5,3.1))
    a,b,c = axes
    a.scatter(x["h_null_mean"], x["h_A"], s=14,c="#326A8A",alpha=.58,linewidths=0)
    upper = float(max(x["h_null_mean"].max(),x["h_A"].max()))*1.06
    a.plot([0,upper],[0,upper],linestyle="--",color=".35",lw=.85)
    a.set(xlabel="Conditional-null concentration",ylabel="Observed concentration",xlim=(0,upper),ylim=(0,upper))
    _clean(a,"a")
    b.hist(x["h_ratio"].replace([np.inf,-np.inf],np.nan).dropna(),bins=24,color="#326A8A",edgecolor="white",lw=.35)
    b.axvline(1,ls="--",color=".35",lw=.9)
    b.axvline(x["h_ratio"].median(),color="#BA5B3B",lw=1)
    b.set(xlabel="Observed / conditional-null concentration",ylabel="Number of languages")
    _clean(b,"b")
    c.scatter(x["M_concepts"],x["h_ratio"],s=14,c="#326A8A",alpha=.58,linewidths=0)
    c.axhline(1,ls="--",color=".35",lw=.9)
    c.set(xlabel="Documented concepts",ylabel="Concentration ratio")
    _clean(c,"c")
    fig.subplots_adjust(left=.065,right=.99,bottom=.20,top=.93,wspace=.32)
    _save(fig,dest,"fig1_conditional_occupancy")

    fig, axes = plt.subplots(1,2,figsize=(7.0,3.05))
    for ax,obs,null,letter,lab in [
        (axes[0],"largest_component_frac","largest_component_frac_null_mean","a","Largest component fraction"),
        (axes[1],"isolated_concepts_frac","isolated_concepts_frac_null_mean","b","Isolated-concept fraction"),
    ]:
        ax.scatter(x[null],x[obs],s=16,c="#326A8A",alpha=.6,linewidths=0)
        lo = min(float(x[obs].min()),float(x[null].min()))
        hi = max(float(x[obs].max()),float(x[null].max()))
        margin = (hi-lo)*.05 if hi>lo else .02
        lo,hi = max(0,lo-margin),min(1,hi+margin)
        ax.plot([lo,hi],[lo,hi],ls="--",c=".35",lw=.9)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Degree-preserving null",ylabel="Observed " + lab.lower())
        _clean(ax,letter)
    fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.93,wspace=.36)
    _save(fig,dest,"fig2_colexification_components")

    rb = pd.read_csv(tables/"robustness_thresholds.csv").sort_values("min_concepts")
    fig,ax=plt.subplots(figsize=(5.0,2.9))
    ax.plot(rb["min_concepts"], rb["median_h_ratio"],marker="o",lw=1.25,c="#326A8A")
    ax.axhline(1,ls="--",c=".35",lw=.9)
    ax.set(xlabel="Minimum documented concepts",ylabel="Median concentration ratio")
    _clean(ax,"a")
    fig.subplots_adjust(left=.18,right=.97,bottom=.23,top=.93)
    _save(fig,dest,"figS_coverage_robustness")

    if {"Macroarea","Family"}.issubset(x.columns):
        fam = x[x["Macroarea"].isin(AMERICAN_MACROAREAS)].dropna(subset=["Family"]).copy()
        fam = fam[~fam["Family"].isin(EXCLUDED_IMPORTED_FAMILIES)]
        if not fam.empty:
            counts = fam.groupby(["Macroarea","Family"]).size().reset_index(name="n")
            picks = pd.concat([g.nlargest(6,"n") for _,g in counts.groupby("Macroarea")],ignore_index=True)
            chosen = set(zip(picks["Macroarea"],picks["Family"]))
            chosen.update((area, family) for area, family in zip(counts["Macroarea"], counts["Family"])
                          if family in FORCE_AMERICAN_FAMILIES)
            selected = fam[[ (a,b) in chosen for a,b in zip(fam["Macroarea"],fam["Family"]) ]].copy()
            cats = sorted(selected["Family"].unique())
            if cats:
                fig,ax=plt.subplots(figsize=(6.4,max(3.0,.31*len(cats)+1)))
                for i, label in enumerate(cats):
                    r = selected[selected["Family"] == label]
                    jitter = np.random.default_rng(2026+i).uniform(-.13,.13,len(r))
                    ax.scatter(r["h_ratio"],i+jitter,c="#326A8A",s=14,alpha=.7,lw=0)
                    ax.plot([r["h_ratio"].median()]*2,[i-.19,i+.19],lw=2,c="#BA5B3B")
                ax.axvline(1,ls="--",c=".35",lw=.9)
                ax.set_yticks(range(len(cats)),cats)
                ax.invert_yaxis()
                ax.set_xlabel("Observed / conditional-null concentration")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.subplots_adjust(left=.30,right=.98,bottom=.14,top=.97)
                _save(fig,dest,"figS_american_families_exploratory")
                selected.to_csv(tables/"american_families_selected_data.csv",index=False)
    print("Figures saved in", dest)
