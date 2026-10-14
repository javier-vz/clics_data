"""Windows/Anaconda-friendly command-line interface with resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import run_analysis
from .figures import plot_all
from .io import file_sha256, read_forms, read_languages, store_provenance
from .report import create_report


def _locations(root: Path):
    return (root/"data"/"processed",root/"results"/"tables",
            root/"results"/"figures",root/"results"/"reports")


def prepare(root: Path, forms_path: Path, languages_path: Path, reuse: bool = True) -> tuple[pd.DataFrame,pd.DataFrame,str]:
    processed,_,_,_= _locations(root)
    processed.mkdir(parents=True,exist_ok=True)
    pairs_path=processed/"pairs.csv"
    metadata_path=processed/"languages_metadata.csv"
    manifest=processed/"source_manifest.json"
    for p in (forms_path,languages_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}. Copy the source files to data/raw/ or use --forms / --languages.")
    provenance={
        "forms_path":str(forms_path.resolve()),
        "forms_sha256":file_sha256(forms_path),
        "languages_path":str(languages_path.resolve()),
        "languages_sha256":file_sha256(languages_path),
        "normalization":"NFC + casefold + whitespace-collapse; retain accents; unique Language_ID/Parameter_ID/form_norm",
    }
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    provenance["input_fingerprint"]=fingerprint
    if (reuse and pairs_path.exists() and metadata_path.exists() and manifest.exists()
        and json.loads(manifest.read_text(encoding="utf-8"))==provenance):
        print("Reusing validated clean data (input hashes unchanged).",flush=True)
        pairs=pd.read_csv(pairs_path,dtype=str,keep_default_na=False)
        metadata=pd.read_csv(metadata_path,dtype=str,keep_default_na=False)
    else:
        pairs=read_forms(forms_path)
        metadata=read_languages(languages_path)
        pairs.to_csv(pairs_path,index=False,encoding="utf-8-sig")
        metadata.to_csv(metadata_path,index=False,encoding="utf-8-sig")
        store_provenance(manifest,provenance)
    store_provenance(root/"results"/"provenance.json",provenance)
    return pairs,metadata,fingerprint


def _analyze(root: Path, pairs: pd.DataFrame, metadata: pd.DataFrame, fingerprint: str, args) -> None:
    _,tables,_,_=_locations(root)
    run_analysis(pairs,metadata,tables,
                 samples=args.samples,component_samples=args.component_samples,
                 seed=args.seed,swaps_burn=args.swaps_burn,
                 swaps_thin=args.swaps_thin,min_concepts=args.min_concepts,
                 input_fingerprint=fingerprint,resume=not args.no_resume)


def _plot_report(root: Path, args) -> None:
    _,tables,figures,reports=_locations(root)
    plot_all(tables,figures,min_concepts=args.min_concepts)
    create_report(tables,reports)


def demo_data(root: Path) -> tuple[Path,Path]:
    dest=root/"data"/"demo"
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    lang_rows=[]
    rng=np.random.default_rng(2031)
    for lid,area,family in [
        ("101","South America","Quechuan"),
        ("102","South America","Aymaran"),
        ("103","North America","Mayan"),
        ("104","Eurasia","Indo-European"),
        ("105","South America","Tupian"),
    ]:
        lang_rows.append({"ID":lid,"Name":"Synthetic language "+lid,"Macroarea":area,"Family":family})
        m=35
        for j in range(m):
            form_idx=j if j < 20 else int(rng.integers(25))
            rows.append({"Language_ID":lid,"Parameter_ID":f"c{j:03d}","Form":f"form_{form_idx:03d}"})
            if j%13==0:
                rows.append({"Language_ID":lid,"Parameter_ID":f"c{j:03d}","Form":f"alternative_{j:03d}"})
    forms=dest/"forms_demo.csv"
    languages=dest/"languages_demo.csv"
    pd.DataFrame(rows).to_csv(forms,index=False)
    pd.DataFrame(lang_rows).to_csv(languages,index=False)
    return forms,languages


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(
        prog="lexrow",description="CLICS empirical row-occupancy pipeline for Windows/Anaconda."
    )
    sub=parser.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--root",type=Path,default=Path.cwd(),help="Repository root; default: current directory")
    opts=argparse.ArgumentParser(add_help=False,parents=[common])
    opts.add_argument("--samples",type=int,default=199,help="IID conditional-surjection replicates per language (final analyses: 999+)")
    opts.add_argument("--component-samples",type=int,default=49,help="Degree-preserving bipartite MCMC replicates")
    opts.add_argument("--seed",type=int,default=2026)
    opts.add_argument("--min-concepts",type=int,default=500)
    opts.add_argument("--swaps-burn",type=int,default=10,help="Burn-in attempted edge swaps per association")
    opts.add_argument("--swaps-thin",type=int,default=3,help="Between-sample attempted edge swaps per association")
    opts.add_argument("--no-resume",action="store_true",help="Recompute nulls even when matching checkpoint exists")
    input_args=argparse.ArgumentParser(add_help=False,parents=[opts])
    input_args.add_argument("--forms",type=Path,default=Path("data/raw/forms.xlsx"))
    input_args.add_argument("--languages",type=Path,default=Path("data/raw/languages.csv"))
    sub.add_parser("prepare",parents=[input_args],help="Clean Excel/CSV inputs; keep source provenance")
    sub.add_parser("run",parents=[input_args],help="Prepare, calculate both nulls, plot and report")
    sub.add_parser("analyze",parents=[opts],help="Run both nulls on already prepared pairs.csv")
    sub.add_parser("plot",parents=[common],help="Recreate publication figures from result CSVs").add_argument("--min-concepts",type=int,default=500)
    sub.add_parser("report",parents=[common],help="Regenerate text report from result CSVs")
    sub.add_parser("demo",parents=[opts],help="Generate synthetic demo inputs and run entire pipeline")
    args=parser.parse_args(argv)
    root=args.root.resolve()
    if args.command=="demo":
        forms,langs=demo_data(root)
        p,m,signature=prepare(root,forms,langs,reuse=False)
        _analyze(root,p,m,signature,args)
        _plot_report(root,args)
    elif args.command in ("prepare","run"):
        forms=args.forms if args.forms.is_absolute() else root/args.forms
        langs=args.languages if args.languages.is_absolute() else root/args.languages
        p,m,signature=prepare(root,forms,langs)
        if args.command=="run":
            _analyze(root,p,m,signature,args)
            _plot_report(root,args)
    elif args.command=="analyze":
        processed,_,_,_=_locations(root)
        manifest=processed/"source_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("First run: lexrow prepare --forms ... --languages ...")
        signature=json.loads(manifest.read_text(encoding="utf-8"))["input_fingerprint"]
        p=pd.read_csv(processed/"pairs.csv",dtype=str,keep_default_na=False)
        m=pd.read_csv(processed/"languages_metadata.csv",dtype=str,keep_default_na=False)
        _analyze(root,p,m,signature,args)
    elif args.command=="plot":
        _,tables,figures,_=_locations(root)
        plot_all(tables,figures,min_concepts=args.min_concepts)
    elif args.command=="report":
        _,tables,_,reports=_locations(root)
        create_report(tables,reports)
    print("Done.",flush=True)

if __name__=="__main__":
    main()
