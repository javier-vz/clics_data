"""Input ingestion, quality checks, deterministic normalization and provenance."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED = ["Language_ID", "Parameter_ID", "Form"]


def normalize_form(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().strip().split())


def normalize_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    return re.sub(r"^(\d+)\.0+$", r"\1", s)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_forms(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        # CLICS original forms.xlsx uses shared strings and can take several minutes.
        print("Reading Excel workbook (first time may be slow):", path, flush=True)
        df = pd.read_excel(path, usecols=REQUIRED, engine="openpyxl", dtype=str)
    elif path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(path, usecols=REQUIRED, sep="\t" if path.suffix.lower()==".tsv" else ",", dtype=str,
                         low_memory=False, encoding="utf-8-sig")
    else:
        raise ValueError("forms file must be .xlsx, .xlsm, .csv, or .tsv")
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in forms: {sorted(missing)}")
    before = len(df)
    df = df[REQUIRED].copy()
    for c in ("Language_ID", "Parameter_ID"):
        df[c] = df[c].map(normalize_id)
    df["form_norm"] = df["Form"].map(normalize_form)
    df = df[(df["Language_ID"]!="") & (df["Parameter_ID"]!="") & (df["form_norm"]!="")].copy()
    df = df[["Language_ID", "Parameter_ID", "form_norm"]].drop_duplicates()
    df = df.sort_values(["Language_ID", "Parameter_ID", "form_norm"], kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable language--concept--form associations after cleaning")
    print(f"Cleaned forms: {before:,} input rows -> {len(df):,} unique associations", flush=True)
    return df


def read_languages(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Languages metadata not found: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    if "ID" not in df.columns:
        if "Language_ID" in df.columns:
            df = df.rename(columns={"Language_ID": "ID"})
        else:
            raise ValueError("Metadata must include ID (or Language_ID) column")
    df["Language_ID"] = df["ID"].map(normalize_id)
    keep = ["Language_ID"] + [c for c in ("Name", "Family", "Macroarea", "Glottocode", "ISO639P3code") if c in df.columns]
    return df[keep].drop_duplicates("Language_ID")


def store_provenance(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
