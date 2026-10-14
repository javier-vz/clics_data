# CLICS row-occupancy analysis

**Reproducible local analysis of lexical row occupancy and colexification components, with two deliberately different null models.**

This repository connects a row-occupancy perspective on form–concept networks with empirical CLICS records. It **does not assume in advance** that empirical languages are more or less concentrated than an appropriate null. All conclusions are derived from the outputs.

## 1. Quick start — Anaconda Prompt (Windows)

Download/extract this folder, open **Anaconda Prompt**, then:

```bat
cd C:\ruta\donde\guardaste\clics-row-occupancy
conda env create -f environment.yml
conda activate clics-row
python -m pytest -q
python -m lexrow demo --samples 49 --component-samples 19 --min-concepts 20
```

The demo uses **synthetic** language records, not real CLICS observations. Inspect generated PDFs in `results/figures/`, CSVs in `results/tables/`, and `results/reports/summary.md`.

> If `conda env create` complains that an environment named `clics-row` already exists, use `conda env update -f environment.yml --prune` instead.

### Actual CLICS inputs

Copy your source files into `data/raw/`, keeping the original files separate from generated outputs:

```text
data/raw/forms.xlsx
data/raw/languages.csv
```

The provided CLICS workbook has the columns `Language_ID`, `Parameter_ID` and `Form`. The language metadata CSV has `ID`, `Name`, `Family` and `Macroarea`. If you kept the original filename `languages(2).csv`, specify it in quotes (see below).

Then run:

```bat
python -m lexrow run --forms data/raw/forms.xlsx --languages data/raw/languages.csv --samples 199 --component-samples 49 --min-concepts 500 --seed 2026
```

Or, using the original metadata filename:

```bat
python -m lexrow run --forms data/raw/forms.xlsx --languages "data/raw/languages(2).csv" --samples 199 --component-samples 49 --min-concepts 500
```

**First Excel import may take several minutes:** `forms.xlsx` contains many rows and a large shared-string table. Subsequent runs reuse the cleaned CSV if the source SHA-256 hashes are unchanged.

The pipeline checkpoints each language under `results/checkpoints/`. Rerunning the *same command* resumes completed languages without recomputing them. To explicitly recompute, use `--no-resume`. To change sample counts, pass the new values (a new checkpoint configuration is created).

### Final-quality run

Start with 199 / 49 samples for engineering checks. For the publication figures and inference, use e.g.:

```bat
python -m lexrow analyze --samples 999 --component-samples 199 --min-concepts 500 --seed 2026 --swaps-burn 20 --swaps-thin 5
python -m lexrow plot --min-concepts 500
python -m lexrow report
```

The second command can be **computationally expensive**, particularly the bipartite edge-swap null for all languages. The checkpoint system supports interruption and restart. The occupancy null is IID, but the graph null is a Markov chain; for publication, compare results under larger `--swaps-burn`, `--swaps-thin`, and alternate `--seed` values to evaluate stability.

## 2. Scientific design (three stages)

### Stage I — Define comparable nulls

For each language, let **A** be the number of *distinct form–concept pairs*, **V** the number of observed distinct *active forms*, and **M** the number of documented concepts. Form occupancy is `rho_i = number of distinct concepts attached to form i`, with `sum(rho_i)=A` and all `rho_i>=1`.

The concentration statistic is

```text
h_A = [sum_i rho_i (rho_i - 1)] / [A (A - 1)]
```

It is the probability that two distinct randomly sampled association tokens belong to the same active form. Because V counts already-active forms, the original *unconditioned* random assignment onto V bins is an unsuitable comparator for empirical active-vocabulary dispersion. The new null **fixes A, fixes V and conditions every one of those V forms to be occupied**.

**Null 1 — Uniform surjection (occupation).** Uniformly allocate A distinguishable associations to V labeled active forms, *conditioned on all V receiving at least one association*. The induced probability over occupancy vectors is proportional to `1 / prod(rho_i!)`. Counts are drawn *exactly and independently* using zero-truncated Poisson random variables conditioned on their sum (rejection sampling); no MCMC convergence issue applies to this null. A and V are identical between observations and null, so **sV*=V/A is identical and is not a tested outcome**. We compare `h_A`, the maximum occupancy, number of forms used at least twice and at least three times.

**Null 2 — Fixed bipartite degrees (components).** Construct the binary form–concept graph per language and randomize it using valid bipartite double-edge swaps. This maintains *every individual concept degree* (synonymy) and *every individual form degree* (occupancy) while preventing duplicate binary edges. Project onto concepts via shared forms; compare the largest connected component and isolated-concept fraction. This **does not test row occupancy** because it preserves occupancies by construction. Unlike the surjection null, the edge-swap samples form an MCMC trajectory, so simulation length and mixing must be examined.

### Stage II — Empirical estimation and robustness

- Reproducible cleaning: Unicode NFC, lowercase via `casefold`, collapse whitespace, preserve diacritics; deduplicate language × concept × form; no unadvertised phonetic transformations.
- Analyze all available languages and filter the primary plots at `M>=500` (configurable). Retain every language in `language_metrics.csv`.
- Save *every simulated null draw*, source hashes, RNG seeds, p-values with +1 Monte Carlo correction and BH-FDR q-values.
- Robustness across concept thresholds. Metadata by macroarea/family for descriptive, non-phylogenetic comparisons.
- Separately inspect documentation coverage, normalization choices, family dependence and null-chain sensitivity before interpreting the patterns.

### Stage III — Figures and manuscript

- `fig1_conditional_occupancy`: observed vs conditional-null h; ratio distribution; ratio vs coverage.
- `fig2_colexification_components`: observed vs fixed-degree network null.
- `figS_coverage_robustness`: medians across coverage filters.
- `figS_american_families_exploratory`: descriptive family visualization (six most represented families per American macroarea, plus prespecified Quechuan/Quechua/Aymaran where present; macroarea selection is *not* an assertion of indigenous identity for every record).
- `reports/summary.md`: automatically generated **after** measuring the observations, so it can reveal small, absent, reversed or mixed effects.

See [docs/METHODS.md](docs/METHODS.md) for formulas, assumptions, and suggested additional controls, and [paper/OUTLINE.md](paper/OUTLINE.md) for a conservative article structure.

## 3. Output files

```text
src/lexrow/                   analysis source code
src/lexrow/cli.py             terminal commands
tests/                        unit tests
data/raw/                      you provide CLICS sources; .gitignored
data/processed/pairs.csv      normalized unique form–concept pairs (generated)
data/processed/source_manifest.json    hashes and normalization (generated)
results/checkpoints/           per-language resumable checkpoints (generated)
results/tables/language_metrics.csv
results/tables/occupancy_null_draws.csv
results/tables/component_null_draws.csv
results/tables/robustness_thresholds.csv
results/tables/family_summary.csv
results/tables/analysis_config.json
results/figures/*.pdf, *.png
results/reports/summary.md
results/provenance.json
```

The data folders, checkpoints and generated results are excluded from Git by default. To publish non-sensitive derived figures/tables, create a `release/` folder and explicitly copy vetted outputs there. Review CLICS redistribution and citation terms before committing the raw data.

## 4. Useful commands

```bat
python -m lexrow --help
python -m lexrow run --help
python -m lexrow prepare --forms data/raw/forms.xlsx --languages data/raw/languages.csv
python -m lexrow analyze --samples 199 --component-samples 49
python -m lexrow plot --min-concepts 500
python -m lexrow report
python -m pytest -q
```

Alternative launcher (after installing dependencies) if the `lexrow` module command does not resolve:

```bat
python run_pipeline.py run --forms data/raw/forms.xlsx --languages data/raw/languages.csv
```

## 5. Publish to GitHub

Create a **new empty repository** on GitHub, then from Anaconda Prompt:

```bat
git init
git add .
git commit -m "Initial reproducible row-occupancy pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the URL with the actual repository. Choose the code's license explicitly (e.g. MIT or GPL) before publishing. Add a CLICS source version/date and official citation in [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md); the sample's numerical results should not be cited as real CLICS findings.
