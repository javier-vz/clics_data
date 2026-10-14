# Validation record

Implemented tests (`python -m pytest -q`):

1. Unicode normalization and source-language identifier normalization.
2. Exact-surjection generator always produces V positive occupancies summing to A.
3. Small analytically known case (A=4, V=2): marginal occupancies must occur in proportions 2/7, 3/7 and 2/7.
4. Correct minimum-concentration formula and A=V degenerate limit.
5. Correct component structure for a small concept–form graph and consistency of degree-preserving randomization.

An end-to-end synthetic-data run completed: five synthetic languages, 24 occupation-null draws per language and 12 component-null draws per language. All expected tables, four PDFs/PNGs and a data-driven report were generated.

**Scope:** A full empirical run on original CLICS forms.xlsx has not been performed here. As a result no empirical effect size, significance claim or revised language count has been verified using this new null.

Do not publish synthetic demo outputs as empirical results.
