# Statistical methods and validation checklist

## Definitions

For each language ℓ, build a binary edge set E_ℓ of observed `(form, concept)` associations. A=|E|, M=# documented concepts, V=# active distinct forms. Define rho_i as the number of concepts linked to form i. Every rho_i≥1 and Σrho_i=A. The active-vocabulary fraction `sV* = V/A` is a descriptive statistic, **not a test statistic** under a null that conditions on A and V.

Concentration: `h_A = Σ rho_i(rho_i−1) / [A(A−1)]`. For A=V, all rho_i=1, h=0 and there is no occupancy variation; h-ratios are undefined. The minimum feasible concentration for A=qV+r, 0≤r<V, is `[ (V−r) q(q−1) + r(q+1)q ]/[A(A−1)]`.

## Occupancy null — uniformly random surjections

Let the A association tokens be distinguishable, and V observed forms be labeled. A uniformly random function f:[A]→[V] conditioned to hit all V forms is a uniformly random surjection. The probability of a particular feasible occupancy vector rho is

`P(rho | A,V, onto) = (A! / prod_i rho_i!) / (V! S(A,V))`,

where S(A,V) is a Stirling number of the second kind. In particular, the random occupancy baseline `h=1/V` **does not apply unchanged**, since conditioning on all V forms being occupied changes collision probabilities.

### Exact IID sampling proof

Draw V independent positive Poisson variables `X_i ~ Poisson(lambda) | X_i>0`. Their joint probability is proportional to `lambda^(sum X_i) / prod(X_i!)`. Upon conditioning on `sum X_i = A`, the lambda term is constant, leaving probability proportional to `1/prod(X_i!)`. Thus the conditional law of X is precisely the uniform-surjection occupancy law. We tune lambda to satisfy `lambda/(1-exp(-lambda)) = A/V`, which makes the target sum typical and helps rejection efficiency; lambda is **not fitted from lexical outcomes**. To reduce computation, probabilities of X_i > A−V+1 can be dropped before rejection because those counts cannot occur when the sum is exactly A. The sampler is **exact up to floating-point arithmetic**, and samples are independent.

For a language, report the mean and standard deviation of h under this null, empirical/null ratio, lower/upper/two-sided Monte Carlo tail probabilities, and the rho_max and forms-with-multiple-concepts diagnostics. Zero-variance null cases must not be assigned arbitrary z-scores.

## Graph null — exact degrees, nonuniform finite-length MCMC

Random binary form–concept edges by selecting distinct edges `(c1,f1)`, `(c2,f2)` with `c1!=c2`, `f1!=f2`, and replacing them by `(c1,f2)`, `(c2,f1)` only if neither candidate exists. Reject otherwise. This leaves the exact degree of **each** concept and each form unchanged, and does not introduce parallel edges. A standard double-edge-swap chain is symmetric over allowed switches; sampling after burn-in and thinning approximates the stationary fixed-degree ensemble if the chain is well mixed. Do not claim independent exact-uniform network samples with finite swaps.

Concept projection connects distinct concepts sharing a form. Compute the largest component fraction and the fraction of isolated concepts. Because the form degrees are fixed, this null is **structural**, not a reference distribution for h_A.

## Interpretation and sensitivity analyses

1. Run with at least 999 IID occupation replicates per language for finer tail resolution (one-sided p_min=0.001), and check results at ≥1999 if many languages approach the floor.
2. Compare results at `M>=250, 500, 750, 1000, 1250, 1500`.
3. For the graph null, compare `--swaps-burn 10 --swaps-thin 3` vs `20,5` and e.g. `40,10`, and multiple seeds. Assess stability of component-null means and quantiles.
4. Consider excluding loan-specific entries, forms with orthographic variants, and overly ambiguous tokenization, if source fields make this possible. This would require a documented extension of the initial data-preparation policy.
5. Control phylogenetic dependence using family-level summaries/cluster bootstrap and, if possible, phylogenetic comparative methods. Macroarea is a descriptive contextual variable, not a causal mechanism.
6. The present design characterizes marginal occupancy and conditional network topology. Inferring *historical* change or physical equilibrium from these correlations requires additional models and evidence.

## Mathematical subtlety: counting constraints

A language may have more than one form for the same concept (synonymy). The association count A therefore may exceed M. Uniform surjections treat pair tokens as exchangeable and do **not** preserve the per-concept counts. The second graph null preserves these degrees, but because it also fixes occupancies, cannot independently test concentration. A later sensitivity null could preserve concept degrees while conditioning V and randomizing form occupancy (without fixing form degrees), subject to no duplicate binary edges. This is separate future work, rather than a claim that the present nulos test every dimension simultaneously.
