"""Exact binary bipartite-degree preserving null via valid bipartite edge swaps."""
from __future__ import annotations

import numpy as np


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.size = [1] * n

    def root(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, x: int, y: int) -> None:
        a, b = self.root(x), self.root(y)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.p[b] = a
            self.size[a] += self.size[b]


def bipartite_edges(lang_pairs) -> tuple[list[tuple[int,int]], int, int]:
    concepts = sorted(lang_pairs["Parameter_ID"].unique().tolist())
    forms = sorted(lang_pairs["form_norm"].unique().tolist())
    ci = {v:i for i,v in enumerate(concepts)}
    fi = {v:i for i,v in enumerate(forms)}
    edges = sorted({(ci[c], fi[f]) for c,f in zip(lang_pairs["Parameter_ID"], lang_pairs["form_norm"])})
    return edges, len(concepts), len(forms)


def component_stats(edges: list[tuple[int,int]], n_concepts: int, n_forms: int) -> dict[str,float]:
    dsu = DSU(n_concepts)
    first = [-1] * n_forms
    for c, f in edges:
        if first[f] == -1:
            first[f] = c
        else:
            dsu.join(c, first[f])
    sizes = {}
    for i in range(n_concepts):
        r = dsu.root(i)
        sizes[r] = sizes.get(r, 0) + 1
    s = list(sizes.values())
    return {
        "largest_component_frac": max(s) / n_concepts if s else float("nan"),
        "largest_component_size": max(s) if s else 0,
        "isolated_concepts_frac": sum(x == 1 for x in s) / n_concepts if s else float("nan"),
        "nontrivial_components": sum(x > 1 for x in s),
        "components_total": len(s),
    }


def bipartite_swap_chain(
    observed_edges: list[tuple[int,int]], n_concepts: int, n_forms: int,
    samples: int, seed: int, burnin_multiplier: int = 10,
    thinning_multiplier: int = 3,
) -> list[dict[str,float]]:
    """MCMC on simple bipartite graphs; preserves EVERY concept and form degree exactly.

    This samples degree-preserving graphs by switching pairs of edges, without
    multi-edges. It is a Markov chain; convergence depends on burn-in/thinning.
    It is not an IID or provably uniform sampler at finite chain length.
    """
    if samples < 1:
        raise ValueError("samples must be >=1")
    rng = np.random.default_rng(seed)
    edges = list(observed_edges)
    edge_set = set(edges)
    E = len(edges)

    def attempt(n: int) -> None:
        if E < 2:
            return
        for _ in range(n):
            i = int(rng.integers(E))
            j = int(rng.integers(E-1))
            if j >= i:
                j += 1
            c1, f1 = edges[i]
            c2, f2 = edges[j]
            if c1 == c2 or f1 == f2:
                continue
            a, b = (c1, f2), (c2, f1)
            if a in edge_set or b in edge_set:
                continue
            edge_set.remove(edges[i]); edge_set.remove(edges[j])
            edge_set.add(a); edge_set.add(b)
            edges[i], edges[j] = a, b

    attempt(E * burnin_multiplier)
    result = []
    for _ in range(samples):
        attempt(E * thinning_multiplier)
        result.append(component_stats(edges, n_concepts, n_forms))
    return result


def degrees(edges: list[tuple[int,int]], m: int, v: int) -> tuple[list[int],list[int]]:
    concepts = [0]*m
    forms = [0]*v
    for c,f in edges:
        concepts[c] += 1
        forms[f] += 1
    return concepts, forms
