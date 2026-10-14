from collections import Counter
import numpy as np
import pandas as pd
from lexrow.occupancy import sample_surjection_counts, minimal_concentration, concentration
from lexrow.components import bipartite_edges, component_stats, bipartite_swap_chain
from lexrow.io import normalize_form, normalize_id


def test_normalization():
    assert normalize_form("  CAFÉ  ") == "café"
    assert normalize_id("26.0") == "26"
    assert normalize_form("cafe\u0301")=="café"


def test_surjection_constraints():
    draws=sample_surjection_counts(17,12,200,42)
    assert draws.shape==(200,12)
    assert np.all(draws>0)
    assert np.all(draws.sum(axis=1)==17)


def test_surjection_distribution_a4_v2():
    # Uniform onto maps: occupancy first bin 1,2,3 has probabilities 4/14,6/14,4/14.
    draws=sample_surjection_counts(4,2,9000,5)
    p=np.bincount(draws[:,0],minlength=4)[1:4]/len(draws)
    assert np.max(np.abs(p-np.array([2/7,3/7,2/7])))<0.025


def test_h_min_and_degenerate():
    assert minimal_concentration(4,4)==0
    assert abs(minimal_concentration(5,4)-2/20)<1e-12
    assert np.all(sample_surjection_counts(4,4,5,1)==1)
    assert concentration(np.array([1,1,1,1]))==0


def test_graph_components():
    df=pd.DataFrame({"Parameter_ID":["a","b","c","d"],
                     "form_norm":["x","x","y","z"]})
    edges,m,v=bipartite_edges(df)
    r=component_stats(edges,m,v)
    assert r["largest_component_size"]==2
    assert r["nontrivial_components"]==1
    null=bipartite_swap_chain(edges,m,v,samples=8,seed=11)
    assert len(null)==8
    # All forms have occupancies 2,1,1 and concepts all degree 1.
    # Edge swaps cannot change the largest colexification component size 2.
    assert all(x["largest_component_size"]==2 for x in null)
