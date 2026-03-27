"""Shared helper utilities for correction workflows."""

from __future__ import annotations

import datetime
from typing import Literal

import numpy as np
from scipy.sparse import diags

from . import wknn


def tprint(*args, **kwargs):
    """Print with an ISO-8601 timestamp prefix."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)

WeightingScheme = Literal[
    "n",
    "top_n",
    "jaccard",
    "jaccard_square",
    "gaussian",
    "dist",
]
NeighborFlavor = wknn.NeighborFlavor


def propagate_weighted(
    emb_new: np.ndarray,
    ref_emb: np.ndarray,
    ref_disp: np.ndarray,
    k: int,
    weighting_scheme: WeightingScheme,
    flavor: NeighborFlavor,
    verbose: bool,
) -> np.ndarray:
    """Propagate known displacement vectors from reference cells to new cells.

    Parameters
    ----------
    emb_new
        Embeddings of cells that should receive a correction.
    ref_emb
        Embeddings of reference cells with known displacement vectors.
    ref_disp
        Displacement vectors associated with ``ref_emb``.
    k
        Number of neighbours used during propagation.
    weighting_scheme
        Weighting scheme used by the weighted KNN graph.
    flavor
        Neighbor-search backend.
    verbose
        If ``True``, print neighbour-search progress messages.

    Returns
    -------
    numpy.ndarray
        Propagated displacement matrix with one row per cell in ``emb_new``.
    """
    k_eff = min(k, ref_emb.shape[0])
    wknn_prop = wknn.get_wknn(
        ref=ref_emb,
        query=emb_new,
        k=k_eff,
        query2ref=True,
        ref2query=False,
        weighting_scheme=weighting_scheme,
        flavor=flavor,
        verbose=verbose,
    )
    row_sums = np.array(wknn_prop.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    return diags(1.0 / row_sums).dot(wknn_prop).dot(ref_disp)