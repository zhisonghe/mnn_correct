"""Weighted k-nearest-neighbour graph utilities for MNN batch correction."""

from __future__ import annotations

import importlib.util
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from pynndescent import NNDescent
from scipy import sparse
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors

NeighborFlavor = Literal["auto", "cuml", "sklearn", "pynndescent"]


def gaussian_kernel(d, sigma=None):
    """Apply a Gaussian kernel to distances *d*.

    Parameters
    ----------
    d
        Array of distances.
    sigma
        Bandwidth.  Defaults to ``max(d) / 3``.

    Returns
    -------
    numpy.ndarray
        Distance values transformed by a Gaussian kernel.
    """
    if sigma is None:
        sigma = np.max(d) / 3
    return np.exp(-0.5 * np.square(d) / np.square(sigma))


def nn2adj(
    nn,
    n1=None,
    n2=None,
    weight: Literal["unweighted", "dist", "gaussian_kernel"] = "unweighted",
    sigma=None,
):
    """Convert a pynndescent-style ``(indices, distances)`` tuple to a sparse adjacency matrix.

    Parameters
    ----------
    nn
        ``(indices, distances)`` tuple returned by :class:`~pynndescent.NNDescent`.
    n1
        Number of rows (query cells).  Inferred from ``nn[0]`` when ``None``.
    n2
        Number of columns (reference cells).  Inferred from ``nn[0]`` when ``None``.
    weight
        ``"unweighted"``: binary matrix; ``"dist"``: raw distances;
        ``"gaussian_kernel"``: Gaussian-kernel-transformed distances.
    sigma
        Bandwidth for the Gaussian kernel (ignored unless ``weight="gaussian_kernel"``).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix with shape ``(n1, n2)``.
    """
    if n1 is None:
        n1 = nn[0].shape[0]
    if n2 is None:
        n2 = int(np.max(nn[0].flatten())) + 1

    df = pd.DataFrame(
        {
            "i": np.repeat(range(nn[0].shape[0]), nn[0].shape[1]),
            "j": nn[0].flatten(),
            "x": nn[1].flatten(),
        }
    )

    if weight == "unweighted":
        adj = sparse.csr_matrix(
            (np.ones(df.shape[0], dtype=np.float32), (df["i"], df["j"])),
            shape=(n1, n2),
        )
    else:
        if weight == "gaussian_kernel":
            df["x"] = gaussian_kernel(df["x"], sigma)
        adj = sparse.csr_matrix((df["x"], (df["i"], df["j"])), shape=(n1, n2))

    return adj


def build_nn(
    ref,
    query=None,
    k: int = 100,
    weight: Literal["unweighted", "dist", "gaussian_kernel"] = "unweighted",
    sigma=None,
    flavor: NeighborFlavor = "auto",
    verbose: bool = False,
):
    """Build a k-nearest-neighbour graph from *query* into *ref*.

    Supports exact CPU neighbours via scikit-learn, approximate CPU neighbours
    via :class:`~pynndescent.NNDescent`, and GPU neighbours via cuML.

    ``flavor="auto"`` selects the backend as follows:

    1. Use scikit-learn when both ``ref`` and ``query`` contain fewer than
       1000 cells.
    2. Otherwise use cuML when CUDA is available and ``cuml`` is installed.
    3. Otherwise fall back to pynndescent.

    Parameters
    ----------
    ref
        Reference embedding, shape ``(n_ref, d)``.
    query
        Query embedding, shape ``(n_query, d)``.  Defaults to *ref* (self-graph).
    k
        Number of neighbours per query cell.
    weight
        Edge-weighting scheme (see :func:`nn2adj`).
    sigma
        Gaussian-kernel bandwidth (ignored unless ``weight="gaussian_kernel"``).
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print backend selection message.

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix of shape ``(n_query, n_ref)``.
    """
    if query is None:
        query = ref

    if flavor not in ("auto", "cuml", "sklearn", "pynndescent"):
        raise ValueError(
            f"Unsupported flavor '{flavor}'. Expected one of "
            "'auto', 'cuml', 'sklearn', or 'pynndescent'."
        )

    has_gpu = torch.cuda.is_available()
    has_cuml = importlib.util.find_spec("cuml") is not None

    if flavor == "auto":
        if ref.shape[0] < 1000 and query.shape[0] < 1000:
            backend = "sklearn"
        elif has_gpu and has_cuml:
            backend = "cuml"
        else:
            backend = "pynndescent"
    else:
        backend = flavor

    if backend == "cuml":
        if not has_gpu or not has_cuml:
            raise RuntimeError(
                "flavor='cuml' requires both CUDA availability and the cuml package."
            )
        if verbose:
            print("GPU detected and cuml installed. Use cuML for neighborhood estimation...")
        from cuml.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=k)
        model.fit(ref)
        knn = (model.kneighbors(query)[1], model.kneighbors(query)[0])
    elif backend == "sklearn":
        if verbose:
            print("Using exact neighborhood estimation with scikit-learn")
        model = SklearnNearestNeighbors(n_neighbors=k)
        model.fit(ref)
        distances, indices = model.kneighbors(query)
        knn = (indices, distances)
    else:
        if verbose:
            print("Falling back to neighborhood estimation using CPU with pynndescent")
        index = NNDescent(ref)
        knn = index.query(query, k=k)

    return nn2adj(knn, n1=query.shape[0], n2=ref.shape[0], weight=weight, sigma=sigma)


def build_mutual_nn(
    dat1,
    dat2=None,
    k1: int = 100,
    k2: Optional[int] = None,
    flavor: NeighborFlavor = "auto",
    verbose: bool = True,
):
    """Return the mutual nearest-neighbour adjacency matrix between *dat1* and *dat2*.

    Parameters
    ----------
    dat1
        Embedding of the first dataset, shape ``(n1, d)``.
    dat2
        Embedding of the second dataset, shape ``(n2, d)``.  Defaults to *dat1*.
    k1
        Number of neighbours from *dat2* into *dat1*.
    k2
        Number of neighbours from *dat1* into *dat2*.  Defaults to *k1*.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print backend selection messages.

    Returns
    -------
    scipy.sparse.csr_matrix
        Element-wise product of the two directed NN graphs, shape ``(n1, n2)``.
        A non-zero entry ``(i, j)`` means cell *i* in *dat1* and cell *j* in
        *dat2* are mutual nearest neighbours.
    """
    if dat2 is None:
        dat2 = dat1
    if k2 is None:
        k2 = k1

    adj_21 = build_nn(
        ref=dat1,
        query=dat2,
        k=k1,
        weight="unweighted",
        flavor=flavor,
        verbose=verbose,
    )
    adj_12 = build_nn(
        ref=dat2,
        query=dat1,
        k=k2,
        weight="unweighted",
        flavor=flavor,
        verbose=verbose,
    )

    return adj_12.multiply(adj_21.T)


def get_wknn(
    ref,
    query,
    ref2=None,
    k: int = 100,
    query2ref: bool = True,
    ref2query: bool = True,
    weighting_scheme: Literal[
        "n", "top_n", "jaccard", "jaccard_square", "gaussian", "dist"
    ] = "jaccard_square",
    top_n: Optional[int] = None,
    sigma: Optional[float] = None,
    return_adjs: bool = False,
    flavor: NeighborFlavor = "auto",
    verbose: bool = True,
):
    """Build a weighted k-nearest-neighbour graph between *query* and *ref*.

    Parameters
    ----------
    ref
        Reference embedding used to build the ref-query neighbour graph.
    query
        Query embedding.
    ref2
        Secondary reference embedding used for the ref-ref neighbour graph
        (Jaccard-based schemes).  Defaults to *ref*.
    k
        Number of neighbours per cell.
    query2ref
        Include query-to-reference directed edges.
    ref2query
        Include reference-to-query directed edges.
    weighting_scheme
        How to weight edges:

        * ``"n"`` / ``"top_n"`` / ``"jaccard"`` / ``"jaccard_square"`` —
          share-of-neighbours-based weights.
        * ``"dist"`` — raw distances.
        * ``"gaussian"`` — Gaussian-kernel-transformed distances.
    top_n
        Threshold for ``"top_n"`` scheme.  Defaults to ``k // 4``.
    sigma
        Gaussian bandwidth (``"gaussian"`` scheme only).
    return_adjs
        If ``True``, also return intermediate adjacency matrices as a dict.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print progress messages.

    Returns
    -------
    scipy.sparse.csr_matrix or tuple
        Weighted KNN matrix with shape ``(n_ref, n_query)``. When
        ``return_adjs=True``, returns ``(wknn, adjs)`` where ``adjs`` contains
        the intermediate adjacency matrices used during construction.
    """
    weight_for_nn = "dist" if weighting_scheme in ("gaussian", "dist") else "unweighted"

    adj_q2r = build_nn(
        ref=ref, query=query, k=k, weight=weight_for_nn, flavor=flavor, verbose=verbose
    )

    adj_r2q = None
    if ref2query:
        adj_r2q = build_nn(
            ref=query, query=ref, k=k, weight=weight_for_nn, flavor=flavor, verbose=verbose
        )

    if query2ref and not ref2query:
        adj_knn = adj_q2r.T
    elif ref2query and not query2ref:
        adj_knn = adj_r2q
    elif ref2query and query2ref:
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)
    else:
        warnings.warn(
            "At least one of query2ref and ref2query should be True. "
            "Resetting to both=True."
        )
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)

    adj_ref = None
    if weighting_scheme in ("n", "top_n", "jaccard", "jaccard_square"):
        if ref2 is None:
            ref2 = ref
        adj_ref = build_nn(ref=ref2, k=k, flavor=flavor, verbose=verbose)
        num_shared = adj_q2r @ adj_ref.T
        wknn = num_shared.multiply(adj_knn.T).copy()

        if weighting_scheme == "top_n":
            if top_n is None:
                top_n = k // 4 if k > 4 else 1
            wknn = (wknn > top_n) * 1
        elif weighting_scheme == "jaccard":
            wknn.data = wknn.data / (k + k - wknn.data)
        elif weighting_scheme == "jaccard_square":
            wknn.data = (wknn.data / (k + k - wknn.data)) ** 2
    else:
        wknn = adj_knn.T
        if weighting_scheme == "gaussian":
            wknn.data = gaussian_kernel(wknn.data, sigma=sigma)

    if return_adjs:
        adjs = {
            "q2r": adj_q2r,
            "r2q": adj_r2q,
            "knn": adj_knn,
            "r2r": adj_ref,
        }
        return wknn, adjs

    return wknn
