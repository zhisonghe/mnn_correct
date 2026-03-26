"""mnn_correct — MNN-based batch correction for AnnData objects."""

from .correct import MNNCorrector, mnn_correct, mnn_correct_adata

__version__ = "0.1.0"
__all__ = ["MNNCorrector", "mnn_correct", "mnn_correct_adata"]
