# mnn-correct

`mnn-correct` provides mutual-nearest-neighbour-based batch correction for
`AnnData` objects. The package is centered on a stateful `MNNCorrector` that
supports three steps:

1. `fit()` estimates per-batch displacement models.
2. `correct()` applies the fitted correction to the training dataset.
3. `project()` propagates an already learned batch displacement onto new cells
	based on a batch column in the query `AnnData`.

By default, `fit()` updates the current `MNNCorrector` in place and returns
`None`. Pass `return_corrector=True` if you want it to return the fitted
corrector instance for chaining.

## Installation

Install the package in editable mode for development:

```bash
pip install -e ".[dev]"
```

Core runtime dependencies are listed in [pyproject.toml](pyproject.toml).

## Concepts

The corrector expects a shared representation for all cells that should be
aligned. In practice this is usually something like:

- `adata.obsm["X_pca"]`
- `adata.obsm["X_scVI"]`
- `adata.obsm["X_harmony"]`

If no representation is supplied, the corrector can fit a PCA model directly on
`adata.X` by passing `use_rep=None`. This PCA fallback is only used for the
current correction run and does not support future projection.

During fitting, the model:

1. identifies mutual nearest neighbour pairs between a reference and query
   batch,
2. estimates anchor-cell displacement vectors,
3. propagates those displacements to all cells in the query batch,
4. stores batch-specific projection state for later reuse when a reusable
	representation was supplied.

## Main Workflow

### 1. Fit a correction model

```python
from mnn_correct import MNNCorrector

corrector = MNNCorrector(
	k_mnn=10,
	k_propagate=20,
	weighting_scheme="jaccard_square",
)

corrector.fit(
	adata,
	batch_key="batch",
	batch_order=["reference", "query_a", "query_b"],
	use_rep="X_scVI",
)

# Optional chaining-friendly form:
same_corrector = corrector.fit(
	adata,
	batch_key="batch",
	batch_order=["reference", "query_a", "query_b"],
	use_rep="X_scVI",
	return_corrector=True,
)
```

### 2. Apply the fitted correction

```python
corrector.correct(adata)

corrected = adata.obsm["X_scVI_mnn_corrected"]
```

### 3. Project new cells from fitted batches

```python
new_adata.obs["batch"] = ["query_b", "query_b", "reference", ...]

corrector.project(
	new_adata,
	batch_key="batch",
)

projected = new_adata.obsm["X_scVI_mnn_corrected"]
```

`project()` validates that every batch category in `new_adata.obs[batch_key]`
was seen during `fit()`. Cells from the initial sequential batch or the fixed
reference batch are projected as an identity mapping. If you need to apply the
learned correction back to the original training dataset, use `correct()`
instead.

If `fit()` was run with `use_rep=None`, `project()` is unavailable because the
PCA fallback is not stored as a reusable projection model.

## Correction Modes

### Sequential correction

In sequential mode, batches are corrected one after another. Each corrected
batch is merged into the growing reference before the next round.

```python
corrector.fit(
	adata,
	batch_key="batch",
	batch_order=["batch0", "batch1", "batch2"],
	use_rep="X_pca",
)
```

This is useful when the batches form a natural progression or when no single
batch should be privileged as the sole reference.

### Fixed-reference correction

In fixed-reference mode, each non-reference batch is corrected independently
against one chosen reference batch.

```python
corrector.fit(
	adata,
	batch_key="batch",
	reference="atlas",
	use_rep="X_scVI",
)
```

This is useful when one batch serves as the canonical reference, such as an
atlas or a well-curated control dataset.

## Convenience Wrappers

### `mnn_correct()`

Use `mnn_correct()` when you already have separate reference and query
`AnnData` objects.

```python
from mnn_correct import mnn_correct

corrector = mnn_correct(
	adata_ref,
	adata_query,
	use_rep="X_scVI",
	batch_label="query_batch",
)

corrected_query = adata_query.obsm["X_scVI_mnn_corrected"]
```

The reference object is left unchanged. The corrected embedding is written only
to the query object.

### `mnn_correct_adata()`

Use `mnn_correct_adata()` when all batches are already contained in a single
`AnnData` object.

```python
from mnn_correct import mnn_correct_adata

_, corrector = mnn_correct_adata(
	adata,
	batch_key="batch",
	batch_order=["batch0", "batch1", "batch2"],
	use_rep="X_pca",
)
```

## Output Keys

By default, corrected embeddings are written to:

- `"{use_rep}_mnn_corrected"` when `use_rep` is provided
- `"X_pca_mnn_corrected"` when `use_rep=None`

You can override this with `key_added` in `fit()`, `correct()`, `project()`,
`mnn_correct()`, or `mnn_correct_adata()`.

## Important Behavior

- `fit()` does not modify the input `AnnData`.
- `fit()` updates the `MNNCorrector` in place and returns `None` unless
	`return_corrector=True` is passed.
- `correct()` applies only to the same fitted dataset and validates that both
  the cells and source representation match the fitted state.
- `project()` is for new cells whose batch assignments are stored in a column of
	`adata.obs`, and every batch in that column must already be known from
	`fit()`.
- Projection state is stored per batch label in `corrector.projection_data_`
	only when a reusable representation was supplied.
- The initial sequential batch or fixed reference batch projects as an identity
	mapping and is stored separately from `corrector.projection_data_`.
- If `use_rep=None`, `fit()` falls back to PCA for the current correction and
	forces `store_for_projection=False`.

## Parameters That Matter Most

- `k_mnn`: number of neighbors used to identify MNN pairs.
- `k_propagate`: number of neighbors used to smooth anchor-cell displacement to
  the full batch.
- `weighting_scheme`: how propagation edges are weighted. The default is
  `"jaccard_square"`.
- `use_rep`: which latent embedding to correct.

## Development

Run the test suite:

```bash
pytest
```

Run linting:

```bash
ruff check src tests
```

Run type checking:

```bash
mypy src
```

## License

MIT
