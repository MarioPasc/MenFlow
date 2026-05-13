# Unified MenFlow HDF5 Schema

Every cohort (BraTS-MEN, BraTS-GLI, MenGrowth, future datasets) is converted to a single `.h5` file with the same layout. Downstream code (encoder, decoder, flow trainer, E1 evaluator) reads only this schema and stays cohort-agnostic. The schema is the contract; every dataset converter must satisfy it on write, every consumer may rely on it.

The single source of truth is `src/menflow/data/h5_schema.py` (`H5Schema` + `validate_h5` + `assert_h5_valid`). Cross-field invariants are checked there; do not duplicate them elsewhere.

## Required root attributes

| Attribute | Type | Purpose |
|---|---|---|
| `schema_version` | string | Currently `"1.0"`. Bump on any breaking change. |
| `dataset_name` | string | Cohort identifier, e.g. `"BraTS-MEN-2023"`, `"MenGrowth"`. |
| `dataset_type` | string | `"cross-sectional"` or `"longitudinal"`. |
| `n_scans` | int | Total rows of `/images`. |
| `n_patients` | int | Unique-patient count. |
| `modalities` | string[M] | Channel-order modality names, e.g. `["t1c","t1n","t2f","t2w"]`. |
| `n_modalities` | int | `M`. |
| `spatial_shape` | int[3] | `(H, W, D)` — every scan in this file has this shape. |
| `spacing_mm` | float[3] | Cohort-uniform voxel spacing. |
| `orientation` | string | Anatomical code (`"RAS"`, `"LAS"`, …). |
| `label_map` | string | JSON-encoded `{int: name}` for tumor mask labels. |
| `intensity_normalized` | bool | False = raw source intensities; True = z-scored / rescaled at write-time. |
| `created_at` | string | ISO-8601 UTC timestamp. |
| `has_segmentation_any` | bool | True iff at least one scan in `/segmentations` is annotated. |

## Required datasets

| Path | Shape | dtype | Notes |
|---|---|---|---|
| `images` | `(N, M, H, W, D)` | float32 | Native source intensities by default. |
| `segmentations` | `(N, H, W, D)` | int8 | Zero-volume entries for unannotated scans. |
| `has_segmentation` | `(N,)` | bool | Distinguishes annotated from placeholder zero-segs. |
| `scan_ids` | `(N,)` | vlen str | Unique per row, e.g. `"BraTS-MEN-00008-000"`. |
| `patient_ids` | `(N,)` | vlen str | Possibly repeated for longitudinal cohorts. |
| `timepoint_idx` | `(N,)` | int32 | 0-based index *within* each patient's scan sequence. |
| `longitudinal/patient_offsets` | `(n_patients+1,)` | int32 | CSR offsets: patient `i` owns scans `[offsets[i]:offsets[i+1]]`. |
| `longitudinal/patient_list` | `(n_patients,)` | vlen str | Unique patient IDs in CSR order. |

## Required groups

| Path | Purpose |
|---|---|
| `longitudinal/` | CSR layout above. Trivial for cross-sectional (`offsets = [0,1,2,…,N]`). |
| `metadata/` | Free-form per-scan covariates (grade, age, sex, subset, …). May be empty but must exist. |

## Optional groups

| Path | Notes |
|---|---|
| `splits/<name>` | int32 indices into `patient_list` (patient-level splits). E.g. `splits/train`, `splits/val`. |
| `metadata/<field>` | One dataset per covariate, length `n_scans`. dtype declared by the converter via `metadata_fields()`. |
| `features/<name>` | Self-describing, cohort-specific per-scan features attached at build time. Each dataset carries `units`, `description`, `source`, `dtype`, `leading_dim` attrs. The group itself carries `schema_version` and `registry_json`. Declared by the converter via `feature_registry()` and materialised by `compute_features()`. |

### Per-feature documentation (`/features/`)

`menflow-build` runs the cheap feature pipeline in the same invocation that
produces the unified H5 — no separate "compute features" step is needed.

The contract lives in `src/menflow/data/features.py`:

- `FeatureSpec(name, dtype, shape, units, description, source, leading_dim, required)`
  declares one feature. Required attributes on the dataset: `units` (`"cm^3"`,
  `"voxels"`, ...), `description` (one-sentence semantic meaning), `source`
  (`"derived:segmentation"`, `"derived:image"`, `"external:<file>"`).
- `FeatureRegistry(dataset_name, features, schema_version)` collects every
  feature for one cohort. It is serialised into `/features/.attrs["registry_json"]`
  so any consumer can introspect what is attached without cohort-specific
  knowledge.
- `validate_features(path)` / `assert_features_valid(path)` enforce that every
  declared feature is present, well-typed, and documented; they are called
  automatically by `H5Converter.convert()` when a feature registry is declared.

Reference implementations:
`src/menflow/data/conversors/brats_men.py` (cross-sectional: `n_voxels_tumor`,
`log_volume_cm3`, `laterality`, `grade`, `age`, `sex`) and
`src/menflow/data/conversors/mengrowth.py` (longitudinal: adds CSR-aware
`delta_log_volume`).

Opt out with `menflow-build --no-features` for raw dry runs.

## Cross-field invariants enforced by the validator

1. `images.shape == (n_scans, n_modalities, *spatial_shape)`.
2. `segmentations.shape == (n_scans, *spatial_shape)`.
3. `len(modalities) == n_modalities`.
4. `longitudinal/patient_offsets[0] == 0`, `[-1] == n_scans`, monotonic non-decreasing.
5. `len(longitudinal/patient_list) == n_patients`.
6. `dataset_type ∈ {"cross-sectional", "longitudinal"}`.
7. `label_map` is valid JSON.
8. Every `<dataset>` listed with `leading_dim="n_scans"` has its first axis equal to `n_scans`.

## Storage policy

- **Default to raw, reversible storage.** Do not bake in resampling, intensity normalization, or augmentation at conversion time. Downstream consumers (encoder, evaluator) apply their own preprocessing so the inverse is well-defined.
- **Compression**: gzip level 4 on `images`, `segmentations`. Chunk `(1, M, H, W, D)` so streaming a single scan is one read.
- **Dtypes**: `images` float32 (preserves source intensity), `segmentations` int8 (labels in `[0, 127]`).
- **Native shape preferred**. BraTS cohorts arrive at `(240, 240, 155)`; store at native. Only resize at the converter level if the source itself has heterogeneous shapes.

## Adding a new cohort

Subclass `menflow.data.h5_converter.H5Converter`. Implement the abstract properties (`dataset_name`, `modalities`, `label_map`, `is_longitudinal`, `spatial_shape`, `spacing_mm`, `orientation`) and the abstract methods (`discover_scans`, `load_scan`). Optionally override `feature_registry()` and `compute_features(records, h5_file)` to attach cheap, cohort-specific per-scan features (the writer reopens the just-written H5 in `r+` and gives the converter the records list plus the open file so features can be computed from `/segmentations`, `/images`, or `/metadata/`). The base class handles the H5 writing, CSR layout, validation, and — when a registry is declared — feature writing + validation. Add an integration test that runs the converter end-to-end on a 2-3-subject synthetic fixture.

Reference: `src/menflow/data/conversors/brats_men.py` (cross-sectional), `src/menflow/data/conversors/mengrowth.py` (longitudinal).

## Provisional schemas

Some artifacts (e.g. MAISI-v2 latents) have their own provisional H5 schema; those are documented in the module that defines them (e.g. `src/menflow/maisi_autoencoder/latents_h5.py`). They are *not* the unified schema and consumers must not assume cross-compatibility — only files validated by `assert_h5_valid` follow this contract.
