# Routines Pattern

A *routine* is a runnable task (encode a cohort, decode latents, train a flow predictor, run E1, …) that wraps a configurable engine. All routines live under `routines/<name>/` and follow this layout exactly:

```
routines/<name>/
├── __init__.py          # empty
├── cli.py               # entrypoint: `python -m routines.<name>.cli <yaml>`
├── configs/             # one yaml per concrete invocation (default + smoke + per-cohort)
│   └── default.yaml
├── slurm/               # Picasso submission scripts (.sh launcher + worker)
│   └── .gitkeep         # keep the directory even when empty
└── engine/
    ├── __init__.py      # re-exports `<Name>Engine` and `<Name>RoutineConfig`
    └── <name>_engine.py # implementation
```

## Invariants

1. **`cli.py` takes one positional argument**: the path to a YAML config. No other flags. Logging level is read from the YAML, not the command line.
2. **The engine module exports two public symbols**: a frozen-dataclass `<Name>RoutineConfig` (with a `from_yaml(path)` classmethod) and an `<Name>Engine` class with a single `run() -> Path` method that returns the produced artifact path.
3. **The engine is dataset-agnostic where possible**. Any cohort-specific behaviour (file naming, label set, modality order) lives in the input H5's attrs (see `h5-format.md`); the engine reads attrs and adapts.
4. **Configs are reproducible**. Persist every parameter that influenced the output into the produced artifact's attrs (model architecture JSON, inference config JSON, checkpoint path, ISO-8601 timestamp). The engine should never silently use values absent from the YAML.
5. **Validate on close**. If the routine produces an H5, the engine must call `assert_h5_valid(...)` from `menflow.data.h5_schema` before exiting `run()`. A non-conformant artifact must not reach disk in a "successful" state.
6. **Console scripts**. Register each routine in `pyproject.toml` `[project.scripts]` as `menflow-<name> = "routines.<name>.cli:main"` so it works as both `menflow-<name> cfg.yaml` and `python -m routines.<name>.cli cfg.yaml`.
7. **`configs/` is a directory, not a single file**. Keep the canonical `default.yaml` plus task-specific variants (e.g. `smoke_test_3060.yaml`, `picasso_a100_full.yaml`). One YAML per executed configuration; never mutate `default.yaml` for one-off runs.
8. **`slurm/` matches the `picasso-sbatch` skill conventions**. Pair files: `launcher_<name>.sh` (sbatch wrapper) + `worker_<name>.sh` (per-array-task body). Singularity-only — no Docker, no `module load python`. Use `pip install --no-deps -e /opt/MenFlow` inside the .sif and rely on the NGC base image for torch.
9. **No top-level imports of the engine in `cli.py`** that trigger heavy work (no global model loads, no `cuda` calls at import time). All side effects belong inside `Engine.run()`.
10. **One routine, one responsibility**. If a routine grows a second mode (e.g. encode-and-decode-in-one-shot), split it into two routines that share an engine helper, not a multi-mode flag.

## Reference implementations

- `routines/encode/` — encode a unified MenFlow H5 with MAISI-v2; produces a provisional latent H5.
- `routines/decode/` — decode latents back into a unified MenFlow H5; validates against the schema on close.

## Testing

Each engine ships a unit test in `tests/<routine>/test_<name>_engine.py` that exercises a synthetic config end-to-end without GPU when possible (mock the model). GPU-only smoke tests live in `tests/<routine>/test_<name>_smoke.py` and are marked `@pytest.mark.slow`.
