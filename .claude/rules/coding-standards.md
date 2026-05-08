# Coding Standards

1. **Type hints** on ALL function signatures and return types.
2. **Google-style docstrings** on all public functions and classes.
3. **Brief inline comments** on non-obvious code only.
4. **Logging** via Python `logging` module with `rich` handler. INFO for training events, DEBUG for shapes/values.
5. **No magic numbers** — all hyperparams from YAML configs via OmegaConf.
6. **Prefer library functions**: MONAI transforms, `einops.rearrange`, `F.scaled_dot_product_attention`.
7. **Tests use pytest**: `~/.conda/envs/menflow/bin/python -m pytest tests/ -v`
8. **Keep functions atomic** — one conceptual task per function.
