# E2.1 — Volume Recoverability

Linear (Ridge) and MLP probes on log V from mask-anchored MAISI-v2 latents.
See `docs/E2/E2_1_recoverability.md` for the protocol.

Run:

```bash
~/.conda/envs/menflow/bin/python -m experiments.E2.E2_1_recoverability.cli \
  experiments/E2_1_recoverability/configs/local_brats_men.yaml
```

Outputs land at `output_dir`, including `result.json`, `direction.npz`, and
PNG figures.
