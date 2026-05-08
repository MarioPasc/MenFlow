---
name: test
description: Run pytest tests for the MenGrowth project
disable-model-invocation: true
---

Run the following tests and report results:

`~/.conda/envs/growth/bin/python -m pytest $ARGUMENTS -v --tb=short`

If `$ARGUMENTS` is empty, run the safe default (excludes slow and real_data tests):

`~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short`

Marker shortcuts: `phase0`, `phase1`, `phase2`, `evaluation`, `experiment`, `unit`, `slow`, `real_data`.
Combine with: `-m "phase1 and unit"`, `-m "not slow"`.
