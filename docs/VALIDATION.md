# Release validation

Validated during the 2026-09-28 packaging pass:

- All 200 imported and added Python sources parse successfully.
- All 24 retained runtime/tool CLI help commands exit successfully.
- All relative Markdown links resolve.
- `uv lock --check --offline` accepts the renamed project's lockfile.
- 71 focused unit/integration tests pass, covering the action fence, suffix-only
  base gradients, unrecoverable supervision, constrained critics, checkpoint
  round trips, actor storage and episode snapshots, experiment identity, and
  dynamic task loading.
- FFprobe reads all 62 source videos without probe failures.

The test command, run from the release repository with the existing source
environment's Python 3.14.2 / PyTorch 2.12.0+rocm7.2, was:

```bash
python -m unittest \
  learner.test_algorithm learner.test_suffix_base_floor \
  learner.test_unrecoverable learner.test_checkpoint \
  learner.test_constrained_critic actor.test_storage \
  actor.test_episode_snapshots shared.test_experiment_identity tasks.test_loading
```

These tests use CPU tensors, temporary replay, and synthetic/mocked components.
No physical robot was moved, no new training experiment was run, and no fresh
CUDA/ROCm dependency installation was validated. The complete repository test
suite was not run. The source environment emitted a tensor-to-scalar warning
and system-library diagnostics, but the selected tests completed successfully.

Web media preparation additionally decodes every generated video with FFmpeg
and records hashes and stream metadata in the webpage's `assets/media.json`.
Web build and browser validation are recorded on the `webpages` branch.
