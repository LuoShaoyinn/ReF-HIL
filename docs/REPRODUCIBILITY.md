# Paper results and reproduction scope

The supplied manuscript `main-6.pdf` is an anonymous, eight-page manuscript
under review. Its Table I reports the following ReF-HIL results:

| Task | Autonomous success (%) | Penalized length (steps) | Minutes to 90% success |
| --- | --- | --- | --- |
| Push-T | 100.0 ± 0.0 | 69.3 ± 2.9 | 23.3 |
| Cap unscrewing | 100.0 ± 0.0 | 42.4 ± 1.4 | 18.2 |
| Gear assembly | 95.0 ± 5.0 | 40.1 ± 5.6 | 25.9 |
| Plug insertion | 91.7 ± 2.9 | 64.4 ± 5.0 | 40.7 |
| Dual-branch cable routing | 95.0 ± 5.0 | 78.3 ± 7.0 | 62.3 |

Success and length values are means ± sample standard deviations across
evaluation batches, **not independent training runs**. Length uses actual steps
for autonomous successes and the full task horizon for every other episode.
Times are Figure 1's active-training times to first reach 90% autonomous
success. Training plots use 20-episode moving averages.

The full Table I, including HG-DAgger, HIL-SERL, and SiLRI, is transcribed in
[CSV](paper-results.csv) and [JSON](paper-results.json). These are paper-table
transcriptions, not regenerated measurements.

## What is included

- The current source runtime and all ten task profiles, with existing tests.
- Learner objectives matching the manuscript's value shaping and action fence.
- Tools for classifier collection/training, replay collection, validation,
  and plotting new runs.
- Source hashes, manuscript hash, and the original `final-paper/*` tag commits.
- Dependency lockfile and the validation record for this packaging pass.

## What is not bundled

Raw demonstrations, trained classifiers, policy/critic checkpoints, TensorBoard
event logs, and the raw per-trial records behind the paper figures are not
included. The original source's baseline and ablation branches remain in `srt`;
this `main` release exposes one method and does not claim to bundle their
implementations. See [source refs](source-refs.json) for exact local tag commits.

The source's `final-paper/main` tag predates later camera, crop, workspace,
reset, and task-name changes. Comparing that tag with the packaged source shows
the learner implementation unchanged except for the CLI's default task name.
The latest task calibration is retained for practical use, but it must not be
treated as proof that every historical experiment used those exact settings.

Full experimental reproduction therefore requires the matching robot setup,
scene calibration, human demonstrations, trained success models, and recorded
run configuration. Software tests exercise the implementation; they do not
reproduce the paper's physical results.

## Mechanism findings

The paper's ablations show task-dependent effects: removing value shaping
reduces success on every evaluated task; removing the fence improves gear
assembly while reducing success on the other four tasks. The manuscript also
compares critic values on one selected cable-routing trajectory and compares
in-fence imitation in individual training runs. These analyses support the
mechanism discussion but are not multi-seed causal estimates.
