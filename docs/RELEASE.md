# Release organization and provenance

The publication code is a curated snapshot of source commit
`53a30ed350f1572184ab6d63e329edd9cf6b4421` from `srt/main`.
The source repository, paper drafts, and original video directory are preserved.

## Branches

- `main`: real-robot implementation, tests, reusable tools, documentation,
  dependency lockfile, and GPLv3 code license.
- `webpages`: anonymous project page, manuscript, figures, and web media.
  Videos use Git LFS. The website is maintained through a separate worktree.

Authors remain omitted while the paper is under review. The website must not
link to the submitting user's profile or identifiable repository URL. Git commit
identity is separate from public page content, as requested by the author.

## Curation

Runtime Python modules retain their paths, class names, task names, and
serialization contracts. Trailing whitespace is cleaned without changing Python syntax trees.
No learning objective or physical control behavior
was changed during packaging. Existing runtime and task tests are retained.
Project metadata is renamed from `srt` to `ref-hil`; the lockfile is committed.

The release omits `.env`, virtual environments, caches, experiment outputs,
old command scratchpads, obsolete method reports, and run-specific audit/probe
scripts. The legacy offline H trainer is excluded because it assumes an older
checkpoint naming and calibration scheme. Reusable collection, classifier,
evaluation, and calibration tools remain in `tools/`.

The root, architecture, method, and setup documentation are rewritten against
the actual source. The cap-task README's gripper force is corrected to match
its configuration (200). The additional task profiles remain available and
are distinguished from the five video/paper task associations.

`source-manifest.json` records original hashes of imported files;
`source-refs.json` records the original final-paper tags, including baseline
and ablation snapshots. These files establish provenance and do not import
the original repository's history into the publication repository.

## Licensing

Code is licensed under **GPL-3.0-only**, following the requested GPLv3 license.
See the root `LICENSE`. This code license does not relicense the manuscript,
research videos, or third-party template assets; their notices are maintained
separately on `webpages`. No author list, affiliation, or publication venue is
invented for the under-review manuscript.
