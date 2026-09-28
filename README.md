# ReF-HIL anonymous project page

This branch contains the anonymous project page. Research code lives on `main`.
It adapts the modernized Academic Project Page Template; see [template provenance](TEMPLATE.md).

## Local preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`. No Node build, CDN scripts, external fonts,
analytics, or third-party video player are required.

## Edit

- `index.html`: page text, paper links, task tabs, and result table.
- `static/css/refhil.css`: project theme over the retained template styles.
- `static/js/refhil.js`: accessible task tabs and video playback coordination.
- `assets/videos/`: directly Git-tracked AV1 MP4 media; overview retains audio, training and recovery clips are muted.
- `assets/media.json`: source/output hashes and decode verification.
- `assets/paper/ref-hil.pdf`: the supplied anonymous manuscript `main-6.pdf`.
- `assets/results.json` / `results.csv`: manuscript Table I and Figure 1 values.
- `assets/figures/`: direct renderings of manuscript Figures 2–4.

Each task presents (a) the supplied edited full training/evaluation timelapse,
excluding preparation, and (b) three separate recovery scenes. Existing playback
speeds are preserved. Training includes human interaction and is labeled as such.
The wall appears only as a faint decorative hero background. Reported success rates are paper results; uncertainty is
across evaluation batches, not independent training seeds.

## CI Pages and anonymous forwarding

The `pages.yml` workflow checks out ordinary Git assets, verifies media hashes,
exports public files to `_site`, and deploys a Pages artifact. No Git LFS download
is required. For workflow deployment, Repository Settings → Pages → Source
should be **GitHub Actions**. The branch also contains a self-contained static
site for branch-based publishing.

All public assets use relative paths, so the artifact works under a project
subpath and through the existing anonymous forwarding URL:

`https://anonymous.4open.science/w/ReF-HIL-7762/`

The page omits author names, affiliations, and personal repository links. The
Code button points to the anonymous mirror of main:
`https://anonymous.4open.science/r/ReF-HIL-D7C2/`.

To export locally, choose an empty destination:

```bash
python3 tools/build_site.py --output /tmp/ref-hil-site
```

The exporter rejects missing assets, invalid fragment links, unresolved LFS
pointers, and media hash mismatches. It omits `.git`, CI files, and local
documentation. Videos are ordinary Git blobs under `assets/videos/`; reviewers fetch
same-origin videos from the deployed Pages artifact/proxy.

## Licensing

The adapted template/page presentation retains upstream CC BY-SA 4.0 and its
footer attribution. Bulma retains its MIT notice. Research code on `main` is
GPL-3.0-only. The paper and research media are not relicensed by the template;
the overview retains its original attribution for motivation footage.

Authors remain omitted while the manuscript is under review.

Video regeneration is documented in `docs/MEDIA.md` on the main branch.
