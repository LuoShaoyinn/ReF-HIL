# Website validation

Local validation completed on 2026-09-28:

- Chromium desktop (1440 px) and mobile (390 px) rendering inspected.
- All 21 videos fully decoded with FFmpeg and played in Chromium, including
  the continuous 177.83-second 1920×1080 overview.
- No JavaScript errors, failed local asset requests, or external asset requests.
- No horizontal page overflow at the tested mobile width.
- Each task has an edited full-sequence training timelapse (preparation excluded)
  and three separate recovery scenes. The wall is only a decorative background.
- Switching tabs pauses every video in the hidden panel, including recovery clips.
- Task tabs work by pointer and keyboard (arrow keys, Home, End).
- All figure images load and decode; paper figures are direct PDF renders.
- The materialized artifact works below a URL subpath, matching project Pages.
- Without JavaScript, all five task panels remain visible.
- All 21 videos use AV1 and are ordinary Git blobs under `assets/videos/`.
- Static export validates 65 links, all video hashes, and directly tracked media.
- The Code button links to the anonymous mirror of `main`, as requested.
  Author names remain omitted from the page.

The page is adapted from the existing template, with local CSS, scripts, and
media. Browser screenshots cover desktop and mobile layouts; no physical robot
or new training experiment is involved in this validation.

## Hosted status at inspection time

The existing origin serves the old blank index. The anonymous root
`/w/ReF-HIL-7762/` returned HTTP 400 (`folder_not_supported`), while
`/w/ReF-HIL-7762/index.html` returned HTTP 200 with the old blank page. The
explicit index route therefore reaches the current forwarded origin.

The new site and CI workflow have been validated locally. A deployment and a
subsequent check through the anonymous proxy are still needed to establish
hosted playback, including video range requests and anonymous forwarding.
This record does not claim that the new artifact is already deployed.
