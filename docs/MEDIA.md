# Video review and webpage assets

The supplied directory contains **62 MP4 videos, totaling 21.87 GB**. Every file
passes FFprobe metadata inspection; [media-inventory.json](media-inventory.json)
records relative filenames, sizes, durations, streams, and resolutions. This
inventory is not a claim that every frame of every raw recording was watched.

## Selected edits

| Source | Duration | Use |
| --- | --- | --- |
| `video-3-1080p.mp4` | 177.83 s | 1080p HEVC master for the overview |
| `final-av1.mp4` | 177.83 s | Compact 1080p AV1 overview, 19.88 MB |
| `video_clip_wall.mp4` | 10.01 s | Decorative hero background only (poster) |
| `result/push_t/full-10s.mp4` | 10.02 s | Push-T training timelapse |
| `result/rotate_knob/full-10s.mp4` | 10.75 s | Cap unscrewing training timelapse |
| `result/insert-gear/full-10s.mp4` | 10.05 s | Gear assembly training timelapse |
| `result/plug_power_socket/full-10s.mp4` | 10.00 s | Plug insertion training timelapse |
| `result/hang_double_strings/full-10s.mp4` | 10.00 s | Cable-routing training timelapse |

Sampled openings and training/evaluation frames were inspected for all five
`full-10s.mp4` exports. These are the existing edited full-sequence timelapses,
with preparation excluded and training/evaluation speed labels retained. Raw
`src/full-run.mp4` recordings are not used as task players. Four separate
recovery edits are included per task (20 total), with their original playback
speed preserved. Their exact filenames are listed in `tools/prepare_web_media.py`
and the generated `.github/media.json` manifest. Sampled recovery contact sheets were inspected. The overview includes task training, autonomous
disturbance cases, and comparison results. Its motivation slide includes
attributed clips from other work. Those source clips remain separate and are
not presented as ReF-HIL experiments. The training edits contain
human interaction; the website must not label them as fully autonomous trials.

The video directory also contains 19 Shotcut projects with 84 machine-specific
resource references. Their paths point to the old workspace layout, so the
editable projects are not portable as supplied. The webpage uses rendered
assets rather than copying broken project files or the 20 GB raw footage.

## Rebuild webpage media

With a separate `webpages` checkout and FFmpeg available:

```bash
python -m tools.inventory_media ../srt-video-clip --output docs/media-inventory.json
python -m tools.prepare_web_media ../srt-video-clip --output ../ReF-HIL-webpages
```

The preparation tool produces a silent 1080p AV1 overview from the master,
960px-wide muted AV1 training and recovery clips, JPEG posters, and a manifest containing
source/output SHA-256 hashes. It strips container metadata, enables MP4
fast-start, and fully decodes every generated video. Existing output videos
are not overwritten. Playback acceleration already embedded in the source
edits is preserved.

The webpage branch tracks AV1 MP4 files directly in Git under `assets/videos/`.
It does not use Git LFS. FFmpeg uses the software SVT-AV1 encoder (preset 8,
CRF 32), preserves source timing, and enables MP4 fast-start. The static export
verifies hashes and rejects accidental LFS pointer files.

All page assets use relative paths. Author metadata and personal repository
links are omitted for anonymous review; no external analytics or embedded
third-party video players are required.
