"""Build anonymous webpage assets from the supplied video edits.

Requires FFmpeg/ffprobe. Run this from main, with --output pointing to the
webpages worktree. Source videos are never changed. No network upload occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.inventory_media import probe


CLIPS = (
    ("overview", "video-3-1080p.mp4", "Full project video", True),
    ("push-t-training", "result/push_t/full-10s.mp4", "Push-T training timelapse", False),
    ("cap-unscrewing-training", "result/rotate_knob/full-10s.mp4", "Cap unscrewing training timelapse", False),
    ("gear-assembly-training", "result/insert-gear/full-10s.mp4", "Gear assembly training timelapse", False),
    ("plug-insertion-training", "result/plug_power_socket/full-10s.mp4", "Plug insertion training timelapse", False),
    ("cable-routing-training", "result/hang_double_strings/full-10s.mp4", "Dual-branch cable routing training timelapse", False),
    ('push-t-recovery-1', 'result/push_t/run-4_Sub_02.mp4', 'Contact recovery', False),
    ('push-t-recovery-2', 'result/push_t/run-4_Sub_03.mp4', 'Orientation adjustment', False),
    ('push-t-recovery-3', 'result/push_t/run-4_Sub_04.mp4', 'Object relocation', False),
    ('cap-unscrewing-recovery-1', 'result/rotate_knob/disturb1.mp4', 'Recovery example 1', False),
    ('cap-unscrewing-recovery-2', 'result/rotate_knob/Rotate-knov_disturb3.mp4', 'Recovery example 2', False),
    ('cap-unscrewing-recovery-3', 'result/rotate_knob/Rotate-knov_disturb4.mp4', 'Arm deflection', False),
    ('gear-assembly-recovery-1', 'result/insert-gear/disturb1.mp4', 'Gear reorientation', False),
    ('gear-assembly-recovery-2', 'result/insert-gear/disturb2.mp4', 'Arm disturbance', False),
    ('gear-assembly-recovery-3', 'result/insert-gear/disturb3.mp4', 'Contact recovery', False),
    ('plug-insertion-recovery-1', 'result/plug_power_socket/20260919-100126-074221_Sub_02.mp4', 'Recovery example 1', False),
    ('plug-insertion-recovery-2', 'result/plug_power_socket/20260919-100126-074221_Sub_03.mp4', 'Recovery example 2', False),
    ('plug-insertion-recovery-3', 'result/plug_power_socket/20260919-100126-074221_Sub_04.mp4', 'Recovery example 3', False),
    ('cable-routing-recovery-1', 'result/hang_double_strings/disturb1.mp4', 'Cable lift', False),
    ('cable-routing-recovery-2', 'result/hang_double_strings/disturb3.mp4', 'Cable tug', False),
    ('cable-routing-recovery-3', 'result/hang_double_strings/disturb4.mp4', 'Arm disturbance', False),
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    missing = [name for _, name, _, _ in CLIPS if not (args.source / name).is_file()]
    if missing:
        parser.error(f"missing source files: {missing}")
    videos = args.output / "assets/videos"
    posters = args.output / "assets/posters"
    videos.mkdir(parents=True, exist_ok=True)
    posters.mkdir(parents=True, exist_ok=True)
    # The wall is a decorative background, never a featured experiment player.
    background_source = args.source / "video_clip_wall.mp4"
    background = posters / "training-wall.jpg"
    run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", "1",
         "-i", str(background_source), "-frames:v", "1", "-vf", "scale=960:-2",
         "-q:v", "3", "-map_metadata", "-1", str(background)])
    rows = []
    for slug, source_name, caption, audio in CLIPS:
        source = args.source / source_name
        target = videos / f"{slug}.mp4"
        poster = posters / f"{slug}.jpg"
        if target.exists() or poster.exists():
            raise FileExistsError(f"refusing to overwrite existing assets for {slug}")
        print(f"Preparing {slug}", flush=True)
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
            "-map", "0:v:0", "-map_metadata", "-1", "-map_chapters", "-1",
            "-vf", "scale=1920:-2" if audio else "scale=960:-2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "24" if audio else "25",
            "-pix_fmt", "yuv420p", "-threads", str(args.threads),
        ]
        if audio:
            command += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
        else:
            command += ["-an"]
        command += ["-movflags", "+faststart", str(target)]
        run(command)
        run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", "1", "-i", str(target),
             "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", "-map_metadata", "-1", str(poster)])
        # Decode every frame, including audio in the full project video.
        run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"])
        rows.append({
            "id": slug, "caption": caption,
            "source": source_name, "source_sha256": digest(source),
            "video": target.relative_to(args.output).as_posix(),
            "poster": poster.relative_to(args.output).as_posix(),
            "sha256": digest(target), "full_decode_passed": True, **probe(target),
        })
    manifest = args.output / "assets/media.json"
    manifest.write_text(json.dumps({"videos": rows, "background": {
        "source": "video_clip_wall.mp4", "source_sha256": digest(background_source),
        "poster": background.relative_to(args.output).as_posix(),
        "sha256": digest(background), "usage": "decorative hero background only",
    }}, indent=2) + "\n")
    print(f"Prepared and decoded {len(rows)} videos -> {manifest}")


if __name__ == "__main__":
    main()
