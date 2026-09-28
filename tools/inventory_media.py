"""Inventory local publication videos without copying or modifying them."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return {
        "bytes": path.stat().st_size,
        "duration_seconds": float(data.get("format", {}).get("duration", 0)),
        "streams": [
            {k: s[k] for k in ("codec_type", "codec_name", "width", "height", "pix_fmt", "avg_frame_rate", "sample_rate", "channels") if k in s}
            for s in data["streams"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.source.rglob("*.mp4")):
        row = {"path": path.relative_to(args.source).as_posix()}
        try:
            row.update(probe(path))
        except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": args.source.name, "videos": rows}, indent=2) + "\n")
    failed = sum("error" in row for row in rows)
    print(f"Inventoried {len(rows)} videos; {failed} probe failures -> {args.output}")
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
