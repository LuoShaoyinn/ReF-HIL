"""Validate local assets and export only public website files for Pages."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError(f"duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs["id"])
        for name in ("href", "src", "poster"):
            if attrs.get(name):
                self.links.append(attrs[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    page = (ROOT / "index.html").read_text()
    assets = Assets()
    assets.feed(page)
    for link in assets.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        if not parsed.path:
            if parsed.fragment and parsed.fragment not in assets.ids:
                raise ValueError(f"missing section: {link}")
            continue
        path = (ROOT / unquote(parsed.path)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise FileNotFoundError(link)
    videos = list((ROOT / "assets/videos").glob("*.mp4"))
    manifest = json.loads((ROOT / "assets/media.json").read_text())
    for entry in manifest["videos"]:
        path = ROOT / entry["video"]
        with path.open("rb") as handle:
            if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                raise RuntimeError(f"Git LFS pointer found: {path.name}; website assets must be ordinary Git blobs")
            handle.seek(0)
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"media hash mismatch: {path.name}")
    if len(videos) != len(manifest["videos"]):
        raise ValueError("video manifest does not match exported video set")
    output = args.output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output cannot be the source tree or an ancestor")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    for name in ("index.html", ".nojekyll"):
        shutil.copyfile(ROOT / name, output / name)
    for name in ("assets", "static"):
        shutil.copytree(ROOT / name, output / name)
    print(f"Exported {len(videos)} directly tracked AV1 videos and checked {len(assets.links)} links -> {output}")


if __name__ == "__main__":
    main()
