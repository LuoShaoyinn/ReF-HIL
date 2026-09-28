"""Package the committed main-branch code without Git history or zip comments."""

import argparse
import io
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("assets/code/ref-hil.zip"))
    args = parser.parse_args()
    branch = subprocess.check_output(
        ["git", "-C", str(args.source), "branch", "--show-current"], text=True
    ).strip()
    if branch != "main":
        raise ValueError("code archive must come from the main checkout")
    archive = subprocess.check_output(
        ["git", "-C", str(args.source), "archive", "--format=zip", "--prefix=ref-hil/", "HEAD"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Repack entries to omit Git's commit hash comment and extended metadata.
    with zipfile.ZipFile(io.BytesIO(archive)) as source, zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for item in source.infolist():
            if item.is_dir():
                continue
            clean = zipfile.ZipInfo(item.filename, date_time=(2026, 1, 1, 0, 0, 0))
            clean.compress_type = zipfile.ZIP_DEFLATED
            clean.external_attr = 0o100644 << 16
            target.writestr(clean, source.read(item))
    print(f"Packaged anonymous source download -> {args.output}")


if __name__ == "__main__":
    main()
