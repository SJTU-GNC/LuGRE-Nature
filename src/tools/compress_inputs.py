from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
        digest.update(block)
    return digest.hexdigest().upper()


def compress_one(relative_path: str) -> None:
    source = (ROOT / relative_path).resolve()
    if ROOT not in source.parents or not source.is_file():
        raise FileNotFoundError(f"Missing or unsafe source: {source}")
    if source.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(f"Expected a CSV or text input: {source}")

    target = source.with_name(source.name + ".gz")
    temporary = target.with_name(target.name + ".tmp")
    with source.open("rb") as source_handle:
        source_hash = sha256_stream(source_handle)
        source_handle.seek(0)
        with temporary.open("wb") as compressed_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=compressed_handle,
                mtime=0,
            ) as gzip_handle:
                shutil.copyfileobj(source_handle, gzip_handle, CHUNK_SIZE)

    with gzip.open(temporary, "rb") as decompressed_handle:
        restored_hash = sha256_stream(decompressed_handle)
    if restored_hash != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Lossless verification failed: {source}")

    os.replace(temporary, target)
    print(
        f"{source.relative_to(ROOT).as_posix()}: "
        f"{source.stat().st_size:,} -> {target.stat().st_size:,} bytes; "
        f"content SHA256={source_hash}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, losslessly verified gzip inputs."
    )
    parser.add_argument("paths", nargs="+", help="Package-relative input paths.")
    args = parser.parse_args()
    for relative_path in args.paths:
        compress_one(relative_path)


if __name__ == "__main__":
    main()
