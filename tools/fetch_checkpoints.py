"""Download pinned public checkpoints with SHA-256 verification and resume.

Existing destination files are verified and never replaced. Interrupted
downloads remain in a .part file; a matching download resumes with HTTP Range.
Publication uses an exclusive hard link, so a concurrent file cannot be
overwritten. Checkpoints are not uploaded or added to Git by this tool.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request


@dataclass(frozen=True)
class CheckpointFile:
    repository: str
    revision: str
    source: str
    target: str
    size: int
    sha256: str

    @property
    def url(self):
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{self.source}"


AUDIO_REVISION = "a5f2404e63c84e92f5238908e1706316324ebafa"
IMAGE_REVISION = "db70091623513f616c6da7a049773ec5934be88e"
FILES = {
    "audio": (
        CheckpointFile("zai-org/glm-4-voice-tokenizer", AUDIO_REVISION, "config.json", "vqwhisper/config.json", 1746,
                       "5a5181fcc293ced0a1e7455f6e3503ce4e8be9229922ea34aeb785f62720c0aa"),
        CheckpointFile("zai-org/glm-4-voice-tokenizer", AUDIO_REVISION, "preprocessor_config.json", "vqwhisper/preprocessor_config.json", 340,
                       "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
        CheckpointFile("zai-org/glm-4-voice-tokenizer", AUDIO_REVISION, "model.safetensors", "vqwhisper/model.safetensors", 1458374480,
                       "2800bd503f52b51e45f0c53cfd5c31dcfe8ef7f13d22b396aa3d53e0280dd1e4"),
    ),
    "image": (
        CheckpointFile("X-Omni/X-Omni-En", IMAGE_REVISION, "vit/vit_g.pth", "vqsiglip/vit_g.pth", 2376032176,
                       "250be9ba1a52a5b1365cfd79276cbf022df63577eed5d9cc234f8603d06ef626"),
        CheckpointFile("X-Omni/X-Omni-En", IMAGE_REVISION, "vit/siglip_vq.pt", "vqsiglip/siglip_vq.pt", 249612138,
                       "225aa2d41094ee6c83d2d184d2b6cc9cdadb4f72c1a94501f916cffd42d3b567"),
    ),
}
BLOCK_BYTES = 1024 * 1024


def file_digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    return digest


def verify(path: Path, specification: CheckpointFile):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular checkpoint file: {path}")
    if path.stat().st_size != specification.size or file_digest(path).hexdigest() != specification.sha256:
        raise ValueError(f"Existing file does not match the pinned checkpoint and will not be overwritten: {path}")


def fetch_file(specification: CheckpointFile, output: Path, *, opener=None, timeout=60):
    relative = PurePosixPath(specification.target)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in specification.target or ":" in specification.target:
        raise ValueError("Checkpoint target must stay inside the output directory")
    output = output.resolve()
    destination = output.joinpath(*relative.parts)
    if not destination.parent.resolve().is_relative_to(output):
        raise ValueError("Checkpoint parent resolves outside the output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    lock = destination.with_name(destination.name + ".lock")
    for path in (destination, partial, lock):
        if path.is_symlink():
            raise ValueError(f"Refusing a symbolic-link download path: {path}")
    if destination.exists():
        verify(destination, specification)
        return {"file": str(destination), "status": "verified", "sha256": specification.sha256}
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Download is locked; check for another process before removing {lock}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        if destination.exists():
            verify(destination, specification)
            return {"file": str(destination), "status": "verified", "sha256": specification.sha256}
        position = partial.stat().st_size if partial.exists() else 0
        if position > specification.size:
            raise ValueError(f"Partial checkpoint exceeds the pinned size; inspect and remove {partial}")
        digest = file_digest(partial) if position else hashlib.sha256()
        resumed = position
        if position < specification.size:
            headers = {"User-Agent": "omni-tokenizer-checkpoint-fetch/1", "Accept-Encoding": "identity"}
            if position:
                headers["Range"] = f"bytes={position}-"
            request = urllib.request.Request(specification.url, headers=headers)
            open_url = opener or urllib.request.urlopen
            with open_url(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != position or int(match[3]) != specification.size or not position <= int(match[2]) < specification.size:
                        raise ValueError("Server returned an invalid checkpoint byte range")
                elif status == 200:
                    if position:
                        position = 0
                        digest = hashlib.sha256()
                else:
                    raise ValueError(f"Unexpected download HTTP status {status}")
                with partial.open("ab" if position else "wb") as stream:
                    while block := response.read(BLOCK_BYTES):
                        if position + len(block) > specification.size:
                            raise ValueError("Download exceeds the pinned checkpoint size")
                        stream.write(block)
                        digest.update(block)
                        position += len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
        if position != specification.size or digest.hexdigest() != specification.sha256:
            raise ValueError(f"Checkpoint size or SHA-256 mismatch; unverified data remains at {partial}")
        try:
            os.link(partial, destination)
        except FileExistsError:
            verify(destination, specification)
        partial.unlink()
        return {"file": str(destination), "status": "downloaded", "resumed_bytes": resumed,
                "bytes": position, "sha256": specification.sha256, "revision": specification.revision}
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("audio", "image", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--list", action="store_true", help="Print pinned file metadata without downloading")
    args = parser.parse_args()
    selected = ("audio", "image") if args.model == "all" else (args.model,)
    try:
        for kind in selected:
            for specification in FILES[kind]:
                result = specification.__dict__ if args.list else fetch_file(specification, args.output)
                print(json.dumps(result), flush=True)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
