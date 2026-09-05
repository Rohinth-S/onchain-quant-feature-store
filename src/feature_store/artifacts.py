"""Write reports safely and preserve the caller's study inputs."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            staged = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)

def require_distinct_paths(**paths: Path | None) -> None:
    known: list[tuple[str, Path]] = []
    for label, path in paths.items():
        if path is None:
            continue
        resolved = path.resolve()
        for other_label, other_path in known:
            same_file = resolved.exists() and other_path.exists() and resolved.samefile(other_path)
            if resolved == other_path or same_file:
                raise ValueError(f"{label} and {other_label} must use distinct paths")
        known.append((label, resolved))
