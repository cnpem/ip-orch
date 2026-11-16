"""Serialize CSV updates across local processes using an adjacent lock file."""

import csv
from pathlib import Path

from filelock import FileLock


def append_csv(path, row):
    """Append one record with a single header; reject incompatible schemas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=60):
        nonempty = path.exists() and path.stat().st_size > 0
        if nonempty:
            with path.open(newline="", encoding="utf-8") as handle:
                if next(csv.reader(handle)) != list(row):
                    raise ValueError(f"CSV schema mismatch: {path}")
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not nonempty:
                writer.writeheader()
            writer.writerow(row)
