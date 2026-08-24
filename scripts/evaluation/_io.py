"""Small JSON/CSV helpers for evaluation artifact writes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  digest.update(path.read_bytes())
  return digest.hexdigest()


def sha256_file(path: Path) -> str:
  return sha256(path)


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    path.write_text("", encoding="utf-8")
    return
  fieldnames = fields or list(rows[0].keys())
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
