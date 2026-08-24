"""Dataset helpers for DyPhyGraph-Health training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class HealthEdgeDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], *, ratio: float = 1.0, max_len: int | None = None):
        self.samples = samples
        self.ratio = float(ratio)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        s = self.samples[idx]
        key = str(self.ratio)
        if "features_partial" in s and key in s["features_partial"]:
            x = np.asarray(s["features_partial"][key], dtype=np.float32)
        else:
            x = np.asarray(s["features_full"], dtype=np.float32)
        if self.max_len is not None and x.shape[0] > self.max_len:
            x = x[: self.max_len]
        y = np.float32(s["label"])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y),
            "scene_id": s["scene_id"],
            "subject_id": s["subject_id"],
            "host_id": s["host_id"],
        }


def collate_pad(batch: list[dict]) -> dict[str, Any]:
    xs = [b["x"] for b in batch]
    t_max = max(x.shape[0] for x in xs)
    f = xs[0].shape[1]
    out = torch.zeros(len(xs), t_max, f)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
        # ensure mask zero on pad
        if x.shape[0] < t_max:
            out[i, x.shape[0] :, -1] = 0.0
    return {
        "x": out,
        "y": torch.stack([b["y"] for b in batch]),
        "scene_id": [b["scene_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "host_id": [b["host_id"] for b in batch],
    }


def load_dataset(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def split_samples(dataset: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    buckets = {"train": [], "val": [], "test": []}
    for s in dataset["samples"]:
        part = s.get("split", "train")
        if part not in buckets:
            buckets[part] = []
        buckets[part].append(s)
    return buckets
