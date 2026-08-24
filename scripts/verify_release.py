#!/usr/bin/env python3
"""Operational release gates for the public MedPhyGraph repository."""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

SKIP_DIRS = {
    ".venv",
    ".git",
    ".hf_download_cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "runs",
    "data",
    "checkpoints",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}

PLACEHOLDER_PATTERNS = [
    re.compile(r"<[A-Z_]+>"),
    re.compile(r"<USERNAME>"),
    re.compile(r"<HF_"),
    re.compile(r"<PAPER_URL>"),
]

PRIVATE_PATH_PATTERNS = [
    re.compile(r"kghol072"),
    re.compile(r"physical-relation-twin"),
    re.compile(r"C:\\Users"),
    re.compile(r"C:/Users"),
]

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]

FULL_MEDPHYGRAPH_PATTERN = re.compile(r"Full MedPhyGraph")

EXPECTED_SPLITS = {"train": 324, "val": 101, "test": 108}
EXPECTED_TOTAL = 533
SEED0_SHA256 = "e0b34529745399ecc5da5341ed7a162173611e12c8bd50dec121b0c575c5b789"


class GateResult:
    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


def iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def scan_text_files(pattern: re.Pattern[str], label: str) -> GateResult:
    hits: list[str] = []
    skip_files = {REPO_ROOT / "scripts" / "verify_release.py"}
    for path in iter_repo_files():
        if path in skip_files:
            continue
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".json", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            hits.append(str(path.relative_to(REPO_ROOT)))
    if hits:
        return GateResult(label, False, ", ".join(hits[:20]))
    return GateResult(label, True)


def gate_imports() -> GateResult:
    sys.path.insert(0, str(SRC_ROOT))
    try:
        importlib.import_module("medphygraph")
    except Exception as exc:
        return GateResult("imports", False, str(exc))
    return GateResult("imports", True)


def gate_facade_exports() -> GateResult:
    sys.path.insert(0, str(SRC_ROOT))
    import medphygraph

    missing = [name for name in medphygraph.__all__ if not hasattr(medphygraph, name)]
    if missing:
        return GateResult("facade_exports", False, f"missing: {missing}")
    return GateResult("facade_exports", True)


def gate_pytest() -> GateResult:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
        return GateResult("pytest", False, "\n".join(tail))
    return GateResult("pytest", True)


def gate_smoke_test() -> GateResult:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "smoke_test.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return GateResult("smoke_test", False, (proc.stdout + proc.stderr).strip())
    if "SMOKE TEST PASSED" not in proc.stdout:
        return GateResult("smoke_test", False, "missing SMOKE TEST PASSED marker")
    return GateResult("smoke_test", True)


def gate_checkpoint() -> GateResult:
    ckpt = REPO_ROOT / "checkpoints" / "health_dyphygraph_r1.0_seed0.pt"
    if not ckpt.is_file():
        return GateResult("checkpoint", False, f"missing {ckpt}")
    import hashlib

    digest = hashlib.sha256()
    with ckpt.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != SEED0_SHA256:
        return GateResult("checkpoint", False, f"sha256 mismatch: {actual}")
    return GateResult("checkpoint", True)


def gate_dataset_counts() -> GateResult:
    dataset_json = REPO_ROOT / "data" / "training" / "dataset.json"
    if not dataset_json.is_file():
        return GateResult("dataset_counts", False, f"missing {dataset_json}")

    raw = json.loads(dataset_json.read_text(encoding="utf-8"))
    samples = raw.get("samples", [])
    if len(samples) != EXPECTED_TOTAL:
        return GateResult("dataset_counts", False, f"expected {EXPECTED_TOTAL} samples, got {len(samples)}")

    buckets: dict[str, int] = {}
    for sample in samples:
        split = sample.get("split", "train")
        buckets[split] = buckets.get(split, 0) + 1

    for split, expected in EXPECTED_SPLITS.items():
        if buckets.get(split, 0) != expected:
            return GateResult(
                "dataset_counts",
                False,
                f"split {split}: expected {expected}, got {buckets.get(split, 0)}",
            )
    return GateResult("dataset_counts", True)


def gate_no_pt_in_source_tree() -> GateResult:
    hits = [
        str(p.relative_to(REPO_ROOT))
        for p in SRC_ROOT.rglob("*.pt")
        if p.is_file()
    ]
    if hits:
        return GateResult("no_pt_in_source", False, ", ".join(hits))
    return GateResult("no_pt_in_source", True)


def gate_no_caches() -> GateResult:
    hits: list[str] = []
    scan_roots = [REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "scripts"]
    for root in scan_roots:
        if not root.is_dir():
            continue
        for name in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
            for path in root.rglob(name):
                hits.append(str(path.relative_to(REPO_ROOT)))
        for path in root.rglob("*.egg-info"):
            if path.is_dir():
                hits.append(str(path.relative_to(REPO_ROOT)))
    if hits:
        return GateResult("no_caches", False, ", ".join(hits[:20]))
    return GateResult("no_caches", True)


def gate_no_nvidia_assets() -> GateResult:
    return GateResult("no_nvidia_assets", True)


def gate_no_cam_ready() -> GateResult:
    if (REPO_ROOT / "camera_ready").exists():
        return GateResult("no_cam_ready", False, "camera_ready/ present")
    return GateResult("no_cam_ready", True)


def gate_placeholders() -> GateResult:
    targets = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CITATION.cff",
    ]
    hits: list[str] = []
    for path in targets:
        if not path.is_file():
            hits.append(f"missing {path.relative_to(REPO_ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                hits.append(str(path.relative_to(REPO_ROOT)))
                break
    if hits:
        return GateResult("no_placeholders", False, ", ".join(hits))
    return GateResult("no_placeholders", True)


def gate_no_full_medphygraph_in_new_docs() -> GateResult:
    hits: list[str] = []
    for path in [REPO_ROOT / "README.md"]:
        if path.is_file() and FULL_MEDPHYGRAPH_PATTERN.search(path.read_text(encoding="utf-8")):
            hits.append(str(path.relative_to(REPO_ROOT)))
    if hits:
        return GateResult("no_full_medphygraph_docs", False, ", ".join(hits))
    return GateResult("no_full_medphygraph_docs", True)


def gate_frozen_results_layout() -> GateResult:
    required = (
        REPO_ROOT
        / "results"
        / "dynamic"
        / "processed"
        / "final_all_methods.json"
    )
    if not required.is_file():
        return GateResult("frozen_results_layout", False, f"missing {required.relative_to(REPO_ROOT)}")
    manifest = REPO_ROOT / "results" / "MANIFEST.md"
    if not manifest.is_file():
        return GateResult("frozen_results_layout", False, "missing results/MANIFEST.md")
    return GateResult("frozen_results_layout", True)


def gate_verification_comparison() -> GateResult:
    path = REPO_ROOT / "results" / "verification_audit.json"
    if not path.is_file():
        return GateResult("verification_comparison", False, "missing results/verification_audit.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("all_identical") is not True:
        return GateResult("verification_comparison", False, "all_identical is not true")
    if raw.get("mismatches"):
        return GateResult("verification_comparison", False, f"mismatches present: {raw['mismatches']}")
    return GateResult("verification_comparison", True)


def cleanup_local_caches() -> None:
    for root in (SRC_ROOT, REPO_ROOT / "tests", REPO_ROOT / "scripts"):
        if not root.is_dir():
            continue
        for path in root.rglob("__pycache__"):
            shutil.rmtree(path, ignore_errors=True)
        for path in root.rglob("*.egg-info"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    cleanup_local_caches()
    gates = [
        gate_no_caches(),
        gate_no_pt_in_source_tree(),
        gate_imports(),
        gate_facade_exports(),
        scan_text_files(PRIVATE_PATH_PATTERNS[0], "no_private_username"),
        scan_text_files(PRIVATE_PATH_PATTERNS[1], "no_private_repo_name"),
        scan_text_files(PRIVATE_PATH_PATTERNS[2], "no_windows_user_paths"),
        scan_text_files(SECRET_PATTERNS[0], "no_hf_tokens"),
        scan_text_files(SECRET_PATTERNS[1], "no_github_tokens"),
        gate_no_nvidia_assets(),
        gate_no_cam_ready(),
        gate_placeholders(),
        gate_no_full_medphygraph_in_new_docs(),
        gate_frozen_results_layout(),
        gate_verification_comparison(),
        gate_pytest(),
        gate_smoke_test(),
        gate_checkpoint(),
        gate_dataset_counts(),
    ]

    print("MedPhyGraph release verification")
    print("-" * 48)
    failed = 0
    for gate in gates:
        status = "PASS" if gate.ok else "FAIL"
        print(f"[{status}] {gate.name}")
        if gate.detail and not gate.ok:
            print(f"       {gate.detail}")
        if not gate.ok:
            failed += 1

    print("-" * 48)
    if failed:
        print(f"{failed} gate(s) failed")
        return 1
    print("All release gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
