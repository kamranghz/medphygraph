#!/usr/bin/env python3
"""Download paper-frozen MedPhyGraph artifacts from Hugging Face.

Writes into ``data/`` and ``checkpoints/`` only (see medphygraph.paths).
HF repo IDs are unchanged. Idempotent: skips existing identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medphygraph.paths import (  # noqa: E402
    checkpoint_seed0,
    ckpt_root,
    data_root,
    dataset_hard_dir,
    expanded_transfer_root,
    multiseed_checkpoints_dir,
    procedural_scenes_root,
)

MODEL_REPO = "MedPhyGraph/CF-SupportNet"
DATA_REPO = "MedPhyGraph/support-graph-data"

SEED0_NAME = "health_dyphygraph_r1.0_seed0.pt"
SEED0_SHA256 = "e0b34529745399ecc5da5341ed7a162173611e12c8bd50dec121b0c575c5b789"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def place_file(src: Path, dst: Path, *, force: bool) -> str:
    if not src.is_file():
        raise FileNotFoundError(f"missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(src) == sha256_file(dst):
            return "skip_identical"
        if not force:
            return "skip_conflict"
    shutil.copy2(src, dst)
    return "copied"


def copy_tree_contents(src_dir: Path, dst_dir: Path, *, force: bool) -> dict[str, int]:
    counts = {"copied": 0, "skip_identical": 0, "skip_conflict": 0}
    if not src_dir.is_dir():
        raise FileNotFoundError(f"missing source directory: {src_dir}")
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        action = place_file(src, dst_dir / rel, force=force)
        counts[action] += 1
    return counts


def download_repo(repo_id: str, cache_dir: Path, *, repo_type: str = "model") -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=cache_dir,
        )
    )


def verify_seed0(path: Path) -> bool:
    if not path.is_file():
        print(f"verify: missing {path}")
        return False
    actual = sha256_file(path)
    if actual != SEED0_SHA256:
        print(f"verify: seed0 SHA256 mismatch (expected {SEED0_SHA256}, got {actual})")
        return False
    print(f"verify: seed0 SHA256 OK ({path})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MedPhyGraph HF artifacts into data/ and checkpoints/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files even when hashes differ")
    parser.add_argument("--verify", action="store_true", help="Verify seed-0 checkpoint SHA256 after placement")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".hf_download_cache",
        help="Local HF snapshot cache directory (default: .hf_download_cache)",
    )
    args = parser.parse_args()

    cache_root = args.cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    data_root().mkdir(parents=True, exist_ok=True)
    ckpt_root().mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str, str]] = []

    try:
        model_root = download_repo(MODEL_REPO, cache_root / "CF-SupportNet", repo_type="model")
        hf_data = download_repo(DATA_REPO, cache_root / "support-graph-data", repo_type="dataset")
    except Exception as exc:
        print(f"download failed: {exc}")
        return 1

    seed0_dst = checkpoint_seed0()
    seed0_action = place_file(model_root / SEED0_NAME, seed0_dst, force=args.force)
    summary.append((SEED0_NAME, str(seed0_dst.relative_to(REPO_ROOT)), seed0_action))

    multi_dir = multiseed_checkpoints_dir()
    for seed in range(1, 5):
        name = f"health_dyphygraph_r1.0_seed{seed}.pt"
        dst = multi_dir / name
        action = place_file(model_root / name, dst, force=args.force)
        summary.append((name, str(dst.relative_to(REPO_ROOT)), action))

    dataset_counts = copy_tree_contents(hf_data / "training_data", dataset_hard_dir(), force=args.force)
    procedural_counts = copy_tree_contents(
        hf_data / "procedural_scenes", procedural_scenes_root(), force=args.force
    )
    expanded_counts = copy_tree_contents(
        hf_data / "expanded_transfer_procedural_v1",
        expanded_transfer_root(),
        force=args.force,
    )

    print("\nPlacement summary:")
    print(f"{'artifact':<42} {'destination':<58} action")
    print("-" * 110)
    for artifact, dest, action in summary:
        print(f"{artifact:<42} {dest:<58} {action}")

    print(
        f"\ntraining_data -> data/training: "
        f"copied={dataset_counts['copied']} identical={dataset_counts['skip_identical']} "
        f"conflicts={dataset_counts['skip_conflict']}"
    )
    print(
        f"procedural_scenes -> data/procedural_scenes: "
        f"copied={procedural_counts['copied']} identical={procedural_counts['skip_identical']} "
        f"conflicts={procedural_counts['skip_conflict']}"
    )
    print(
        f"expanded_transfer_procedural_v1 -> data/expanded_transfer: "
        f"copied={expanded_counts['copied']} identical={expanded_counts['skip_identical']} "
        f"conflicts={expanded_counts['skip_conflict']}"
    )

    conflicts = (
        sum(1 for _, _, action in summary if action == "skip_conflict")
        + dataset_counts["skip_conflict"]
        + procedural_counts["skip_conflict"]
        + expanded_counts["skip_conflict"]
    )
    if conflicts and not args.force:
        print(f"\n{conflicts} file(s) skipped due to hash conflicts. Re-run with --force to overwrite.")
        return 2

    if args.verify:
        if not verify_seed0(seed0_dst):
            return 3

    print("\nDownload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
