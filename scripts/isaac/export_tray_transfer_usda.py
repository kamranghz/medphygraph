#!/usr/bin/env python3
"""Export tray prep visualization USDA (cabinet → beside monitor cart)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph.i4h_visuals import resolve_i4h_root
from medphygraph.tray_transfer_demo import (
    build_tray_transfer_layout,
    build_tray_transfer_timeline,
    include_clinician,
    layout_report,
    resolve_clinician_usd,
    write_tray_transfer_demo_usda,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "transition_demo" / "tray_transfer_demo_i4h.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runs" / "transition_demo" / "tray_transfer_layout.json",
    )
    parser.add_argument(
        "--with-clinician",
        action="store_true",
        help="EXPERIMENTAL: include Male Doctor (no walk/grasp clips; not LinkedIn quality)",
    )
    args = parser.parse_args()

    i4h_root = resolve_i4h_root()
    if i4h_root is None:
        raise SystemExit("I4H_ASSETS_ROOT not found")

    layout = build_tray_transfer_layout()
    timeline = build_tray_transfer_timeline()
    with_clinician = True if args.with_clinician else include_clinician()
    clinician = resolve_clinician_usd() if with_clinician else None
    out = write_tray_transfer_demo_usda(
        args.output,
        i4h_root=i4h_root,
        layout=layout,
        timeline=timeline,
        clinician_usd=clinician,
        with_clinician=with_clinician,
    )

    report = {
        "usda_path": str(out.resolve()),
        "i4h_root": str(i4h_root.resolve()),
        "clinician_usd": str(clinician.resolve()) if clinician else None,
        "with_clinician": bool(with_clinician and clinician),
        "motion": (
            "experimental_clinician_root_slide"
            if with_clinician and clinician
            else "surface_to_surface_tray_keyframes"
        ),
        "limitation": (
            "No CDN walk/grasp clips; clinician is off by default for LinkedIn quality."
        ),
        "timeline": {
            "fps": timeline.fps,
            "end_frame": timeline.end_frame,
            "duration_seconds": timeline.duration_seconds,
            "approach_end": timeline.approach_end,
            "hold_start_end": timeline.hold_start_end,
            "lift_end": timeline.lift_end,
            "travel_end": timeline.travel_end,
            "lower_end": timeline.lower_end,
            "hold_place_end": timeline.hold_place_end,
        },
        "layout": layout_report(layout),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
