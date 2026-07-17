"""Build the publication-ready Upper-Air Data Watch social graphic package.

Run after ``python scripts/run_upper_air_monitor.py --refresh`` for fresh data,
or pass ``--manifest`` to re-render the social layouts without refreshing data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from upper_air_network_monitor.social_graphics import (
    build_social_package,
    load_monitor_inputs,
    render_social_graphics_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Upper-Air Data Watch social graphics.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs" / "upper_air_network_monitor" / "social",
        help="Directory for the nine PNGs and JSON manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Re-render the suite from an existing metrics manifest without reading monitor CSVs.",
    )
    parser.add_argument(
        "--dpi-scale",
        type=float,
        default=1.0,
        help="Optional render scale for testing; leave at 1.0 for production social dimensions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest:
        states = args.repo_root / "comfortwx" / "mapping" / "data" / "us_states.geojson"
        payload, paths = render_social_graphics_from_manifest(
            args.manifest,
            args.outdir,
            states_geojson_path=states,
            dpi_scale=args.dpi_scale,
        )
        print(f"Rendered from manifest: {args.manifest}")
        print(f"Latest complete archive date: {payload.latest_date or 'unavailable'}")
        print(f"NCO cycle: {payload.nco_cycle or 'unavailable'}")
        manifest = args.manifest
    else:
        inputs = load_monitor_inputs(args.repo_root)
        metrics, paths, manifest = build_social_package(inputs, args.outdir, dpi_scale=args.dpi_scale)
        print(f"Latest complete archive date: {metrics.latest_complete['date'].date().isoformat() if metrics.latest_complete is not None else 'unavailable'}")
        print(f"NCO cycle: {metrics.nco_cycle_text}")
    print(f"Social graphics: {args.outdir}")
    for path in paths:
        print(f"  {path}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
