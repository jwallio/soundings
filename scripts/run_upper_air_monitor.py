"""One-command runner for the CONUS upper-air network monitor."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CONUS upper-air monitor.")
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-igra", action="store_true")
    parser.add_argument("--skip-spc", action="store_true")
    parser.add_argument("--skip-nco", action="store_true")
    parser.add_argument(
        "--nco-source",
        choices=("nco", "iem"),
        default="nco",
        help="NCO message source: rolling live page (default) or IEM ADMSDM archive.",
    )
    parser.add_argument("--nco-start", help="IEM archive start date (YYYY-MM-DD), required with --nco-source iem.")
    parser.add_argument("--nco-end", help="IEM archive end date (YYYY-MM-DD), required with --nco-source iem.")
    parser.add_argument("--nco-limit", type=int, default=9999, help="IEM archive message limit.")
    parser.add_argument("--nco-archive-dir", default="data/raw/nco_admsdm", help="Raw IEM archive directory.")
    parser.add_argument("--outdir", default="outputs")
    return parser.parse_args()


def run_step(name: str, command: list[str], required: bool) -> tuple[int, str]:
    print(f"\n== {name} ==")
    print(" ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0 and required:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    if completed.returncode != 0:
        print(f"WARNING: {name} failed; continuing.", file=sys.stderr)
    return completed.returncode, completed.stdout + completed.stderr


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summary(outdir: Path) -> None:
    stations = read_csv(Path("data") / "upper_air_station_master.csv", dtype=str)
    igra = read_csv(outdir / "conus_balloon_launches_by_year_daily.csv")
    nco = read_csv(Path("data") / "nco_raob_availability.csv", dtype=str)
    issues = read_csv(Path("data") / "nco_raob_station_issues.csv", dtype=str)
    spc = read_csv(Path("data") / "spc_sounding_availability.csv", dtype=str)

    print("\n== Final summary ==")
    if not stations.empty:
        active = stations[stations["active_expected"].str.lower() == "true"]
        print(f"Station master count: {len(stations)} ({len(active)} active expected)")
    else:
        print("Station master count: unavailable")

    if not igra.empty:
        igra["date"] = pd.to_datetime(igra["date"], errors="coerce")
        for column in ["launches_7d_avg", "baseline_5yr_avg", "percent_vs_baseline"]:
            igra[column] = pd.to_numeric(igra[column], errors="coerce")
        years = ", ".join(str(int(year)) for year in sorted(igra["year"].dropna().unique()))
        latest = igra.dropna(subset=["date"]).sort_values("date").iloc[-1]
        print(f"IGRA years included: {years}")
        print(f"Latest IGRA date: {latest['date'].date().isoformat()}")
        print(f"Latest {int(latest['year'])} 7-day launch average: {latest['launches_7d_avg']:.2f}")
        if pd.notna(latest.get("baseline_5yr_avg")):
            print(f"Prior 5-year same-date baseline: {latest['baseline_5yr_avg']:.2f}")
            print(f"Percent difference: {latest['percent_vs_baseline']:.1f}%")
        else:
            print("Prior 5-year same-date baseline: unavailable")
            print("Percent difference: unavailable")
    else:
        print("IGRA years included: unavailable")
        print("Latest IGRA date: unavailable")
        print("Latest 7-day launch average: unavailable")
        print("Prior 5-year same-date baseline: unavailable")
        print("Percent difference: unavailable")

    if not nco.empty:
        nco["message_dt"] = pd.to_datetime(nco["message_time_utc"], utc=True, errors="coerce")
        nco["cycle_dt"] = pd.to_datetime(
            nco["cycle_date_utc"] + " " + nco["cycle_hour"].str.zfill(2) + ":00",
            utc=True,
            errors="coerce",
        )
        latest_nco = nco.dropna(subset=["cycle_dt"]).sort_values(["cycle_dt", "message_dt"]).iloc[-1]
        print(
            f"Latest NCO cycle: {latest_nco['cycle_date_utc']} {str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco['model']}"
        )
        print(f"Latest NCO CONUS RAOB count: {latest_nco['conus_count']}")
        if not issues.empty:
            latest_issues = issues[
                (issues["cycle_date_utc"] == latest_nco["cycle_date_utc"])
                & (issues["cycle_hour"].astype(str).str.zfill(2) == str(latest_nco["cycle_hour"]).zfill(2))
            ]
            issue_stations = ", ".join(sorted(latest_issues["station_id"].dropna().astype(str).unique()))
            print(f"Latest NCO missing/problem stations: {issue_stations or 'none'}")
        else:
            print("Latest NCO missing/problem stations: unavailable")
    else:
        print("Latest NCO cycle: unavailable")
        print("Latest NCO CONUS RAOB count: unavailable")
        print("Latest NCO missing/problem stations: unavailable")

    if not spc.empty:
        row = spc.iloc[-1]
        print(f"SPC parser status: {row.get('parser_method', 'unknown')}")
    else:
        print("SPC parser status: unavailable")

    print(f"Output paths:")
    print(f"  {outdir / 'conus_balloon_launches_by_year_daily.csv'}")
    print(f"  {outdir / 'conus_balloon_launches_station_deficits.csv'}")
    print(f"  {outdir / 'conus_balloon_launches_by_year_comparison.png'}")
    print(f"  {outdir / 'upper_air_network_dashboard.png'}")
    print(f"  {outdir / 'social_upper_air'}")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    python = sys.executable
    refresh = ["--refresh"] if args.refresh else []

    try:
        run_step(
            "Build station master",
            [python, "scripts/build_upper_air_station_master.py", *refresh],
            required=True,
        )
        if not args.skip_igra:
            run_step(
                "Build IGRA launch counts",
                [
                    python,
                    "scripts/build_conus_igra_launch_counts_by_year.py",
                    "--years",
                    str(args.years),
                    "--outdir",
                    str(outdir),
                    *refresh,
                ],
                required=True,
            )
        if not args.skip_nco:
            nco_command = [python, "scripts/parse_nco_sdm_raob_messages.py"]
            if args.nco_source == "iem":
                if not args.nco_start or not args.nco_end:
                    raise RuntimeError("--nco-start and --nco-end are required with --nco-source iem")
                nco_command.extend(
                    [
                        "--source",
                        "iem",
                        "--start",
                        args.nco_start,
                        "--end",
                        args.nco_end,
                        "--limit",
                        str(args.nco_limit),
                        "--archive-dir",
                        args.nco_archive_dir,
                    ]
                )
            run_step(
                "Parse NCO SDM RAOB messages",
                nco_command,
                required=False,
            )
        if not args.skip_spc:
            run_step(
                "Parse SPC sounding page",
                [python, "scripts/parse_spc_sounding_page.py"],
                required=False,
            )
        run_step(
            "Make dashboard",
            [python, "scripts/make_upper_air_dashboard.py", "--outdir", str(outdir)],
            required=True,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
