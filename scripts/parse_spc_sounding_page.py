"""Parse the SPC observed sounding page without OCR."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


SPC_SOUNDING_URL = "https://www.spc.noaa.gov/exper/soundings/"
STATION_MASTER_PATH = Path("data") / "upper_air_station_master.csv"
OUTPUT_PATH = Path("data") / "spc_sounding_availability.csv"
FIELDS = [
    "run_time_utc",
    "cycle_hour",
    "available_count",
    "expected_count",
    "availability_percent",
    "available_station_ids",
    "missing_station_ids",
    "parser_method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse SPC observed sounding availability.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path.")
    parser.add_argument("--station-master", default=str(STATION_MASTER_PATH))
    parser.add_argument("--experimental-image-detect", action="store_true")
    return parser.parse_args()


def expected_station_ids(path: Path) -> set[str]:
    if not path.exists():
        raise RuntimeError(f"Missing station master at {path}.")
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        raise RuntimeError(f"Station master is empty: {path}.")
    active = df[df["active_expected"].str.lower() == "true"].copy()
    return set(active["station_id"].dropna().astype(str).str.upper())


def download_page() -> str:
    try:
        response = requests.get(
            SPC_SOUNDING_URL,
            timeout=60,
            headers={"User-Agent": "upper-air-network-monitor/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download SPC sounding page: {exc}") from exc
    return response.text


def infer_cycle_hour(html: str, run_time: dt.datetime) -> str:
    cycle_matches = [int(value) for value in re.findall(r"\b(00|12)Z\b", html, flags=re.I)]
    if cycle_matches:
        return f"{cycle_matches[0]:02d}"
    return "12" if run_time.hour >= 12 else "00"


def extract_station_ids(html: str, expected_ids: set[str]) -> tuple[set[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: set[str] = set()

    for attr in ("href", "src", "data-station", "data-site", "title", "alt"):
        for tag in soup.find_all(attrs={attr: True}):
            value = str(tag.get(attr, ""))
            candidates.update(re.findall(r"(?<![A-Z0-9])K?([A-Z]{3})(?![A-Z0-9])", value.upper()))

    text = soup.get_text(" ")
    candidates.update(re.findall(r"(?<![A-Z0-9])K?([A-Z]{3})(?![A-Z0-9])", text.upper()))

    script_text = "\n".join(script.get_text("\n") for script in soup.find_all("script"))
    candidates.update(re.findall(r"(?<![A-Z0-9])K?([A-Z]{3})(?![A-Z0-9])", script_text.upper()))

    available = {station for station in candidates if station in expected_ids}
    if available:
        return available, "html_links_scripts"
    return set(), "unavailable_no_text_station_ids"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.experimental_image_detect:
        print(
            "WARNING: --experimental-image-detect is reserved; OCR/image detection is not implemented.",
            file=sys.stderr,
        )

    output = Path(args.output)
    try:
        expected_ids = expected_station_ids(Path(args.station_master))
        html = download_page()
    except RuntimeError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        write_rows(output, [])
        print("SPC parser status: unavailable")
        print(f"SPC output: {output}")
        return 0

    run_time = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    available_ids, method = extract_station_ids(html, expected_ids)
    if not available_ids:
        print(
            "WARNING: SPC sounding station IDs were not available in page HTML, links, scripts, or metadata.",
            file=sys.stderr,
        )
        write_rows(output, [])
        print("SPC parser status: unavailable_no_text_station_ids")
        print(f"SPC output: {output}")
        return 0

    missing_ids = sorted(expected_ids - available_ids)
    available_sorted = sorted(available_ids)
    expected_count = len(expected_ids)
    availability_percent = (len(available_ids) / expected_count * 100.0) if expected_count else 0.0
    row = {
        "run_time_utc": run_time.isoformat().replace("+00:00", "Z"),
        "cycle_hour": infer_cycle_hour(html, run_time),
        "available_count": str(len(available_ids)),
        "expected_count": str(expected_count),
        "availability_percent": f"{availability_percent:.1f}",
        "available_station_ids": ";".join(available_sorted),
        "missing_station_ids": ";".join(missing_ids),
        "parser_method": method,
    }
    write_rows(output, [row])
    print(f"SPC parser status: {method}")
    print(f"SPC available stations: {len(available_ids)} of {expected_count}")
    print(f"SPC output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
