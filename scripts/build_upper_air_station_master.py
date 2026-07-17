"""Build a CONUS upper-air station master from NCO and IGRA metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup


IGRA_STATION_LIST_URL = (
    "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
    "doc/igra2-station-list.txt"
)
NCO_RAOB_STATIONS_URL = "https://www.nco.ncep.noaa.gov/omb/dataqc/stations/"
CACHE_DIR = Path(".cache") / "upper_air"
OUTPUT_PATH = Path("data") / "upper_air_station_master.csv"

CONUS_MIN_LAT = 24.5
CONUS_MAX_LAT = 49.5
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0
EXPECTED_NCO_REGIONS = {"E", "S", "C", "W"}
REGION_COUNTRY = {
    "B": "Caribbean/Bahamas",
    "M": "Mexico",
    "N": "Canada",
}


@dataclass(frozen=True)
class IgraStation:
    igra_id: str
    wmo_id: str
    latitude: float
    longitude: float
    station_name: str
    first_year: int
    last_year: int


@dataclass(frozen=True)
class NcoStation:
    wmo_id: str
    icao_id: str
    station_id: str
    station_name: str
    region: str
    last_use: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CONUS upper-air station metadata.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached source files.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path.")
    return parser.parse_args()


def fetch_text(url: str, cache_path: Path, refresh: bool) -> str:
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8", errors="replace")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "upper-air-network-monitor/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download metadata from {url}: {exc}") from exc

    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def parse_igra_station_list(text: str) -> dict[str, IgraStation]:
    stations: dict[str, IgraStation] = {}
    for line in text.splitlines():
        if len(line) < 81:
            continue
        try:
            igra_id = line[0:11].strip()
            latitude = float(line[12:20])
            longitude = float(line[21:30])
            station_name = line[41:71].strip()
            first_year = int(line[72:76])
            last_year = int(line[77:81])
        except ValueError:
            continue
        if len(igra_id) == 11 and igra_id[3:6] == "000" and igra_id[-5:].isdigit():
            stations[igra_id[-5:]] = IgraStation(
                igra_id=igra_id,
                wmo_id=igra_id[-5:],
                latitude=latitude,
                longitude=longitude,
                station_name=station_name,
                first_year=first_year,
                last_year=last_year,
            )
    return stations


def parse_nco_station_listing(html: str) -> list[NcoStation]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    stations: list[NcoStation] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        parts = line.split()
        if len(parts) < 6:
            continue
        wmo_id, icao_id, station_id = parts[0], parts[1], parts[2]
        region = parts[-2]
        last_use = parts[-1]
        if (
            len(wmo_id) == 5
            and wmo_id.isdigit()
            and len(icao_id) == 4
            and len(station_id) == 3
            and len(region) == 1
            and len(last_use) == 8
            and last_use.isdigit()
        ):
            name = " ".join(parts[3:-2])
            stations.append(
                NcoStation(
                    wmo_id=wmo_id,
                    icao_id=icao_id,
                    station_id=station_id,
                    station_name=name,
                    region=region,
                    last_use=last_use,
                )
            )
    return stations


def state_country_from_name(name: str, region: str) -> tuple[str, str]:
    if "," not in name:
        return "", REGION_COUNTRY.get(region, "US")
    suffix = name.rsplit(",", 1)[1].strip()
    if region in REGION_COUNTRY:
        return suffix, REGION_COUNTRY[region]
    if len(suffix) == 2 and suffix.isalpha():
        return suffix.upper(), "US"
    return "", suffix


def in_conus(latitude: float, longitude: float) -> bool:
    return (
        CONUS_MIN_LAT <= latitude <= CONUS_MAX_LAT
        and CONUS_MIN_LON <= longitude <= CONUS_MAX_LON
    )


def build_rows(nco_stations: list[NcoStation], igra_by_wmo: dict[str, IgraStation]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for nco in nco_stations:
        igra = igra_by_wmo.get(nco.wmo_id)
        if not igra:
            continue
        if not in_conus(igra.latitude, igra.longitude):
            continue
        if nco.wmo_id in seen:
            continue
        seen.add(nco.wmo_id)
        state, country = state_country_from_name(nco.station_name, nco.region)
        active_expected = nco.region in EXPECTED_NCO_REGIONS
        rows.append(
            {
                "wmo_id": nco.wmo_id,
                "station_id": nco.station_id,
                "icao_id": nco.icao_id,
                "igra_id": igra.igra_id,
                "station_name": nco.station_name or igra.station_name,
                "latitude": f"{igra.latitude:.4f}",
                "longitude": f"{igra.longitude:.4f}",
                "state": state,
                "country": country or "US",
                "nco_region": nco.region,
                "igra_first_year": str(igra.first_year),
                "igra_last_year": str(igra.last_year),
                "source": "NCO RAOB station listing; NOAA/NCEI IGRA v2 station list",
                "active_expected": "true" if active_expected else "false",
            }
        )
    rows.sort(key=lambda row: row["wmo_id"])
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("Station master is empty after CONUS filtering; refusing to write CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wmo_id",
        "station_id",
        "icao_id",
        "igra_id",
        "station_name",
        "latitude",
        "longitude",
        "state",
        "country",
        "nco_region",
        "igra_first_year",
        "igra_last_year",
        "source",
        "active_expected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        igra_text = fetch_text(
            IGRA_STATION_LIST_URL,
            CACHE_DIR / "igra2-station-list.txt",
            args.refresh,
        )
        nco_html = fetch_text(
            NCO_RAOB_STATIONS_URL,
            CACHE_DIR / "nco_raob_stations.html",
            args.refresh,
        )
        igra_by_wmo = parse_igra_station_list(igra_text)
        nco_stations = parse_nco_station_listing(nco_html)
        if not igra_by_wmo:
            raise RuntimeError("No IGRA station metadata could be parsed.")
        if not nco_stations:
            raise RuntimeError("No NCO RAOB station metadata could be parsed.")
        rows = build_rows(nco_stations, igra_by_wmo)
        write_csv(Path(args.output), rows)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    expected = sum(1 for row in rows if row["active_expected"] == "true")
    print(f"Station master rows: {len(rows)}")
    print(f"Expected active CONUS stations: {expected}")
    print(f"Output: {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
