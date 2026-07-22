"""Build a by-year CONUS IGRA sounding launch comparison chart."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import shutil
import statistics
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


BASE_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive"
STATION_LIST_URL = f"{BASE_URL}/doc/igra2-station-list.txt"
Y2D_URL = f"{BASE_URL}/access/data-y2d"
POR_URL = f"{BASE_URL}/access/data-por"
CACHE_DIR = Path(".cache") / "igra"

CONUS_MIN_LAT = 24.5
CONUS_MAX_LAT = 49.5
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0

CSV_FILENAME = "conus_balloon_launches_by_year_daily.csv"
PNG_FILENAME = "conus_balloon_launches_by_year_comparison.png"
METADATA_FILENAME = "conus_balloon_launches_metadata.json"
STATION_DEFICITS_FILENAME = "conus_balloon_launches_station_deficits.csv"
STATION_MASTER_PATH = Path("data") / "upper_air_station_master.csv"
REQUEST_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_WORKERS = 8


@dataclass(frozen=True)
class Station:
    station_id: str
    latitude: float
    longitude: float
    name: str
    first_year: int
    last_year: int


@dataclass(frozen=True)
class DownloadTarget:
    station: Station
    kind: str
    url: str
    path: Path
    start_date: dt.date
    end_date: dt.date


@dataclass(frozen=True)
class ParsedFile:
    station_id: str
    kind: str
    url: str
    path: Path
    launch_counts: Counter[dt.date]
    failed_download: str | None = None
    parse_error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a by-year CONUS IGRA v2 sounding launch comparison."
    )
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--end-date", help="Optional inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--outdir", default="outputs")
    return parser.parse_args()


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid YYYY-MM-DD date: {value}") from exc


def complete_end_date(value: str | None) -> dt.date:
    if value:
        return parse_iso_date(value)
    return dt.date.today() - dt.timedelta(days=1)


def command_used() -> str:
    return "python " + " ".join(sys.argv)


def download_file(url: str, path: Path, refresh: bool) -> Path:
    if path.exists() and not refresh:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "comfortwx-igra-counts/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(path)
    return path


def cached_text(url: str, path: Path, refresh: bool) -> str:
    try:
        download_file(url, path, refresh)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download NOAA URL {url}: {exc}") from exc
    return path.read_text(encoding="utf-8", errors="replace")


def parse_station_list(text: str) -> list[Station]:
    stations: list[Station] = []
    for line in text.splitlines():
        if len(line) < 88:
            continue
        try:
            stations.append(
                Station(
                    station_id=line[0:11].strip(),
                    latitude=float(line[12:20]),
                    longitude=float(line[21:30]),
                    name=line[41:71].strip(),
                    first_year=int(line[72:76]),
                    last_year=int(line[77:81]),
                )
            )
        except ValueError:
            continue
    return stations


def in_conus(station: Station) -> bool:
    return (
        station.station_id.startswith("US")
        and CONUS_MIN_LAT <= station.latitude <= CONUS_MAX_LAT
        and CONUS_MIN_LON <= station.longitude <= CONUS_MAX_LON
    )


def load_station_master(path: Path) -> dict[str, Station]:
    if not path.exists():
        raise RuntimeError(
            f"Missing station master at {path}. Run scripts/build_upper_air_station_master.py first."
        )

    stations: dict[str, Station] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("active_expected", "")).lower() != "true":
                continue
            igra_id = str(row.get("igra_id", "")).strip()
            if not igra_id:
                wmo_id = str(row.get("wmo_id", "")).strip()
                igra_id = f"USM000{wmo_id}" if wmo_id.isdigit() else ""
            try:
                latitude = float(str(row.get("latitude", "")))
                longitude = float(str(row.get("longitude", "")))
                first_year = int(str(row.get("igra_first_year", "0") or "0"))
                last_year = int(str(row.get("igra_last_year", "9999") or "9999"))
            except ValueError:
                continue
            station = Station(
                station_id=igra_id,
                latitude=latitude,
                longitude=longitude,
                name=str(row.get("station_name", "")).strip(),
                first_year=first_year,
                last_year=last_year,
            )
            if in_conus(station):
                stations[igra_id] = station
    if not stations:
        raise RuntimeError(f"No active expected CONUS stations found in {path}.")
    return stations


def parse_y2d_index(html: str) -> dict[str, tuple[str, int]]:
    files: dict[str, tuple[str, int]] = {}
    pattern = re.compile(r"([A-Z0-9]{11}-data-beg(\d{4})\.txt\.zip)")
    for filename, year_text in pattern.findall(html):
        files[filename[:11]] = (filename, int(year_text))
    return files


def parse_year_file_index(html: str) -> dict[tuple[str, int], str]:
    files: dict[tuple[str, int], str] = {}
    pattern = re.compile(r"([A-Z0-9]{11}-data-(\d{4})\.txt\.zip)")
    for filename, year_text in pattern.findall(html):
        files[(filename[:11], int(year_text))] = filename
    return files


def date_range(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    return [
        start_date + dt.timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]


def target_years(end_date: dt.date, years: int) -> list[int]:
    end_year = end_date.year
    return list(range(end_year - years + 1, end_year + 1))


def build_download_targets(
    stations: list[Station],
    y2d_files: dict[str, tuple[str, int]],
    yearly_files: dict[tuple[str, int], str],
    years: list[int],
    requested_end_date: dt.date,
) -> tuple[list[DownloadTarget], int]:
    targets: list[DownloadTarget] = []
    included_station_ids: set[str] = set()
    first_year = years[0]
    final_year = years[-1]

    for station in stations:
        if station.first_year > final_year or station.last_year < first_year:
            continue
        y2d_info = y2d_files.get(station.station_id)
        if not y2d_info:
            continue

        filename, y2d_start_year = y2d_info
        included_station_ids.add(station.station_id)
        yearly_years: set[int] = set()

        for year in years:
            if station.first_year > year or station.last_year < year:
                continue
            yearly_filename = yearly_files.get((station.station_id, year))
            if yearly_filename:
                year_start = dt.date(year, 1, 1)
                year_end = requested_end_date if year == final_year else dt.date(year, 12, 31)
                targets.append(
                    DownloadTarget(
                        station=station,
                        kind=f"data-year-{year}",
                        url=f"{POR_URL}/{yearly_filename}",
                        path=CACHE_DIR / "data-year" / str(year) / yearly_filename,
                        start_date=year_start,
                        end_date=year_end,
                    )
                )
                yearly_years.add(year)

        grouped_years = [
            year
            for year in years
            if year not in yearly_years
            and station.first_year <= year <= station.last_year
        ]
        y2d_years = [year for year in grouped_years if year >= y2d_start_year]
        por_years = [year for year in grouped_years if year < y2d_start_year]

        if y2d_years:
            start_year = min(y2d_years)
            end_year = max(y2d_years)
            targets.append(
                DownloadTarget(
                    station=station,
                    kind="data-y2d",
                    url=f"{Y2D_URL}/{filename}",
                    path=CACHE_DIR / "data-y2d" / filename,
                    start_date=dt.date(start_year, 1, 1),
                    end_date=requested_end_date if end_year == final_year else dt.date(end_year, 12, 31),
                )
            )

        if por_years:
            start_year = min(por_years)
            end_year = max(por_years)
            por_filename = f"{station.station_id}-data.txt.zip"
            targets.append(
                DownloadTarget(
                    station=station,
                    kind="data-por",
                    url=f"{POR_URL}/{por_filename}",
                    path=CACHE_DIR / "data-por" / por_filename,
                    start_date=dt.date(start_year, 1, 1),
                    end_date=requested_end_date if end_year == final_year else dt.date(end_year, 12, 31),
                )
            )

    deduped: dict[tuple[str, str, dt.date, dt.date], DownloadTarget] = {}
    for target in targets:
        key = (target.station.station_id, target.kind, target.start_date, target.end_date)
        deduped[key] = target
    return list(deduped.values()), len(included_station_ids)


def parse_igra_zip(target: DownloadTarget) -> ParsedFile:
    try:
        if not target.path.exists():
            return ParsedFile(
                station_id=target.station.station_id,
                kind=target.kind,
                url=target.url,
                path=target.path,
                launch_counts=Counter(),
                failed_download="file is missing after download",
            )

        counts: Counter[dt.date] = Counter()
        with zipfile.ZipFile(target.path) as archive:
            data_members = [name for name in archive.namelist() if name.endswith(".txt")]
            if not data_members:
                raise ValueError("zip contains no .txt sounding file")

            with archive.open(data_members[0]) as raw:
                for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                    if not line.startswith("#"):
                        continue
                    try:
                        year = int(line[13:17])
                        month = int(line[18:20])
                        day = int(line[21:23])
                        _hour = int(line[24:26])
                        launch_date = dt.date(year, month, day)
                    except ValueError:
                        continue
                    if target.start_date <= launch_date <= target.end_date:
                        counts[launch_date] += 1

        return ParsedFile(
            station_id=target.station.station_id,
            kind=target.kind,
            url=target.url,
            path=target.path,
            launch_counts=counts,
        )
    except Exception as exc:
        return ParsedFile(
            station_id=target.station.station_id,
            kind=target.kind,
            url=target.url,
            path=target.path,
            launch_counts=Counter(),
            parse_error=str(exc),
        )


def download_and_parse_targets(
    targets: list[DownloadTarget], refresh: bool
) -> tuple[Counter[dt.date], dict[str, Counter[dt.date]], list[str], list[str]]:
    total_counts: Counter[dt.date] = Counter()
    station_counts: dict[str, Counter[dt.date]] = {}
    failed_downloads: list[str] = []
    parse_errors: list[str] = []

    def fetch_then_parse(target: DownloadTarget) -> ParsedFile:
        try:
            download_file(target.url, target.path, refresh)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            return ParsedFile(
                station_id=target.station.station_id,
                kind=target.kind,
                url=target.url,
                path=target.path,
                launch_counts=Counter(),
                failed_download=(
                    f"{target.kind} {target.station.station_id}: "
                    f"failed NOAA URL {target.url}: {exc}"
                ),
            )
        return parse_igra_zip(target)

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(fetch_then_parse, target) for target in targets]
        for future in as_completed(futures):
            parsed = future.result()
            if parsed.failed_download:
                failed_downloads.append(parsed.failed_download)
            if parsed.parse_error:
                parse_errors.append(
                    f"{parsed.kind} {parsed.station_id}: {parsed.url}: {parsed.parse_error}"
                )
            total_counts.update(parsed.launch_counts)
            station_counts.setdefault(parsed.station_id, Counter()).update(parsed.launch_counts)

    return total_counts, station_counts, sorted(failed_downloads), sorted(parse_errors)


def moving_average(values: list[int], window: int = 7) -> list[float]:
    averages: list[float] = []
    running_sum = 0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        averages.append(running_sum / min(index + 1, window))
    return averages


def month_day_label(value: dt.date) -> str:
    return value.strftime("%b %d")


def comparison_x_date(value: dt.date) -> dt.date:
    return dt.date(2000, value.month, value.day)


def build_rows(
    counts: Counter[dt.date],
    years: list[int],
    latest_date: dt.date,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int | float]] = []
    for year in years:
        start_date = dt.date(year, 1, 1)
        end_date = latest_date if year == years[-1] else dt.date(year, 12, 31)
        dates = date_range(start_date, end_date)
        launches = [counts.get(launch_date, 0) for launch_date in dates]
        averages = moving_average(launches)

        for launch_date, launch_count, average in zip(dates, launches, averages):
            rows.append(
                {
                    "date": launch_date.isoformat(),
                    "year": launch_date.year,
                    "month": launch_date.month,
                    "day": launch_date.day,
                    "day_of_year": launch_date.timetuple().tm_yday,
                    "month_day": month_day_label(launch_date),
                    "launches": launch_count,
                    "launches_7d_avg": average,
                }
            )

    by_year_month_day = {
        (int(row["year"]), str(row["month_day"])): float(row["launches_7d_avg"])
        for row in rows
    }
    for row in rows:
        year = int(row["year"])
        month_day = str(row["month_day"])
        prior_values = [
            by_year_month_day[(prior_year, month_day)]
            for prior_year in range(year - 5, year)
            if (prior_year, month_day) in by_year_month_day
        ]
        if prior_values:
            baseline = sum(prior_values) / len(prior_values)
            difference = float(row["launches_7d_avg"]) - baseline
            percent = (difference / baseline * 100.0) if baseline else 0.0
            row["baseline_5yr_avg"] = f"{baseline:.2f}"
            row["difference_vs_baseline"] = f"{difference:.2f}"
            row["percent_vs_baseline"] = f"{percent:.1f}"
        else:
            row["baseline_5yr_avg"] = ""
            row["difference_vs_baseline"] = ""
            row["percent_vs_baseline"] = ""
        row["launches_7d_avg"] = f"{float(row['launches_7d_avg']):.2f}"
    return rows


def expected_row_count(years: list[int], latest_date: dt.date) -> int:
    total = 0
    for year in years:
        start_date = dt.date(year, 1, 1)
        end_date = latest_date if year == years[-1] else dt.date(year, 12, 31)
        total += (end_date - start_date).days + 1
    return total


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "date",
        "year",
        "month",
        "day",
        "day_of_year",
        "month_day",
        "launches",
        "launches_7d_avg",
        "baseline_5yr_avg",
        "difference_vs_baseline",
        "percent_vs_baseline",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_complete_date(counts: Counter[dt.date], latest_date: dt.date) -> dt.date:
    """Apply the monitor's partial-day rule to aggregate daily counts."""
    dates = date_range(dt.date(latest_date.year, 1, 1), latest_date)
    if len(dates) < 15:
        return latest_date
    values = [counts.get(value, 0) for value in dates]
    averages = moving_average(values)
    prior_median = statistics.median(values[-15:-1])
    if prior_median > 0 and values[-1] < prior_median * 0.85 and averages[-1] < averages[-2] - 2.0:
        return latest_date - dt.timedelta(days=1)
    return latest_date


def _station_window(
    counts: Counter[dt.date],
    end_date: dt.date,
    days: int,
    baseline_years: list[int],
) -> tuple[float, float, float, float]:
    observed = 0.0
    expected = 0.0
    for offset in range(days):
        current_date = end_date - dt.timedelta(days=offset)
        observed += float(counts.get(current_date, 0))
        same_date_counts: list[float] = []
        for year in baseline_years:
            try:
                baseline_date = current_date.replace(year=year)
            except ValueError:
                continue
            same_date_counts.append(float(counts.get(baseline_date, 0)))
        if same_date_counts:
            expected += sum(same_date_counts) / len(same_date_counts)
    deficit = observed - expected
    percent = deficit / expected * 100.0 if expected else 0.0
    return observed, expected, deficit, percent


def build_station_deficit_rows(
    stations: list[Station],
    station_counts: dict[str, Counter[dt.date]],
    eligible_station_ids: set[str],
    end_date: dt.date,
    baseline_years: list[int],
) -> list[dict[str, object]]:
    """Build station-level archive shortfalls without implying launch causation."""
    if not baseline_years:
        return []
    rows: list[dict[str, object]] = []
    for station in stations:
        if station.station_id not in eligible_station_ids:
            continue
        counts = station_counts.get(station.station_id, Counter())
        row: dict[str, object] = {
            "station_id": station.station_id,
            "name": station.name,
            "latitude": round(station.latitude, 4),
            "longitude": round(station.longitude, 4),
            "latest_complete_date": end_date.isoformat(),
            "baseline_years": ";".join(str(year) for year in baseline_years),
        }
        # Keep the station ranking payload useful for the dashboard's range
        # selector while using the same date-effective baseline calculation.
        for days in (7, 30, 60, 90, 180, 365):
            observed, expected, deficit, percent = _station_window(counts, end_date, days, baseline_years)
            row[f"observed_{days}"] = round(observed, 2)
            row[f"expected_{days}"] = round(expected, 2)
            row[f"deficit_{days}"] = round(deficit, 2)
            row[f"percent_{days}"] = round(percent, 2)
        row["missed_90"] = round(max(-float(row["deficit_90"]), 0.0), 2)
        rows.append(row)
    return sorted(rows, key=lambda item: (-float(item["missed_90"]), str(item["station_id"])))


def write_station_deficits(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "station_id",
        "name",
        "latitude",
        "longitude",
        "latest_complete_date",
        "baseline_years",
        "observed_7",
        "expected_7",
        "deficit_7",
        "percent_7",
        "observed_30",
        "expected_30",
        "deficit_30",
        "percent_30",
        "observed_60",
        "expected_60",
        "deficit_60",
        "percent_60",
        "observed_90",
        "expected_90",
        "deficit_90",
        "percent_90",
        "observed_180",
        "expected_180",
        "deficit_180",
        "percent_180",
        "observed_365",
        "expected_365",
        "deficit_365",
        "percent_365",
        "missed_90",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_chart(path: Path, rows: list[dict[str, str | int]], years: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = plt.get_cmap("tab10").colors
    current_year = years[-1]

    for index, year in enumerate(years):
        year_rows = [row for row in rows if int(row["year"]) == year]
        x_values = [
            comparison_x_date(dt.date(year, int(row["month"]), int(row["day"])))
            for row in year_rows
        ]
        y_values = [float(row["launches_7d_avg"]) for row in year_rows]
        is_current = year == current_year
        ax.plot(
            x_values,
            y_values,
            color=colors[index % len(colors)],
            linewidth=2.8 if is_current else 1.5,
            alpha=0.95 if is_current else 0.72,
            label=str(year),
        )

    ax.set_title("CONUS Weather Balloon Launches by Year — IGRA", fontsize=15, pad=14)
    current_rows = [row for row in rows if int(row["year"]) == current_year]
    baseline_rows = [row for row in current_rows if str(row.get("baseline_5yr_avg", ""))]
    if baseline_rows:
        x_values = [
            comparison_x_date(dt.date(current_year, int(row["month"]), int(row["day"])))
            for row in baseline_rows
        ]
        y_values = [float(row["baseline_5yr_avg"]) for row in baseline_rows]
        ax.plot(
            x_values,
            y_values,
            color="#222222",
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            label="prior 5-year baseline",
        )

    ax.set_xlabel("Month/day")
    ax.set_ylabel("Number of launches/soundings")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    ax.legend(title="Year", ncol=3, frameon=False, loc="upper left")
    ax.set_xlim(dt.date(2000, 1, 1), dt.date(2000, 12, 31))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.text(
        0.01,
        0.01,
        (
            "Daily sounding/header counts from NOAA/NCEI IGRA v2; CONUS stations only. "
            "Lines show 7-day moving averages."
        ),
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def validate_outputs(
    stations: list[Station],
    counts: Counter[dt.date],
    rows: list[dict[str, str | int]],
    years: list[int],
    latest_date: dt.date,
) -> None:
    if not stations:
        raise RuntimeError("No CONUS stations were found in the IGRA station metadata.")
    if not counts:
        raise RuntimeError("No IGRA sounding header records were counted for the requested years.")
    if latest_date.year != years[-1]:
        raise RuntimeError(
            f"No current-year IGRA data were found for {years[-1]}; latest counted date is {latest_date}."
        )

    latest_row_date = max(dt.date.fromisoformat(str(row["date"])) for row in rows)
    if latest_row_date > latest_date:
        raise RuntimeError(
            f"Current-year rows extend past latest available IGRA date: {latest_row_date} > {latest_date}."
        )

    expected_rows = expected_row_count(years, latest_date)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"CSV row count {len(rows)} does not match expected calendar-day count {expected_rows}."
        )


def main() -> int:
    args = parse_args()
    if args.years <= 0:
        print("--years must be positive", file=sys.stderr)
        return 2

    requested_end_date = complete_end_date(args.end_date)
    years = target_years(requested_end_date, args.years)
    outdir = Path(args.outdir)
    csv_path = outdir / CSV_FILENAME
    png_path = outdir / PNG_FILENAME
    metadata_path = outdir / METADATA_FILENAME
    station_deficits_path = outdir / STATION_DEFICITS_FILENAME

    try:
        station_text = cached_text(
            STATION_LIST_URL, CACHE_DIR / "igra2-station-list.txt", args.refresh
        )
        y2d_index_text = cached_text(
            Y2D_URL + "/", CACHE_DIR / "data-y2d-index.html", args.refresh
        )
        por_index_text = cached_text(
            POR_URL + "/", CACHE_DIR / "data-por-index.html", args.refresh
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        master_stations = load_station_master(STATION_MASTER_PATH)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    station_by_id = {station.station_id: station for station in parse_station_list(station_text)}
    conus_stations = [
        station_by_id.get(station_id, master_station)
        for station_id, master_station in master_stations.items()
    ]
    if not conus_stations:
        print("No CONUS stations were found from the station master.", file=sys.stderr)
        return 1

    y2d_files = parse_y2d_index(y2d_index_text)
    yearly_files = parse_year_file_index(por_index_text)
    targets, included_station_count = build_download_targets(
        conus_stations, y2d_files, yearly_files, years, requested_end_date
    )
    if not targets:
        print("No CONUS IGRA station data files matched the requested years.", file=sys.stderr)
        return 1

    counts, station_counts, failed_downloads, parse_errors = download_and_parse_targets(targets, args.refresh)
    if failed_downloads or parse_errors:
        for failure in failed_downloads:
            print(f"Failed download: {failure}", file=sys.stderr)
        for error in parse_errors:
            print(f"Parse error: {error}", file=sys.stderr)
        return 1

    latest_date = max((date for date in counts if date.year == years[-1]), default=None)
    if latest_date is None:
        latest_overall = max(counts, default=None)
        print(
            f"No IGRA header records were found for {years[-1]}; latest overall date is {latest_overall}.",
            file=sys.stderr,
        )
        return 1

    rows = build_rows(counts, years, latest_date)
    try:
        validate_outputs(conus_stations, counts, rows, years, latest_date)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_csv(csv_path, rows)
    complete_date = latest_complete_date(counts, latest_date)
    baseline_years = [year for year in (2021, 2022, 2023, 2024) if year in years]
    eligible_station_ids = {
        target.station.station_id
        for target in targets
        if target.start_date.year <= years[-1] <= target.end_date.year
    }
    station_deficit_rows = build_station_deficit_rows(
        conus_stations,
        station_counts,
        eligible_station_ids,
        complete_date,
        baseline_years,
    )
    write_station_deficits(station_deficits_path, station_deficit_rows)
    write_chart(png_path, rows, years)
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "latest_igra_date": latest_date.isoformat(),
                "requested_end_date": requested_end_date.isoformat(),
                "years_included": years,
                "included_station_count": included_station_count,
                "station_deficit_row_count": len(station_deficit_rows),
                "station_deficit_latest_complete_date": complete_date.isoformat(),
                "station_deficit_baseline_years": baseline_years,
                "station_master_count": len(conus_stations),
                "failed_download_count": 0,
                "parse_error_count": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    totals_by_year = {
        year: sum(int(row["launches"]) for row in rows if int(row["year"]) == year)
        for year in years
    }

    print(f"Command used: {command_used()}")
    print(f"Years included: {', '.join(str(year) for year in years)}")
    print(f"CONUS stations included: {included_station_count}")
    print(f"Latest IGRA date found: {latest_date.isoformat()}")
    print("Total launches per year:")
    for year in years:
        print(f"  {year}: {totals_by_year[year]}")
    print(f"CSV output: {csv_path}")
    print(f"PNG output: {png_path}")
    print(f"Metadata output: {metadata_path}")
    print("Failed downloads: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
