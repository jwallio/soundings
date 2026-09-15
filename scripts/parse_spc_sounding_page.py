"""Parse every currently listed SPC observed-sounding archive period.

The SPC landing page is an index.  Its archive links point to detail pages,
and the detail pages contain the station map (and therefore the station IDs).
This parser follows every ``*_OBS/`` link exposed by the index instead of
assuming that the only useful cycles are 00Z and 12Z.  Supplemental periods
such as 06Z, 15Z, 16Z, 18Z, and 21Z are retained when SPC publishes them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


SPC_SOUNDING_URL = "https://www.spc.noaa.gov/exper/soundings/"
STATION_MASTER_PATH = Path("data") / "upper_air_station_master.csv"
OUTPUT_PATH = Path("data") / "spc_sounding_availability.csv"
REQUEST_TIMEOUT = (20, 15)
MAX_DOWNLOAD_ATTEMPTS = 2
USER_AGENT = "upper-air-network-monitor/1.1 (+https://soundings.wall.cloud/)"

FIELDS = [
    "run_time_utc",
    "sounding_time_utc",
    "cycle_date_utc",
    "cycle_hour",
    "available_count",
    "expected_count",
    "availability_percent",
    "available_station_ids",
    "missing_station_ids",
    "source_station_count",
    "source_station_ids",
    "untracked_station_ids",
    "source_url",
    "page_status",
    "parser_method",
]

ARCHIVE_LINK_RE = re.compile(r"(?<!\d)(?P<stamp>\d{8})_OBS(?:/|$)", flags=re.IGNORECASE)
STATION_ID_RE = re.compile(r"(?<![A-Z0-9])K?([A-Z]{3})(?![A-Z0-9])", flags=re.IGNORECASE)
SHOW_SOUNDING_RE = re.compile(
    r"show_soundings\s*\(\s*['\"]K?([A-Z]{3})['\"]\s*\)",
    flags=re.IGNORECASE,
)
DETAIL_PAGE_RE = re.compile(r"Observed\s+(?:Radiosonde\s+)?Data", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse all SPC observed sounding archive periods.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path.")
    parser.add_argument("--station-master", default=str(STATION_MASTER_PATH))
    parser.add_argument(
        "--experimental-image-detect",
        action="store_true",
        help="Deprecated compatibility flag; image detection is not required by the HTML parser.",
    )
    return parser.parse_args()


def expected_station_ids(path: Path) -> set[str]:
    if not path.exists():
        raise RuntimeError(f"Missing station master at {path}.")
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        raise RuntimeError(f"Station master is empty: {path}.")
    required = {"active_expected", "station_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Station master is missing columns: {', '.join(sorted(missing))}.")
    active = df[df["active_expected"].fillna("").str.lower().isin({"true", "1", "yes"})]
    return set(active["station_id"].dropna().astype(str).str.upper())


def _usable_partial_response(url: str, body: bytes) -> bool:
    """Accept SPC's occasionally unterminated chunked response when useful."""
    if len(body) < 512:
        return False
    text = body.decode("utf-8", errors="replace")
    if url.rstrip("/") == SPC_SOUNDING_URL.rstrip("/"):
        return bool(ARCHIVE_LINK_RE.search(text))
    return bool(DETAIL_PAGE_RE.search(BeautifulSoup(text, "html.parser").get_text(" ")))


def download_page(url: str = SPC_SOUNDING_URL) -> str:
    """Download one SPC page, tolerating an incomplete end-of-stream marker."""
    last_error: requests.RequestException | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        response: requests.Response | None = None
        chunks: list[bytes] = []
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Connection": "close",
                },
            )
            response.raise_for_status()
            end_marker = b""
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    end_marker = (end_marker + chunk.lower())[-64:]
                    if b"</html" in end_marker or b"</body" in end_marker:
                        break
            body = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            return body.decode(encoding, errors="replace")
        except requests.RequestException as exc:
            last_error = exc
            body = b"".join(chunks)
            if _usable_partial_response(url, body):
                print(
                    f"WARNING: SPC response ended early for {url}; using {len(body)} bytes already received.",
                    file=sys.stderr,
                )
                encoding = response.encoding if response is not None else None
                return body.decode(encoding or "utf-8", errors="replace")
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"Failed to download SPC sounding page {url}: {exc}") from exc
        finally:
            if response is not None:
                response.close()
    raise RuntimeError(f"Failed to download SPC sounding page {url}: {last_error}")


def parse_archive_time(stamp: str) -> dt.datetime:
    """Convert SPC's YYMMDDHH directory stamp to a UTC datetime."""
    if not re.fullmatch(r"\d{8}", stamp):
        raise ValueError(f"Invalid SPC archive stamp: {stamp!r}")
    try:
        # SPC archive paths use two-digit years.  The archive is a current
        # seven-day product, so explicitly anchoring it to 2000 avoids
        # datetime's 1969/2068 pivot behavior.
        return dt.datetime(
            2000 + int(stamp[0:2]),
            int(stamp[2:4]),
            int(stamp[4:6]),
            int(stamp[6:8]),
            tzinfo=dt.timezone.utc,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid SPC archive stamp: {stamp!r}") from exc


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def discover_archive_entries(html: str) -> list[dict[str, str]]:
    """Return one deduplicated entry for every ``*_OBS/`` link on the index."""
    soup = BeautifulSoup(html, "html.parser")
    hrefs = [str(tag.get("href", "")) for tag in soup.find_all("a", href=True)]
    # Keep a regex fallback for a malformed index where links are embedded in
    # plain text or an attribute BeautifulSoup does not expose as an anchor.
    hrefs.extend(match.group(0) for match in ARCHIVE_LINK_RE.finditer(html))

    entries: dict[str, dict[str, str]] = {}
    for href in hrefs:
        match = ARCHIVE_LINK_RE.search(href)
        if not match:
            continue
        stamp = match.group("stamp")
        try:
            sounding_time = parse_archive_time(stamp)
        except ValueError:
            continue
        source_url = urljoin(SPC_SOUNDING_URL, href)
        key = _iso(sounding_time)
        entries[key] = {
            "stamp": stamp,
            "sounding_time_utc": key,
            "cycle_date_utc": sounding_time.date().isoformat(),
            "cycle_hour": f"{sounding_time.hour:02d}",
            "source_url": source_url,
        }
    return sorted(entries.values(), key=lambda item: item["sounding_time_utc"])


def _station_candidates(html: str) -> tuple[set[str], bool]:
    """Extract raw station IDs from a detail-page map or station links."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: set[str] = set()
    areas = soup.find_all("area")
    tags = areas or soup.find_all(["a", "area"])
    for tag in tags:
        for attr in ("href", "data-station", "data-site", "title", "alt"):
            value = str(tag.get(attr, ""))
            candidates.update(match.group(1).upper() for match in STATION_ID_RE.finditer(value))

    # The current SPC detail page uses javascript:show_soundings("XXX") in
    # image-map area hrefs.  Keep this explicit fallback for future markup
    # that moves the station map into script text.
    candidates.update(match.group(1).upper() for match in SHOW_SOUNDING_RE.finditer(html))
    return candidates, bool(areas)


def extract_all_station_ids(html: str) -> tuple[set[str], str]:
    """Return all station IDs represented by a detail page."""
    candidates, has_area_map = _station_candidates(html)
    if candidates:
        return candidates, "detail_map_area" if has_area_map else "detail_station_links"
    if DETAIL_PAGE_RE.search(BeautifulSoup(html, "html.parser").get_text(" ")):
        return set(), "detail_page_no_station_ids"
    return set(), "unavailable_no_detail_page_marker"


def extract_station_ids(html: str, expected_ids: set[str]) -> tuple[set[str], str]:
    """Return the subset of detail-page station IDs in the expected inventory."""
    candidates, method = extract_all_station_ids(html)
    available = candidates & {station.upper() for station in expected_ids}
    if available:
        return available, method
    if candidates:
        return set(), "detail_map_no_expected_station_ids"
    return set(), method


def _detail_is_valid(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    return bool(DETAIL_PAGE_RE.search(text))


def _base_row(entry: dict[str, str], run_time: dt.datetime, expected_count: int) -> dict[str, str]:
    return {
        "run_time_utc": _iso(run_time),
        "sounding_time_utc": entry["sounding_time_utc"],
        "cycle_date_utc": entry["cycle_date_utc"],
        "cycle_hour": entry["cycle_hour"],
        "available_count": "",
        "expected_count": str(expected_count),
        "availability_percent": "",
        "available_station_ids": "",
        "missing_station_ids": "",
        "source_station_count": "",
        "source_station_ids": "",
        "untracked_station_ids": "",
        "source_url": entry["source_url"],
        "page_status": "",
        "parser_method": "",
    }


def parse_detail_row(
    entry: dict[str, str],
    html: str,
    expected_ids: set[str],
    run_time: dt.datetime,
) -> dict[str, str]:
    """Build one availability row from one SPC detail page."""
    row = _base_row(entry, run_time, len(expected_ids))
    source_ids, method = extract_all_station_ids(html)
    normalized_expected_ids = {station.upper() for station in expected_ids}
    available_ids = source_ids & normalized_expected_ids
    row["parser_method"] = method
    row["source_station_count"] = str(len(source_ids))
    row["source_station_ids"] = ";".join(sorted(source_ids))
    row["untracked_station_ids"] = ";".join(sorted(source_ids - normalized_expected_ids))
    if not _detail_is_valid(html):
        row["page_status"] = "invalid_detail_page"
        return row

    available_sorted = sorted(available_ids)
    missing_sorted = sorted(normalized_expected_ids - available_ids)
    row["available_count"] = str(len(available_sorted))
    row["availability_percent"] = f"{len(available_sorted) / len(expected_ids) * 100.0:.1f}" if expected_ids else "0.0"
    row["available_station_ids"] = ";".join(available_sorted)
    row["missing_station_ids"] = ";".join(missing_sorted)
    row["page_status"] = "ready_empty" if not available_ids and method == "detail_page_no_station_ids" else "ready"
    return row


def collect_archive_rows(
    index_html: str,
    expected_ids: set[str],
    *,
    run_time: dt.datetime | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Fetch and parse every archive detail page listed by the SPC index.

    The returned failure count is intentionally separate from the rows.  A
    partial index can still be published with explicit failed rows, while the
    runner receives a nonzero status and marks the optional source for review.
    """
    entries = discover_archive_entries(index_html)
    if not entries:
        raise RuntimeError("SPC archive index contained no *_OBS/ entries.")
    captured_at = run_time or dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    rows: list[dict[str, str]] = []
    failures = 0
    for entry in entries:
        try:
            detail_html = download_page(entry["source_url"])
        except RuntimeError as exc:
            row = _base_row(entry, captured_at, len(expected_ids))
            row["page_status"] = "detail_fetch_failed"
            row["parser_method"] = "detail_fetch_failed"
            rows.append(row)
            failures += 1
            print(f"WARNING: {exc}", file=sys.stderr)
            continue
        row = parse_detail_row(entry, detail_html, expected_ids, captured_at)
        if row["page_status"] not in {"ready", "ready_empty"}:
            failures += 1
            print(
                f"WARNING: SPC detail page {entry['source_url']} parsed as {row['page_status']}.",
                file=sys.stderr,
            )
        rows.append(row)
    return rows, failures


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.experimental_image_detect:
        print(
            "NOTE: --experimental-image-detect is retained for compatibility; HTML detail pages provide station IDs.",
            file=sys.stderr,
        )

    output = Path(args.output)
    run_time = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    try:
        expected_ids = expected_station_ids(Path(args.station_master))
        index_html = download_page()
        rows, failures = collect_archive_rows(index_html, expected_ids, run_time=run_time)
    except (RuntimeError, OSError, ValueError) as exc:
        # Keep the previous valid seven-day snapshot intact when the index
        # itself fails.  The optional source step will be marked failed by the
        # runner instead of silently converting a valid snapshot to empty.
        print(f"WARNING: {exc}", file=sys.stderr)
        print(f"SPC output retained: {output}")
        return 1

    write_rows(output, rows)
    ready = sum(row["page_status"] in {"ready", "ready_empty"} for row in rows)
    print(f"SPC archive periods discovered: {len(rows)}")
    print(f"SPC archive periods parsed: {ready} of {len(rows)}")
    print(f"SPC detail-page failures: {failures}")
    if rows:
        latest = rows[-1]
        print(
            f"SPC latest sounding: {latest['sounding_time_utc']} "
            f"({latest['available_count'] or '—'} of {latest['expected_count']} expected stations)"
        )
    print(f"SPC output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
