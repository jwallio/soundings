"""Parse NCO SDM Administrative Messages for RAOB availability and issues."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup


NCO_MESSAGES_URL = "https://www.nco.ncep.noaa.gov/status/messages/"
IEM_AFOS_RETRIEVE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
AVAILABILITY_PATH = Path("data") / "nco_raob_availability.csv"
ISSUES_PATH = Path("data") / "nco_raob_station_issues.csv"
REQUEST_TIMEOUT_SECONDS = 60

AVAILABILITY_FIELDS = [
    "message_time_utc",
    "cycle_date_utc",
    "cycle_hour",
    "model",
    "alaskan_count",
    "canadian_count",
    "conus_count",
    "mexican_count",
    "caribbean_count",
    "pacific_count",
    "raw_message_id",
    "raw_message_excerpt",
    "message_hash",
]
ISSUE_FIELDS = [
    "message_time_utc",
    "cycle_date_utc",
    "cycle_hour",
    "model",
    "wmo_id",
    "station_id",
    "issue_text",
    "issue_category",
    "raw_message_id",
    "raw_message_excerpt",
    "message_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse NCO SDM RAOB messages.")
    parser.add_argument("--output-dir", default="data", help="Directory for CSV outputs.")
    parser.add_argument(
        "--source",
        choices=("nco", "iem"),
        default="nco",
        help="Message source: rolling NCO page (default) or IEM ADMSDM archive.",
    )
    parser.add_argument(
        "--start",
        help="IEM archive start date (YYYY-MM-DD, inclusive). Required with --source iem.",
    )
    parser.add_argument(
        "--end",
        help="IEM archive end date (YYYY-MM-DD, inclusive). Required with --source iem.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=9999,
        help="Maximum IEM messages to request (IEM only; 9999 is the archive endpoint maximum).",
    )
    parser.add_argument(
        "--archive-dir",
        default="data/raw/nco_admsdm",
        help="Directory for raw IEM text and request receipts (IEM only).",
    )
    return parser.parse_args()


def _parse_date(value: str, label: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}.") from exc


def download_messages(
    *,
    source: str = "nco",
    start: str | None = None,
    end: str | None = None,
    limit: int = 9999,
    archive_dir: str | Path | None = None,
) -> str:
    if source == "iem":
        if not start or not end:
            raise ValueError("--start and --end are required when --source iem is selected.")
        start_date = _parse_date(start, "--start")
        end_date = _parse_date(end, "--end")
        if end_date < start_date:
            raise ValueError("--end must be on or after --start.")
        if limit < 1:
            raise ValueError("--limit must be positive.")
        params = {
            "pil": "ADMSDM",
            "sdate": start_date.isoformat(),
            "edate": end_date.isoformat(),
            "fmt": "text",
            "limit": str(limit),
        }
        url = IEM_AFOS_RETRIEVE_URL
    else:
        params = None
        url = NCO_MESSAGES_URL
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "upper-air-network-monitor/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        label = "IEM ADMSDM archive" if source == "iem" else "NCO SDM messages"
        raise RuntimeError(f"Failed to download {label}: {exc}") from exc
    if source == "iem" and archive_dir and start and end:
        raw_dir = Path(archive_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        stem = f"admsdm_{start}_{end}"
        raw_path = raw_dir / f"{stem}.txt"
        receipt_path = raw_dir / f"{stem}.json"
        raw_path.write_text(response.text, encoding="utf-8")
        receipt_path.write_text(
            json.dumps(
                {
                    "source": "IEM ADMSDM archive",
                    "product": "ADMSDM",
                    "start_date_utc": start,
                    "end_date_utc": end,
                    "limit": limit,
                    "request_url": response.url,
                    "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "raw_file": raw_path.name,
                    "bytes": len(response.content),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return response.text


def normalize_text(html: str) -> str:
    # IEM's raw text response uses STX/ETX delimiters around products. They
    # are useful for transport, but should not become part of a parsed line.
    html = html.replace("\x01", "\n").replace("\x03", "\n").replace("\x00", "")
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return soup.get_text("\n")


def split_messages(text: str) -> list[str]:
    marker = "SENIOR DUTY METEOROLOGIST"
    chunks = []
    for chunk in text.split(marker):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunks.append(marker + "\n" + chunk)
    return chunks


def parse_message_time(message: str) -> dt.datetime | None:
    match = re.search(
        r"\b(\d{4})Z\s+\w{3}\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\b",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    hhmm, month_text, day_text, year_text = match.groups()
    month = dt.datetime.strptime(month_text.upper(), "%b").month
    return dt.datetime(
        int(year_text),
        month,
        int(day_text),
        int(hhmm[:2]),
        int(hhmm[2:]),
        tzinfo=dt.timezone.utc,
    )


def message_hash(message: str) -> str:
    normalized = "\n".join(line.rstrip() for line in message.splitlines()).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:16]


def compact_excerpt(text: str, limit: int = 600) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def cycle_date_for(message_time: dt.datetime, cycle_hour: int) -> dt.date:
    # SDM RAOB recaps normally post a few hours after the 00Z/12Z cycle.
    if cycle_hour == 0 and message_time.hour >= 18:
        return (message_time + dt.timedelta(days=1)).date()
    if cycle_hour == 12 and message_time.hour < 6:
        return (message_time - dt.timedelta(days=1)).date()
    return message_time.date()


def parse_availability(message: str, message_time: dt.datetime, digest: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Historical ADMSDM wording varies substantially while retaining the same
    # labeled-count structure. Match the CONUS count and parse the other
    # regional labels from the same sentence, allowing "stations" and an
    # omitted Pacific count in earlier products.
    pattern = re.compile(
        r"(\d+)\s+CONUS\b.*?(?:raobs?|stations?)\s+available\s+for\s+ingest",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(message):
        sentence_start = max(0, match.start() - 360)
        sentence = message[sentence_start : match.end()]
        label_counts: dict[str, str] = {}
        for label in ("Alaskan", "Canadian", "Mexican", "Caribbean", "Pacific"):
            label_matches = list(re.finditer(rf"(\d+)\s+{label}\b", sentence, flags=re.IGNORECASE))
            if label_matches:
                label_counts[label.lower()] = label_matches[-1].group(1)

        cycle_matches = list(re.finditer(r"\b(00|12)Z\b", message[sentence_start : match.start()], flags=re.IGNORECASE))
        cycle_hour = int(cycle_matches[-1].group(1)) if cycle_matches else (0 if message_time.hour < 6 else 12)
        before = message[sentence_start : match.start()]
        after = message[match.end() : match.end() + 240]
        model_matches = list(re.finditer(r"\b(NAM|GFS)\b", before, flags=re.IGNORECASE))
        model_after = re.search(r"\b(?:00|12)Z\s+(?:UPDATED\s+)?(NAM|GFS)\b", after, flags=re.IGNORECASE)
        if model_after:
            model = model_after.group(1).upper()
        elif model_matches:
            model = model_matches[-1].group(1).upper()
        else:
            # Generic NCEP production-suite notices are intentionally kept as
            # their own series; assigning them to NAM/GFS would manufacture
            # model-specific history that the message did not report.
            model = "NCEP"

        conus_count = int(match.group(1))
        if conus_count < 0 or conus_count > 100:
            print(
                f"WARNING: Skipping impossible NCO CONUS count {conus_count} in {digest}.",
                file=sys.stderr,
            )
            continue
        rows.append(
            {
                "message_time_utc": message_time.isoformat().replace("+00:00", "Z"),
                "cycle_date_utc": cycle_date_for(message_time, cycle_hour).isoformat(),
                "cycle_hour": f"{cycle_hour:02d}",
                "model": model,
                "alaskan_count": label_counts.get("alaskan", ""),
                "canadian_count": label_counts.get("canadian", ""),
                "conus_count": str(conus_count),
                "mexican_count": label_counts.get("mexican", ""),
                "caribbean_count": label_counts.get("caribbean", ""),
                "pacific_count": label_counts.get("pacific", ""),
                "raw_message_id": digest,
                "raw_message_excerpt": compact_excerpt(match.group(0)),
                "message_hash": digest,
            }
        )
    return rows


def issue_category(text: str) -> str:
    lowered = text.lower()
    if "no report" in lowered:
        return "no_report"
    if "missing" in lowered:
        return "missing_parts"
    if "short to" in lowered:
        return "short_sounding"
    if "purged" in lowered:
        return "purged_data"
    if "unavailable" in lowered or "10159" in lowered:
        return "unavailable"
    if "equipment failure" in lowered or "10142" in lowered:
        return "equipment_failure"
    return "other"


def parse_issue_blocks(message: str, message_time: dt.datetime, digest: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = [line.rstrip() for line in message.splitlines()]
    active_cycle: int | None = None
    active_model = ""
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            current["issue_text"] = " ".join(current["issue_text"].split())
            current["issue_category"] = issue_category(current["issue_text"])
            rows.append(current)
            current = None

    recap_pattern = re.compile(r"\b(\d{2})Z\s+(?:(NAM|GFS)\s+)?(?:UPDATED\s+)?RAOB RECAP", re.I)
    issue_pattern = re.compile(r"^\s*(\d{5})/([A-Z0-9]{3})\s+-\s+(.+?)\s*$")

    for line in lines:
        recap_match = recap_pattern.search(line)
        if recap_match:
            flush()
            active_cycle = int(recap_match.group(1))
            model = recap_match.group(2)
            if model:
                active_model = model.upper()
            elif "UPDATED" in line.upper():
                active_model = "UPDATED"
            continue

        issue_match = issue_pattern.match(line)
        if issue_match and active_cycle is not None:
            flush()
            wmo_id, station_id, issue_text = issue_match.groups()
            current = {
                "message_time_utc": message_time.isoformat().replace("+00:00", "Z"),
                "cycle_date_utc": cycle_date_for(message_time, active_cycle).isoformat(),
                "cycle_hour": f"{active_cycle:02d}",
                "model": active_model or "UNKNOWN",
                "wmo_id": wmo_id,
                "station_id": station_id,
                "issue_text": issue_text.strip(),
                "issue_category": "",
                "raw_message_id": digest,
                "raw_message_excerpt": compact_excerpt(message),
                "message_hash": digest,
            }
            continue

        if current and line.startswith((" ", "\t")) and line.strip():
            current["issue_text"] += " " + line.strip()

    flush()
    return rows


def parse_messages(html: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    text = normalize_text(html)
    availability_rows: list[dict[str, str]] = []
    issue_rows: list[dict[str, str]] = []
    wanted = (
        "RAOB RECAP",
        "raobs available for ingest",
        "stations available for ingest",
        "00Z NAM",
        "12Z NAM",
        "00Z GFS",
        "12Z GFS",
        "UPDATED NAM",
        "UPDATED GFS",
        "UPDATED RAOB",
    )
    for message in split_messages(text):
        if not any(token.lower() in message.lower() for token in wanted):
            continue
        message_time = parse_message_time(message)
        if not message_time:
            continue
        digest = message_hash(message)
        availability_rows.extend(parse_availability(message, message_time, digest))
        issue_rows.extend(parse_issue_blocks(message, message_time, digest))
    return availability_rows, issue_rows


def read_existing(path: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({field: str(row.get(field, "")) for field in fieldnames})
        return rows


def write_deduped(
    path: Path,
    fieldnames: list[str],
    existing: list[dict[str, str]],
    new: list[dict[str, str]],
    key_fields: list[str],
) -> list[dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = existing + new
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in combined:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        normalized = {field: str(row.get(field, "")) for field in fieldnames}
        if not normalized.get("raw_message_id") and normalized.get("message_hash"):
            normalized["raw_message_id"] = normalized["message_hash"]
        previous = unique.get(key)
        if (
            previous is None
            or normalized.get("message_time_utc", "") > previous.get("message_time_utc", "")
            or (not previous.get("raw_message_excerpt") and normalized.get("raw_message_excerpt"))
        ):
            unique[key] = normalized
    rows = sorted(
        unique.values(),
        key=lambda row: (
            row.get("cycle_date_utc", ""),
            row.get("cycle_hour", ""),
            row.get("message_time_utc", ""),
            row.get("model", ""),
            row.get("wmo_id", ""),
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    availability_path = outdir / AVAILABILITY_PATH.name
    issues_path = outdir / ISSUES_PATH.name

    try:
        html = download_messages(
            source=args.source,
            start=args.start,
            end=args.end,
            limit=args.limit,
            archive_dir=args.archive_dir if args.source == "iem" else None,
        )
        availability_new, issues_new = parse_messages(html)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not availability_new and not issues_new:
        print("WARNING: No RAOB availability or issue rows parsed from the selected message source.", file=sys.stderr)

    availability = write_deduped(
        availability_path,
        AVAILABILITY_FIELDS,
        read_existing(availability_path, AVAILABILITY_FIELDS),
        availability_new,
        ["cycle_date_utc", "cycle_hour", "model"],
    )
    issues = write_deduped(
        issues_path,
        ISSUE_FIELDS,
        read_existing(issues_path, ISSUE_FIELDS),
        issues_new,
        ["cycle_date_utc", "cycle_hour", "model", "wmo_id", "station_id", "issue_text"],
    )

    latest = availability[-1] if availability else None
    latest_issues = [
        row for row in issues
        if latest
        and row["cycle_date_utc"] == latest["cycle_date_utc"]
        and row["cycle_hour"] == latest["cycle_hour"]
    ]
    issue_stations = ", ".join(sorted({row["station_id"] for row in latest_issues})) or "none"
    if latest:
        print(
            f"Latest NCO cycle: {latest['cycle_date_utc']} {latest['cycle_hour']}Z {latest['model']}"
        )
        print(f"Latest NCO CONUS RAOB count: {latest['conus_count']}")
    else:
        print("Latest NCO cycle: unavailable")
        print("Latest NCO CONUS RAOB count: unavailable")
    print(f"Latest NCO issue stations: {issue_stations}")
    print(f"Message source: {args.source}")
    print(f"Availability output: {availability_path}")
    print(f"Station issues output: {issues_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
