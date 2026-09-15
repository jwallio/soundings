from __future__ import annotations

import datetime as dt

import pytest

import scripts.parse_spc_sounding_page as parser


INDEX_HTML = """
<html><body>
  <a href="/exper/soundings/26091518_OBS/"><img alt="09/15/2026 1800 UTC"></a>
  <a href="/exper/soundings/26091518_OBS/">09/15/2026 1800 UTC</a>
  <a href="/exper/soundings/26091512_OBS/">09/15/2026 1200 UTC</a>
  <a href="/exper/soundings/26091506_OBS/">09/15/2026 0600 UTC</a>
</body></html>
"""


def detail_html(*stations: str) -> str:
    areas = "\n".join(
        f'<area href=javascript:show_soundings("{station}") alt="{station}">' for station in stations
    )
    return f"""
    <html><head><title>SPC Sounding Analysis Page</title></head>
    <body><span>Observed Radiosonde Data<br>09/15/2026 18 UTC</span>
    <map name="stations">{areas}</map></body></html>
    """


def test_discover_archive_entries_deduplicates_and_keeps_supplemental_hours() -> None:
    entries = parser.discover_archive_entries(INDEX_HTML)

    assert [entry["cycle_hour"] for entry in entries] == ["06", "12", "18"]
    assert entries[-1]["sounding_time_utc"] == "2026-09-15T18:00:00Z"
    assert entries[-1]["source_url"].endswith("26091518_OBS/")


def test_extract_station_ids_reads_spc_detail_image_map() -> None:
    available, method = parser.extract_station_ids(
        detail_html("CRP", "OAX", "KOUN", "ZZZ"),
        {"CRP", "OAX", "OUN"},
    )

    assert available == {"CRP", "OAX", "OUN"}
    assert method == "detail_map_area"


def test_parse_detail_row_preserves_source_stations_outside_tracker_inventory() -> None:
    entry = parser.discover_archive_entries(INDEX_HTML)[-1]
    row = parser.parse_detail_row(
        entry,
        detail_html("CRP", "OAX", "KOUN", "ZZZ"),
        {"CRP", "OAX", "OUN"},
        dt.datetime(2026, 9, 15, 19, 0, tzinfo=dt.timezone.utc),
    )

    assert row["available_count"] == "3"
    assert row["source_station_count"] == "4"
    assert row["source_station_ids"] == "CRP;OAX;OUN;ZZZ"
    assert row["untracked_station_ids"] == "ZZZ"


def test_collect_archive_rows_fetches_every_archive_period(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {
        "https://www.spc.noaa.gov/exper/soundings/26091506_OBS/": detail_html("CRP"),
        "https://www.spc.noaa.gov/exper/soundings/26091512_OBS/": detail_html("OAX", "OUN"),
        "https://www.spc.noaa.gov/exper/soundings/26091518_OBS/": detail_html("CRP", "OAX", "OUN"),
    }
    calls: list[str] = []

    def fake_download(url: str) -> str:
        calls.append(url)
        return pages[url]

    monkeypatch.setattr(parser, "download_page", fake_download)
    rows, failures = parser.collect_archive_rows(
        INDEX_HTML,
        {"CRP", "OAX", "OUN", "VEF"},
        run_time=dt.datetime(2026, 9, 15, 19, 0, tzinfo=dt.timezone.utc),
    )

    assert failures == 0
    assert calls == list(pages)
    assert [row["cycle_hour"] for row in rows] == ["06", "12", "18"]
    assert rows[-1]["page_status"] == "ready"
    assert rows[-1]["available_count"] == "3"
    assert rows[-1]["availability_percent"] == "75.0"
    assert rows[-1]["available_station_ids"] == "CRP;OAX;OUN"
    assert rows[-1]["missing_station_ids"] == "VEF"


def test_collect_archive_rows_marks_detail_failure_without_discarding_other_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_url = "https://www.spc.noaa.gov/exper/soundings/26091518_OBS/"

    def fake_download(url: str) -> str:
        if url == good_url:
            return detail_html("CRP")
        raise RuntimeError("temporary upstream failure")

    monkeypatch.setattr(parser, "download_page", fake_download)
    rows, failures = parser.collect_archive_rows(
        INDEX_HTML,
        {"CRP", "OAX"},
        run_time=dt.datetime(2026, 9, 15, 19, 0, tzinfo=dt.timezone.utc),
    )

    assert failures == 2
    assert len(rows) == 3
    assert rows[0]["page_status"] == "detail_fetch_failed"
    assert rows[-1]["page_status"] == "ready"
