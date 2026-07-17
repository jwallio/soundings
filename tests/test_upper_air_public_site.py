from pathlib import Path

from scripts.build_upper_air_public_site import build_public_site


def test_public_site_builds_source_backed_standalone_page(tmp_path: Path) -> None:
    page = build_public_site(tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "CONUS Upper-Air Data Watch" in text
    assert "soundings.wall.cloud" in text
    assert "Data-availability diagnostic only" not in text
    assert "SOUNDING AVAILABILITY" in text
    assert "Sounding availability trend" in text
    assert "Archive availability trend" not in text
    for label in ("6MO", "10D", "30D", "60D", "90D", "1YR", "2YR"):
        assert f">{label}<" in text
    assert 'id="custom-range"' in text
    assert "Full Y scale" in text
    assert "NWS Layoffs" in text
    assert "2025-01-01" in text
    assert "2025-04-20" in text
    assert "Difference from baseline" not in text
    assert "Event maximums" in text
    assert "Government Actions" not in text
    assert "NWS workforce events" not in text
    assert "Smoothed observed (30D)" not in text
    assert "Smoothed total (14D)" in text
    assert "Smoothed line: 7 days" in text
    assert "Diamonds label same-date historical event maximums" in text
    assert "OMB/OPM RIF directive" not in text
    assert "DOGE contract list" not in text
    assert "NOAA position directives" not in text
    assert "NWS RAOB reductions" in text
    assert "1,127 targeted federal contracts" not in text
    assert "1,029 slots" not in text
    for window in (7, 14, 30, 60, 90, 180, 360):
        assert f"{window} days" in text
    assert "Largest recent 7-day decline" not in text
    assert "READ THIS FIRST" not in text
    assert "Stations ranked by 90-day archive shortfall" in text
    assert "Stations ranked by 90-day archive surplus" not in text
    assert 'id="station-surpluses"' not in text
    assert 'id="station-shortfalls"' in text
    assert "Search all mapped stations" in text
    assert 'id="station-search"' in text
    assert "Current NCO-reported issues" in text
    assert "Where NCO reported issues" not in text
    assert "Stations with an NCO-reported issue" in text
    assert "Miller-projection map" in text
    assert "NCO reported for ingest" in text
    assert "DATA HEALTH" in text
    assert "NCO ingest counts" in text
    assert "Jan 05, 2025" in text
    assert "What NCO reported for ingest" not in text
    assert 'id="nco-custom-range"' in text
    assert 'id="issue-custom-range"' in text
    assert "Default view: latest 1 year" in text
    assert 'class="nco-range-button" data-days="28">1MO</button>' in text
    assert 'class="nco-range-button active" data-days="365"' in text
    assert "Default view: latest 28 days" in text
    assert 'class="snapshot-inline"' not in text
    assert 'class="snapshot"' not in text
    assert ".station-search-details[open]" in text
    assert 'class="card review-card"' in text
    assert "Hurricane Ian" in text
    assert "not a confirmed cause" not in text
    assert 'min="2021-01-01"' in text
    assert "Download archive CSV" in text
    assert (tmp_path / "og.png").is_file()
    assert (tmp_path / "archive-availability.csv").is_file()
    assert (tmp_path / "latest-station-status.csv").is_file()
    assert (tmp_path / "nco-ingest-history.csv").is_file()
