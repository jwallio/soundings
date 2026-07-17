from pathlib import Path

from scripts.build_upper_air_public_site import build_public_site


def test_public_site_builds_source_backed_standalone_page(tmp_path: Path) -> None:
    page = build_public_site(tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "CONUS Upper-Air Data Watch" in text
    assert "soundings.wall.cloud" in text
    assert "SOUNDING AVAILABILITY" in text and "Sounding availability trend" in text
    assert 'id="custom-range"' in text and "NWS Layoffs" in text
    assert "NCO reported for ingest" in text
    assert "CONUS RAOB Ingest" in text
    assert 'id="nco-heatmap"' in text
    assert 'id="nco-heatmap-custom"' in text
    assert 'id="nco-one-year"' in text and 'id="nco-custom-toggle"' in text
    assert 'id="nco-heatmap-custom" class="nco-heatmap-custom" hidden' in text
    assert 'id="nco-cell-detail"' in text and 'hidden>' in text
    assert "Average ingest" in text
    assert "Latest day:" in text
    assert "Mon" in text and "Wed" in text and "Fri" in text
    assert "ncoFormatDateDetail" in text
    assert "Healthy: 98 to 100 percent" in text
    assert "98 to 100 percent" in text
    assert "combined NCO ingest calendar" in text and "No data" in text
    assert all(label in text for label in ("7D", "14D", "30D", "90D"))
    assert "Sources and data health" in text
    assert "DATA HEALTH / SOURCE COVERAGE" not in text
    assert "Optional SPC feed unavailable" in text
    assert "NCO ingest counts" in text and "Jan 5, 2025" in text
    assert 'id="issue-custom-range"' in text
    assert "Download archive CSV" in text
    assert "Download station CSV" in text
    assert "Download NCO CSV" in text
    assert "archive-availability.csv" in text
    assert "latest-station-status.csv" in text
    assert "nco-ingest-history.csv" in text
    assert (tmp_path / "archive-availability.csv").is_file()
    assert (tmp_path / "latest-station-status.csv").is_file()
    assert (tmp_path / "nco-ingest-history.csv").is_file()
