from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.build_upper_air_public_site import _nco_freshness, build_public_site
from scripts.run_upper_air_monitor import _record_source_step


def test_public_site_builds_source_backed_standalone_page(tmp_path: Path) -> None:
    page = build_public_site(tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "CONUS Upper-Air Data Watch" in text
    assert "soundings.wall.cloud" in text
    assert "SOUNDING AVAILABILITY" in text and "Sounding availability trend" in text
    assert 'id="custom-range"' in text and "NWS Layoffs" in text
    assert "NCO reported for ingest" in text
    assert "CONUS RAOB Ingest" in text
    assert "Stations ranked by 90-day archive surplus" in text
    assert 'id="station-surpluses"' in text
    assert 'id="nco-heatmap"' in text
    assert 'id="nco-heatmap-custom"' in text
    assert 'id="nco-one-year"' in text and 'id="nco-custom-toggle"' in text
    assert 'id="nco-heatmap-custom" class="nco-heatmap-custom" hidden' in text
    assert 'id="nco-cell-detail"' in text and 'hidden>' in text
    assert "Average ingest" in text
    assert "Latest day:" in text
    assert 'aria-label="Weekdays"' in text
    assert all(f">{label}<" in text for label in ("M", "T", "W", "Th", "F", "Sa", "Su"))
    assert "ncoFormatDateDetail" in text
    assert "Healthy: 98 to 100 percent" in text
    assert "98 to 100 percent" in text
    assert "combined NCO operational-message ingest calendar" in text and "No data" in text
    assert "NCO operational-message availability" in text
    assert all(label in text for label in (">Combined<", ">00Z<", ">12Z<"))
    assert "Latest source record:" in text and "Last successful NCO refresh:" in text
    assert all(label in text for label in ("7D", "14D", "30D", "90D"))
    assert "Sources and data health" in text
    assert "Built " in text and "Data snapshot" in text
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
    assert (tmp_path / "og.png").is_file()


def test_nco_freshness_marks_retained_data_stale() -> None:
    snapshot = SimpleNamespace(
        source_status=pd.DataFrame(
            [{
                "source": "NCO availability",
                "coverage_end_utc": "2026-07-20T00:00:00Z",
                "modified_utc": "2026-07-21T15:00:00Z",
            }]
        ),
        refresh_status={
            "sources": {
                "nco": {
                    "status": "failed_retained",
                    "last_successful_fetch_utc": "2026-07-21T15:00:00Z",
                    "latest_successful_record_date": "2026-07-20",
                }
            }
        },
    )
    latest, refresh, stale = _nco_freshness(snapshot)
    assert latest == "Jul 20, 2026"
    assert "Last successful NCO refresh" in refresh
    assert stale is True


def test_failed_nco_refresh_retains_source_record_metadata(tmp_path: Path) -> None:
    nco_path = tmp_path / "nco_raob_availability.csv"
    pd.DataFrame([{"cycle_date_utc": "2026-07-16"}]).to_csv(nco_path, index=False)
    status = {
        "run_started_at_utc": "2026-07-17T02:00:00Z",
        "sources": {"nco": {"latest_successful_record_date": "2026-07-15"}},
    }
    _record_source_step(status, "nco", 1, "HTTP 500 from upstream", nco_path, "cycle_date_utc")
    assert status["sources"]["nco"]["status"] == "failed_retained"
    assert status["sources"]["nco"]["error_kind"] == "upstream_fetch"
    assert status["sources"]["nco"]["latest_successful_record_date"] == "2026-07-15"
    assert "HTTP 500" in status["sources"]["nco"]["last_error"]
