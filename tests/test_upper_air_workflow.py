from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "upper-air-pages.yml"


def test_scheduled_refreshes_use_distinct_full_archive_cron() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '- cron: "45 1,3,13 * * *"' in text
    assert '- cron: "45 15 * * *"' in text
    assert 'github.event.schedule }}" == "45 15 * * *"' in text


def test_nco_only_runs_restore_data_and_skip_archive_sources() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/cache@v4" in text
    assert "outputs/conus_balloon_launches_by_year_daily.csv" in text
    assert "--skip-igra --skip-station-master" in text
    assert "Retained IGRA/station data is unavailable" in text
