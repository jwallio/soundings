from __future__ import annotations

import json

import pandas as pd

from upper_air_network_monitor.social_graphics import (
    MonitorInputs,
    OUTPUT_FILENAMES,
    build_social_package,
    calculate_metrics,
    detect_sharp_drop,
    _miller_xy,
    render_social_graphics_from_manifest,
)


def _igra_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "year": 2026,
            "launches": 100.0,
            "launches_7d_avg": 100.0,
            "baseline_5yr_avg": 120.0,
            "percent_vs_baseline": -16.7,
        }
    )
    data.loc[data.index[-1], ["launches", "launches_7d_avg"]] = [40.0, 95.0]
    historical = []
    for year, value in zip((2021, 2022, 2023, 2024), (112.0, 116.0, 120.0, 124.0)):
        prior_dates = pd.date_range(f"{year}-01-01", periods=20, freq="D")
        historical.append(
            pd.DataFrame(
                {
                    "date": prior_dates,
                    "year": year,
                    "launches": value,
                    "launches_7d_avg": value,
                    "baseline_5yr_avg": 120.0,
                    "percent_vs_baseline": (value - 120.0) / 120.0 * 100.0,
                }
            )
        )
    return pd.concat([*historical, data], ignore_index=True)


def _inputs(*, with_nco: bool = True, with_stations: bool = True) -> MonitorInputs:
    nco = pd.DataFrame()
    issues = pd.DataFrame()
    if with_nco:
        nco = pd.DataFrame(
            {
                "cycle_date_utc": ["2026-01-19"],
                "cycle_hour": ["12"],
                "model": ["NAM"],
                "conus_count": [37],
                "message_time_utc": ["2026-01-19T13:00:00Z"],
            }
        )
        issues = pd.DataFrame(
            {
                "cycle_date_utc": ["2026-01-19"],
                "cycle_hour": ["12"],
                "station_id": ["AAA"],
                "issue_category": ["no_report"],
            }
        )
    stations = pd.DataFrame()
    if with_stations:
        stations = pd.DataFrame(
            {
                "station_id": ["AAA", "BBB"],
                "active_expected": ["true", "true"],
                "latitude": [35.0, 40.0],
                "longitude": [-97.0, -90.0],
            }
        )
    return MonitorInputs(igra=_igra_frame(), nco=nco, issues=issues, stations=stations, input_paths={})


def test_miller_projection_preserves_longitude_and_orders_latitude() -> None:
    x, y = _miller_xy([-100.0, -100.0], [25.0, 50.0])
    assert x[0] == x[1]
    assert y[1] > y[0]


def test_sharp_drop_prefers_the_most_recent_july() -> None:
    series = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01", "2025-07-02", "2026-07-01", "2026-07-02"]),
            "observed": [130.0, 110.0, 125.0, 118.0],
        }
    )
    result = detect_sharp_drop(series)
    assert result is not None
    date, change = result
    assert date == pd.Timestamp("2026-07-02")
    assert change == -7.0


def test_metrics_exclude_partial_latest_archive_day() -> None:
    metrics = calculate_metrics(_inputs())
    assert metrics.latest_complete is not None
    assert metrics.latest_complete["date"].date().isoformat() == "2026-01-19"
    assert metrics.partial_date is not None
    assert metrics.partial_date.date().isoformat() == "2026-01-20"
    assert metrics.impacted_station_count == 1


def test_metrics_handle_missing_optional_nco_and_station_inputs() -> None:
    metrics = calculate_metrics(_inputs(with_nco=False, with_stations=False))
    assert metrics.latest_nco is None
    assert metrics.nco_count is None
    assert metrics.station_statuses.empty
    assert metrics.impacted_station_count == 0


def test_social_package_writes_images_and_manifest(tmp_path) -> None:
    metrics, paths, manifest_path = build_social_package(_inputs(), tmp_path, dpi_scale=0.1)
    assert metrics.latest_complete is not None
    assert len(paths) == len(OUTPUT_FILENAMES) == 9
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 2
    assert manifest["latest_complete_archive_date"] == "2026-01-19"
    assert manifest["excluded_incomplete_archive_date"] == "2026-01-20"
    assert manifest["kpis"]["reported_operational_issue_statuses"] == 1
    assert manifest["time_series"]["date"]
    assert len(manifest["time_series"]["date"]) == 19
    assert len(manifest["time_series"]["baseline_range_low"]) == 19
    assert manifest["time_series"]["baseline_range_low"][0] == 112.0
    assert manifest["time_series"]["baseline_range_high"][0] == 124.0
    assert manifest["station_statuses"]
    assert len(manifest["output_image_paths"]) == 9


def test_manifest_only_render_gracefully_handles_missing_optional_arrays(tmp_path) -> None:
    manifest_path = tmp_path / "legacy_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "latest_complete_archive_date": "2026-01-19",
                "kpis": {
                    "seven_day_archive_percent_difference": -9.6,
                    "recent_windows": [{"days": 30, "percent_difference": -4.9}],
                },
            }
        ),
        encoding="utf-8",
    )
    payload, paths = render_social_graphics_from_manifest(manifest_path, tmp_path / "rendered", dpi_scale=0.1)
    assert payload.series.empty
    assert len(paths) == 9
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
