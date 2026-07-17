from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from upper_air_network_monitor.dashboard_charts import (
    archive_trend_figure,
    archive_windows_figure,
    issue_category_figure,
    nco_ingest_figure,
    station_archive_shortfall_figure,
    station_archive_surplus_figure,
    station_status_map_figure,
)
from upper_air_network_monitor.dashboard_data import (
    archive_detail_series_from_igra,
    archive_window_metrics,
    enrich_archive_variability,
    latest_issue_rows,
    latest_and_previous_comparable_nco,
    issue_counts_by_cycle,
    load_dashboard_snapshot,
    prepare_issues,
    prepare_nco,
    prepare_stations,
    station_issue_changes,
    station_status_frame,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="D"),
            "daily": [100.0] * 12,
            "observed": [100.0, 101.0, 102.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 92.0, 88.0],
            "baseline": [110.0] * 12,
        }
    )


def _nco() -> pd.DataFrame:
    return prepare_nco(
        pd.DataFrame(
            {
                "cycle_date_utc": ["2026-01-10", "2026-01-11"],
                "cycle_hour": ["00", "12"],
                "model": ["GFS", "NAM"],
                "conus_count": [39, 37],
                "message_time_utc": ["2026-01-10T01:00:00Z", "2026-01-11T13:00:00Z"],
            }
        )
    )


def _issues() -> pd.DataFrame:
    return prepare_issues(
        pd.DataFrame(
            {
                "cycle_date_utc": ["2026-01-10", "2026-01-11"],
                "cycle_hour": ["00", "12"],
                "model": ["GFS", "NAM"],
                "station_id": ["AAA", "BBB"],
                "issue_category": ["no_report", "equipment_failure"],
                "issue_text": ["No report", "Equipment failure"],
            }
        )
    )


def _stations() -> pd.DataFrame:
    return prepare_stations(
        pd.DataFrame(
            {
                "station_id": ["AAA", "BBB", "OUT"],
                "station_name": ["Alpha", "Bravo", "Outside"],
                "state": ["OK", "KS", "AK"],
                "latitude": [35.0, 39.0, 61.0],
                "longitude": [-97.0, -98.0, -150.0],
                "active_expected": [True, True, True],
            }
        )
    )


def _comparable_nco() -> pd.DataFrame:
    return prepare_nco(
        pd.DataFrame(
            {
                "cycle_date_utc": ["2026-01-10", "2026-01-11"],
                "cycle_hour": ["12", "12"],
                "model": ["GFS", "GFS"],
                "conus_count": [38, 36],
                "message_time_utc": ["2026-01-10T13:00:00Z", "2026-01-11T13:00:00Z"],
            }
        )
    )


def _transition_issues() -> pd.DataFrame:
    return prepare_issues(
        pd.DataFrame(
            {
                "cycle_date_utc": ["2026-01-10", "2026-01-10", "2026-01-11", "2026-01-11"],
                "cycle_hour": ["12", "12", "12", "12"],
                "model": ["GFS", "GFS", "GFS", "GFS"],
                "station_id": ["AAA", "BBB", "BBB", "CCC"],
                "issue_category": ["no_report", "equipment_failure", "equipment_failure", "no_report"],
                "issue_text": ["No report", "Equipment failure", "Equipment failure", "No report"],
            }
        )
    )


def test_archive_window_metrics_use_selected_end_date() -> None:
    windows = archive_window_metrics(_series(), days=(7, 30))
    seven = windows[windows["days"].eq(7)].iloc[0]
    assert seven["observed"] == 700.0
    assert seven["expected"] == 770.0
    assert round(float(seven["percent"]), 1) == -9.1
    assert int(windows[windows["days"].eq(30)].iloc[0]["observed"]) == 1200


def test_latest_issue_rows_and_station_status_follow_latest_selected_cycle() -> None:
    latest = latest_issue_rows(_issues(), _nco())
    assert latest["station_id"].tolist() == ["BBB"]
    statuses = station_status_frame(_stations(), latest)
    assert len(statuses) == 2
    assert statuses.set_index("station_id").loc["BBB", "status"] == "NCO-reported issue"
    assert statuses.set_index("station_id").loc["AAA", "status"] == "No issue reported"


def test_archive_variability_uses_2021_2024_same_date_range() -> None:
    dates = pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"])
    igra = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "launches_7d_avg": [100.0, 104.0, 108.0, 112.0, 150.0],
        }
    )
    series = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "observed": [98.0], "baseline": [106.0]})
    enriched = enrich_archive_variability(series, igra)
    assert enriched.iloc[0]["baseline_low"] == 100.0
    assert enriched.iloc[0]["baseline_high"] == 112.0
    assert enriched.iloc[0]["baseline_year_count"] == 4


def test_archive_detail_series_starts_in_january_2021_and_stops_at_latest_complete_date() -> None:
    igra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-31", "2021-01-01", "2026-07-12", "2026-07-13"]),
            "launches": [100, 101, 102, 20],
            "launches_7d_avg": [100.0, 101.0, 102.0, 90.0],
            "baseline_5yr_avg": [float("nan"), float("nan"), 111.0, 111.0],
        }
    )
    series = archive_detail_series_from_igra(igra, "2026-07-12")
    assert series["date"].dt.strftime("%Y-%m-%d").tolist() == ["2021-01-01", "2026-07-12"]
    assert series["daily"].tolist() == [101, 102]


def test_nco_transition_diagnostics_compare_same_model_and_cycle_hour() -> None:
    latest, previous = latest_and_previous_comparable_nco(_comparable_nco())
    assert latest is not None and previous is not None
    assert latest["conus_count"] == 36
    assert previous["conus_count"] == 38
    changes = station_issue_changes(_transition_issues(), _comparable_nco()).set_index("station_id")
    assert changes.loc["AAA", "transition"] == "Resolved"
    assert changes.loc["BBB", "transition"] == "Persistent"
    assert changes.loc["CCC", "transition"] == "New issue"


def test_plotly_builders_render_real_chart_structures() -> None:
    windows = archive_window_metrics(_series(), days=(7, 30, 60, 90))
    statuses = station_status_frame(_stations(), latest_issue_rows(_issues(), _nco()))
    assert len(archive_trend_figure(_series()).data) >= 4
    assert len(archive_windows_figure(windows).data) == 1
    vertical_windows = archive_windows_figure(windows, vertical=True)
    assert vertical_windows.data[0].orientation is None
    assert list(vertical_windows.data[0].x) == ["90 days", "60 days", "30 days", "7 days"]
    assert len(nco_ingest_figure(_nco()).data) == 2
    assert all(trace.type == "scatter" for trace in nco_ingest_figure(_nco()).data)
    assert all("(7D)" in str(trace.name) for trace in nco_ingest_figure(_nco()).data)
    smoothed_nco = nco_ingest_figure(_nco(), show_smoothed_trend=True)
    assert any(str(trace.name).startswith("GFS smoothed") for trace in smoothed_nco.data)
    assert all(trace.line.dash == "solid" for trace in smoothed_nco.data if "smoothed" in str(trace.name))
    issue_counts = issue_counts_by_cycle(_issues(), 14)
    smoothed_issues = issue_category_figure(issue_counts, show_smoothed_trend=True)
    assert any(trace.name == "Smoothed total (14D)" for trace in smoothed_issues.data)
    assert next(trace for trace in smoothed_issues.data if trace.name == "Smoothed total (14D)").line.dash == "solid"
    station_map = station_status_map_figure(statuses)
    assert len(station_map.data) == 2
    assert station_map.layout.geo.showsubunits is True

    enriched = _series().assign(baseline_low=105.0, baseline_high=115.0)
    trend = archive_trend_figure(enriched)
    assert "2021–2024 historical range" in [trace.name for trace in trend.data]
    assert len(set(archive_windows_figure(windows).data[0].marker.color)) == 1

    percent_trend = archive_trend_figure(enriched, show_percent_axis=True)
    observed_trace = next(trace for trace in percent_trend.data if trace.name == "Observed 7-day average")
    assert not any(trace.name == "Difference from baseline" for trace in percent_trend.data)
    assert not hasattr(percent_trend.layout, "yaxis2") or percent_trend.layout.yaxis2 is None
    assert "Difference: %{customdata[1]:+.1f}%" in observed_trace.hovertemplate

    event_series = pd.DataFrame(
        {
            "date": pd.date_range("2022-09-24", periods=7, freq="D"),
            "daily": [183.0, 226.0, 228.0, 227.0, 218.0, 185.0, 201.0],
            "observed": [136.7, 150.4, 164.7, 179.1, 191.6, 199.5, 200.1],
            "baseline": [127.6, 127.6, 127.4, 127.4, 127.4, 127.3, 127.3],
        }
    )
    event_trend = archive_trend_figure(event_series, show_event_tags=True)
    event_trace = next(trace for trace in event_trend.data if trace.name == "Event maximums")
    event_points = dict(zip(event_trace.text, zip(event_trace.x, event_trace.y)))
    assert event_points["Hurricane Ian"] == (pd.Timestamp("2022-09-30"), 200.1)
    assert event_points["Hurricane Helene"] == (pd.Timestamp("2022-09-26"), 164.7)
    assert event_trace.marker.symbol == "diamond"
    assert event_trace.mode == "markers+text"
    assert not any("Hurricane Ian" in str(annotation.text) for annotation in event_trend.layout.annotations)

    workforce_series = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-02-13", "2025-02-17", "2025-03-13", "2025-03-20"]),
            "daily": [120.0, 119.0, 118.0, 117.0],
            "observed": [120.0, 119.0, 118.0, 117.0],
            "baseline": [125.0, 125.0, 125.0, 125.0],
        }
    )
    workforce_trend = archive_trend_figure(workforce_series, show_workforce_events=True)
    workforce_traces = [trace for trace in workforce_trend.data if trace.name == "NWS RAOB cuts"]
    assert len(workforce_traces) == 1
    assert all(trace.mode == "lines+text" for trace in workforce_traces)
    assert all(trace.line.dash == "dot" for trace in workforce_traces)
    assert all(trace.line.color == "#FF8A3D" for trace in workforce_traces)
    assert all(trace.showlegend is False for trace in workforce_traces)
    assert all(len(trace.x) == 3 and trace.x[0] == trace.x[1] == trace.x[2] for trace in workforce_traces)
    assert [trace.text[1] for trace in workforce_traces] == ["NWS RAOB cuts"]
    assert all(trace.text[0] == "" and trace.text[2] == "" for trace in workforce_traces)
    assert all("timeline context only" in str(row[1]).lower() for trace in workforce_traces for row in trace.customdata)
    assert next(trace for trace in event_trend.data if trace.name == "Event maximums").marker.color == "#F15BB5"
    assert not any(trace.name == "Government Actions" for trace in workforce_trend.data)
    assert all("not a confirmed cause" not in str(trace.hovertemplate) for trace in event_trend.data)


def test_station_archive_shortfall_chart_ranks_largest_gap_first_visually() -> None:
    deficits = pd.DataFrame(
        {
            "display_label": ["AAA · Alpha", "BBB · Bravo", "CCC · Charlie"],
            "missed_90": [10.0, 40.0, 20.0],
            "observed_90": [170.0, 140.0, 160.0],
            "expected_90": [180.0, 180.0, 180.0],
        }
    )
    figure = station_archive_shortfall_figure(deficits)
    assert figure.data[0].orientation == "h"
    assert list(figure.data[0].x) == [10.0, 20.0, 40.0]
    assert list(figure.data[0].y)[-1] == "BBB · Bravo"


def test_station_archive_surplus_chart_ranks_largest_surplus_visually() -> None:
    deficits = pd.DataFrame(
        {
            "display_label": ["AAA", "BBB", "CCC"],
            "observed_90": [190.0, 210.0, 180.0],
            "expected_90": [180.0, 180.0, 180.0],
        }
    )
    figure = station_archive_surplus_figure(deficits)
    assert figure.data[0].orientation == "h"
    assert list(figure.data[0].x) == [10.0, 30.0]
    assert list(figure.data[0].y)[-1] == "BBB"


def test_real_snapshot_reconciles_latest_station_kpi() -> None:
    snapshot = load_dashboard_snapshot(REPO_ROOT)
    latest = latest_issue_rows(snapshot.issues, snapshot.nco)
    statuses = station_status_frame(snapshot.stations, latest)
    impacted = int(statuses["status"].eq("NCO-reported issue").sum())
    assert impacted == snapshot.payload.issue_count
    assert snapshot.payload.latest_date
    assert not snapshot.payload.series.empty
    assert "coverage_start_utc" in snapshot.source_status
    assert "coverage_end_utc" in snapshot.source_status
    assert "duplicate_rows" in snapshot.source_status
    assert (snapshot.source_status["duplicate_rows"] == 0).all()


def test_streamlit_app_default_state_renders_metrics_charts_and_tables() -> None:
    snapshot = load_dashboard_snapshot(REPO_ROOT)
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert app.radio(key="dashboard_view").options == ["Overview", "Archive detail", "NCO operations", "Station explorer"]
    assert app.radio(key="dashboard_view").value == "Overview"
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Current 7-day archive gap"] == f"{snapshot.payload.gap_percent:.1f}%"
    assert metrics["NCO-reported issue statuses"] == f"{snapshot.payload.issue_count:.0f} / {len(snapshot.stations)}"
    assert len(app.get("plotly_chart")) == 4
    assert len(app.dataframe) == 1


def test_streamlit_navigation_renders_only_selected_view() -> None:
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=30)
    app.run(timeout=30)
    app.radio(key="dashboard_view").set_value("NCO operations")
    app.run(timeout=30)
    assert not app.exception
    assert app.radio(key="dashboard_view").value == "NCO operations"
    assert len(app.get("plotly_chart")) == 3
