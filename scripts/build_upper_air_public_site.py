"""Build the standalone, static dashboard intended for soundings.wall.cloud.

Run after the monitor refresh, or by itself against the latest local outputs:
    python scripts/build_upper_air_public_site.py

The result is a dependency-free static site in ``upper-air-site/dist``. Plotly is
embedded once so the charts retain hover details and date-range controls.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from upper_air_network_monitor.dashboard_charts import (
    archive_trend_figure,
    archive_windows_figure,
    issue_category_figure,
    station_archive_shortfall_figure,
    station_archive_surplus_figure,
)
from upper_air_network_monitor.dashboard_data import (
    archive_window_metrics,
    format_pp_delta,
    issue_counts_by_cycle,
    latest_issue_rows,
    latest_complete_nco_date,
    load_dashboard_snapshot,
    nco_daily_ingest,
    nco_lookback_metrics,
    station_issue_changes,
    station_status_frame,
    source_health_summary,
)
from upper_air_network_monitor.social_graphics import _plot_station_map


DEFAULT_OUTPUT = REPO_ROOT / "upper-air-site" / "dist"
PUBLIC_URL = "https://soundings.wall.cloud"


def _plotly_fragment(figure, *, include_runtime: bool, div_id: str) -> str:
    figure.update_layout(autosize=True)
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True if include_runtime else False,
        config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
        div_id=div_id,
    )


def _map_data_uri(snapshot) -> str:
    fig = plt.figure(figsize=(12, 5.5), facecolor="#0d2538")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    _plot_station_map(
        ax,
        snapshot.payload,
        REPO_ROOT / "comfortwx" / "mapping" / "data" / "us_states.geojson",
        show_legend=True,
        marker_size=100,
        legend_font_size=14,
        legend_marker_size=13,
        legend_columns=1,
        show_empty_legend_entries=True,
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _status_class(value: str) -> str:
    return "issue" if value == "NCO-reported issue" else "clean"


def _station_directory(stations: pd.DataFrame) -> str:
    """Render a compact, searchable alternative to an overly dense labeled map."""
    if stations.empty:
        return '<p class="empty">No mapped station-status rows are available.</p>'
    display = stations.sort_values(["status", "state", "station_id"], ascending=[True, True, True])
    rows: list[str] = []
    for row in display.itertuples(index=False):
        status = str(row.status)
        category = str(row.issue_category).replace("_", " ").title() if pd.notna(row.issue_category) and row.issue_category else "—"
        search = " ".join(
            [str(row.station_id), str(row.station_name), str(row.state), status, category]
        ).lower()
        rows.append(
            f'<tr class="station-row" data-search="{html.escape(search, quote=True)}">'
            f"<td><strong>{html.escape(str(row.station_id))}</strong></td>"
            f"<td>{html.escape(str(row.station_name))}</td>"
            f"<td>{html.escape(str(row.state))}</td>"
            f'<td><span class="station-status {_status_class(status)}">{html.escape(status)}</span></td>'
            f"<td>{html.escape(category)}</td>"
            "</tr>"
        )
    return (
        '<div class="directory-tools"><label for="station-search">Find a station</label>'
        '<input id="station-search" type="search" placeholder="ID, name, state, or issue" autocomplete="off">'
        f'<span id="station-count">{len(rows)} stations</span></div>'
        '<div class="table-wrap station-table"><table><thead><tr><th>ID</th><th>Station</th><th>State</th>'
        '<th>Latest status</th><th>Category</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _issue_rows(snapshot) -> str:
    changes = station_issue_changes(snapshot.issues, snapshot.nco)
    if changes.empty:
        return '<p class="empty">No comparable station-status rows are available.</p>'
    display = changes.copy()
    order = {"New issue": 0, "Category changed": 1, "Persistent": 2, "Resolved": 3}
    display["order"] = display["transition"].map(order).fillna(9)
    display = display.sort_values(["order", "station_id"])
    body = []
    for row in display.itertuples(index=False):
        transition = html.escape(str(row.transition))
        css = transition.lower().replace(" ", "-")
        body.append(
            "<tr>"
            f"<td><strong>{html.escape(str(row.station_id))}</strong></td>"
            f'<td><span class="status {css}">{transition}</span></td>'
            f"<td>{html.escape(str(row.latest_category or 'No issue reported').replace('_', ' '))}</td>"
            f"<td>{html.escape(str(row.previous_category or 'No issue reported').replace('_', ' '))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Station</th><th>Change</th>'
        '<th>Latest category</th><th>Previous comparable</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
    )


def _write_downloads(output_dir: Path, snapshot, station_status: pd.DataFrame) -> None:
    snapshot.payload.series.to_csv(output_dir / "archive-availability.csv", index=False)
    station_status.to_csv(output_dir / "latest-station-status.csv", index=False)
    snapshot.nco.to_csv(output_dir / "nco-ingest-history.csv", index=False)


def _write_fallback_og(output_dir: Path, snapshot) -> None:
    """Create a small, dependency-free OG image for NCO-only refreshes.

    The scheduled NCO checks intentionally skip the expensive social-graphics
    render.  GitHub Pages still needs an ``og.png`` asset, so keep the card
    shareable without making a stale social render a required build input.
    """
    figure = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="#061521")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    axis.text(0.06, 0.72, "CONUS UPPER-AIR DATA WATCH", color="#73d7ff",
              fontsize=21, fontweight="bold", family="DejaVu Sans")
    axis.text(0.06, 0.49, "NCO operational-message availability", color="#f5f7fa",
              fontsize=31, fontweight="bold", family="DejaVu Sans")
    axis.text(0.06, 0.29, "Product records are not unique station counts.", color="#b5c5d1",
              fontsize=17, family="DejaVu Sans")
    axis.text(0.06, 0.12, "soundings.wall.cloud  ·  wall.cloud", color="#6b8190",
              fontsize=14, family="DejaVu Sans")
    figure.savefig(output_dir / "og.png", dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def _date_bounds(frame: pd.DataFrame, column: str, fallback: object, *, default_days: int = 90) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return source minimum, default-window start, and source maximum dates."""
    fallback_date = pd.to_datetime(fallback, errors="coerce")
    if pd.isna(fallback_date):
        fallback_date = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    else:
        fallback_date = pd.Timestamp(fallback_date).normalize()
    if frame.empty or column not in frame:
        return fallback_date, fallback_date, fallback_date
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return fallback_date, fallback_date, fallback_date
    first = dates.min().normalize()
    last = dates.max().normalize()
    default_start = max(first, last - pd.Timedelta(days=default_days - 1))
    return first, default_start, last


def _display_date(value: object) -> str:
    date = pd.Timestamp(value)
    return f"{date.strftime('%b')} {date.day}, {date.year}"


def _source_coverage(snapshot, source: str) -> str:
    rows = snapshot.source_status[snapshot.source_status["source"].eq(source)]
    if rows.empty:
        return "unavailable"
    start = pd.to_datetime(rows.iloc[0].get("coverage_start_utc"), errors="coerce", utc=True)
    end = pd.to_datetime(rows.iloc[0].get("coverage_end_utc"), errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(end):
        return "unavailable"
    return f"{_display_date(start)} – {_display_date(end)}"


def _source_qualifier(snapshot, source: str) -> str:
    if source == "IGRA daily archive" and snapshot.payload.partial_date:
        date = pd.Timestamp(snapshot.payload.partial_date)
        return f"Preliminary {date.strftime('%b')} {date.day} excluded"
    return ""


def _nco_freshness(snapshot) -> tuple[str, str, bool]:
    """Return compact NCO source-record and refresh status text."""
    rows = snapshot.source_status[snapshot.source_status["source"].eq("NCO availability")]
    row = rows.iloc[0] if not rows.empty else None
    status = snapshot.refresh_status.get("sources", {}).get("nco", {}) if isinstance(snapshot.refresh_status.get("sources", {}), dict) else {}
    latest_record = status.get("latest_successful_record_date") if isinstance(status, dict) else None
    last_refresh = status.get("last_successful_fetch_utc") if isinstance(status, dict) else None
    state = str(status.get("status", "")) if isinstance(status, dict) else ""
    if not latest_record and row is not None:
        latest_record = pd.to_datetime(row.get("coverage_end_utc"), errors="coerce", utc=True)
        latest_record = latest_record.strftime("%Y-%m-%d") if pd.notna(latest_record) else None
    if not last_refresh and row is not None:
        modified = pd.to_datetime(row.get("modified_utc"), errors="coerce", utc=True)
        last_refresh = modified.isoformat() if pd.notna(modified) else None
    if latest_record:
        latest_text = _display_date(pd.Timestamp(latest_record))
    else:
        latest_text = "unavailable"
    if last_refresh:
        refresh_time = pd.to_datetime(last_refresh, errors="coerce", utc=True)
    else:
        refresh_time = pd.NaT
    if pd.notna(refresh_time):
        now = pd.Timestamp.now(tz="UTC")
        age_hours = max(0.0, (now - refresh_time).total_seconds() / 3600.0)
        if age_hours < 1:
            age_text = "less than 1 hour ago"
        elif age_hours < 48:
            age_text = f"{age_hours:.0f} hours ago"
        else:
            age_text = f"{age_hours / 24:.1f} days ago"
        refresh_text = f"{refresh_time.strftime('%b')} {refresh_time.day}, {refresh_time.year} {refresh_time.strftime('%H:%M')} UTC"
    else:
        age_text = "unknown"
        refresh_text = "unavailable"
    stale = state in {"failed_retained", "failed"} or (pd.notna(refresh_time) and age_hours > 30)
    return latest_text, f"Last successful NCO refresh: {refresh_text} · Updated {age_text}", stale


def _nco_json_value(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _nco_payload(views: dict[str, pd.DataFrame], source_start: pd.Timestamp, source_end: pd.Timestamp) -> dict[str, object]:
    """Serialize combined and cycle-filtered NCO daily views."""
    payload_views: dict[str, object] = {}
    for view_name, daily in views.items():
        days: list[dict[str, object]] = []
        for row in daily.itertuples(index=False):
            date = pd.Timestamp(row.date).normalize()
            if date < source_start or date > source_end:
                continue
            days.append(
                {
                    "date": date.date().isoformat(),
                    "received": _nco_json_value(row.received),
                    "expected": _nco_json_value(row.expected),
                    "percent": _nco_json_value(row.percent),
                    "available_rows": int(row.available_rows),
                    "models": {
                        "GFS": _nco_json_value(row.gfs_count),
                        "NAM": _nco_json_value(row.nam_count),
                        "NCEP": _nco_json_value(row.ncep_count),
                    },
                }
            )
        payload_views[view_name] = {"days": days}
    return {"views": payload_views, "min_date": source_start.date().isoformat(), "max_date": source_end.date().isoformat()}


def _nco_reference_count(stations: pd.DataFrame) -> int | None:
    if stations.empty:
        return None
    active = stations.get("active_expected", pd.Series(True, index=stations.index))
    return int(active.astype(str).str.lower().isin({"true", "1", "yes"}).sum())


def _nco_latest_text(daily: pd.DataFrame, reference_count: int | None) -> tuple[str, str]:
    if daily.empty:
        return "Latest: —", "No complete NCO day available"
    latest = daily.sort_values("date").iloc[-1]
    received = float(latest.received) if pd.notna(latest.received) else float("nan")
    expected = float(latest.expected) if pd.notna(latest.expected) else float("nan")
    percent = float(latest.percent) if pd.notna(latest.percent) else float("nan")
    date_text = _display_date(latest.date)
    if pd.isna(received):
        return "Latest: —", f"Complete through {date_text}"
    if pd.isna(expected) or not expected:
        return f"Latest: {received:.0f} of {int(latest.available_rows)} product records", f"Complete through {date_text} · expected inventory unavailable"
    return f"Latest: {received:.0f} of {int(latest.available_rows)} product records · {percent:.1f}%", f"Complete through {date_text}"


def _nco_heatmap_markup(
    views: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    source_start: pd.Timestamp,
    source_end: pd.Timestamp,
    default_start: pd.Timestamp,
    default_end: pd.Timestamp,
    stations: pd.DataFrame,
    freshness: tuple[str, str, bool],
) -> str:
    daily = views["combined"][0]
    payload = json.dumps(_nco_payload({name: value[0] for name, value in views.items()}, source_start, source_end), separators=(",", ":"))
    latest_text, latest_detail = _nco_latest_text(daily, _nco_reference_count(stations))
    metrics = views["combined"][1]
    latest_record_text, refresh_text, stale = freshness
    compact_refresh = refresh_text.replace("Last successful NCO refresh: ", "").replace(" · Updated ", " · ")
    metric_cards: list[str] = []
    for days in (7, 14, 30, 90):
        row = metrics[metrics["days"].eq(days)] if not metrics.empty else pd.DataFrame()
        current = row.iloc[0]["current_percent"] if not row.empty else float("nan")
        delta = row.iloc[0]["delta_pp"] if not row.empty else float("nan")
        current_text = f"{float(current):.1f}%" if pd.notna(current) else "—"
        metric_cards.append(
            f'<div class="nco-lookback" role="listitem"><span>{days}D</span>'
            f'<strong id="nco-metric-current-{days}">{current_text}</strong><small id="nco-metric-delta-{days}" title="Compares with the immediately preceding equal-length period">{html.escape(format_pp_delta(delta))}</small></div>'
        )
    return (
        '<div class="nco-ingest-head"><div><div class="chart-title">CONUS RAOB Ingest</div>'
        f'<div class="nco-latest" id="nco-latest">{html.escape(latest_text)}</div>'
        f'<div id="nco-latest-detail" class="nco-latest-detail">{html.escape(latest_detail)}</div></div>'
        '<div class="nco-view-controls"><button type="button" id="nco-one-year" class="nco-view-button active">1Y</button>'
        '<button type="button" id="nco-custom-toggle" class="nco-view-button" aria-expanded="false" aria-controls="nco-heatmap-custom">Custom</button></div></div>'
        '<div class="nco-cycle-controls" role="group" aria-label="NCO cycle view"><span>Cycle</span><button type="button" class="nco-cycle-button active" data-cycle-view="combined">Combined</button><button type="button" class="nco-cycle-button" data-cycle-view="00Z">00Z</button><button type="button" class="nco-cycle-button" data-cycle-view="12Z">12Z</button></div>'
        f'<div class="nco-freshness{" stale" if stale else ""}" title="{html.escape(refresh_text, quote=True)}"><strong>Source:</strong> {html.escape(latest_record_text)} · {html.escape(compact_refresh)}{(" · retained valid data" if stale else "")}</div>'
        '<div class="nco-lookbacks-label" title="Each percentage-point delta compares with the immediately preceding equal-length period">Average ingest</div>'
        '<div class="nco-lookbacks" role="list" aria-label="Average NCO ingest rates; each delta compares with the immediately preceding equal-length period">'
        + "".join(metric_cards)
        + '</div><div id="nco-heatmap-custom" class="nco-heatmap-custom" hidden>'
        f'<label>Start <input id="nco-heatmap-start" type="date" min="{source_start.date().isoformat()}" max="{source_end.date().isoformat()}" value="{default_start.date().isoformat()}"></label>'
        f'<label>End <input id="nco-heatmap-end" type="date" min="{source_start.date().isoformat()}" max="{source_end.date().isoformat()}" value="{default_end.date().isoformat()}"></label>'
        '<button type="button" id="nco-heatmap-apply">Apply</button><button type="button" id="nco-heatmap-reset">Reset</button>'
        '</div><div id="nco-range-summary" class="nco-range-summary" aria-live="polite" hidden></div>'
        '<div class="nco-heatmap-scroller"><div id="nco-heatmap" class="nco-heatmap" role="group" aria-label="Daily combined NCO operational-message ingest calendar"></div></div>'
        '<div class="nco-heatmap-legend" aria-label="Ingest health scale">'
        '<span title="Healthy: 98–100%" aria-label="Healthy: 98 to 100 percent"><i class="health-healthy"></i>Healthy</span>'
        '<span title="Minor: 95–97.9%" aria-label="Minor: 95 to 97.9 percent"><i class="health-minor"></i>Minor</span>'
        '<span title="Reduced: 90–94.9%" aria-label="Reduced: 90 to 94.9 percent"><i class="health-reduced"></i>Reduced</span>'
        '<span title="Degraded: below 90%" aria-label="Degraded: below 90 percent"><i class="health-degraded"></i>Degraded</span>'
        '<span title="No data: monitoring record unavailable" aria-label="No data: monitoring record unavailable"><i class="health-none"></i>No data</span></div>'
        '<div id="nco-cell-tooltip" class="nco-cell-tooltip" role="tooltip" hidden></div>'
        '<p id="nco-health-thresholds" class="sr-only">Healthy: 98 to 100 percent. Minor misses: 95 to 97.9 percent. Reduced: 90 to 94.9 percent. Degraded: below 90 percent. No data: no monitoring record.</p>'
        '<div id="nco-cell-detail" class="nco-cell-detail" tabindex="-1" aria-live="polite" hidden><button type="button" class="nco-detail-close" aria-label="Dismiss selected day details">×</button></div>'
        f'<script type="application/json" id="nco-heatmap-payload" data-min-date="{source_start.date().isoformat()}" data-max-date="{source_end.date().isoformat()}" data-default-start="{default_start.date().isoformat()}" data-default-end="{default_end.date().isoformat()}">{payload}</script>'
    )


def _station_deficit_frame(stations: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    path = REPO_ROOT / "outputs" / "conus_balloon_launches_station_deficits.csv"
    if not path.exists():
        return pd.DataFrame(), "2021–2024 same-date baseline"
    try:
        deficits = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(), "2021–2024 same-date baseline"
    for column in ("missed_90", "observed_90", "expected_90"):
        if column in deficits:
            deficits[column] = pd.to_numeric(deficits[column], errors="coerce")
    master = stations[["igra_id", "station_id", "station_name", "state"]].drop_duplicates("igra_id")
    deficits = deficits.rename(columns={"station_id": "igra_id"}).merge(master, on="igra_id", how="left")
    fallback = deficits["igra_id"].astype(str).str[-5:]
    station_code = deficits["station_id"].fillna(fallback).astype(str)
    station_name = deficits["station_name"].fillna(deficits.get("name", "")).astype(str)
    deficits["display_label"] = station_code + " · " + station_name.str.slice(0, 24)
    baseline = "2021–2024 same-date baseline"
    if "baseline_years" in deficits and not deficits["baseline_years"].dropna().empty:
        years = str(deficits["baseline_years"].dropna().iloc[0]).split(";")
        if years:
            baseline = f"{'–'.join((years[0], years[-1]))} same-date baseline"
    return deficits, baseline


def build_public_site(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    snapshot = load_dashboard_snapshot(REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_issues = latest_issue_rows(snapshot.issues, snapshot.nco)
    current_stations = station_status_frame(snapshot.stations, current_issues)
    station_deficits, station_baseline_label = _station_deficit_frame(snapshot.stations)
    changes = station_issue_changes(snapshot.issues, snapshot.nco)
    active_count = len(current_stations)
    issue_count = int(current_stations["status"].eq("NCO-reported issue").sum()) if not current_stations.empty else 0
    clean_count = max(active_count - issue_count, 0)
    transition_counts = changes["transition"].value_counts() if not changes.empty else pd.Series(dtype="int64")
    new_count = int(transition_counts.get("New issue", 0) + transition_counts.get("Category changed", 0))
    persistent_count = int(transition_counts.get("Persistent", 0))
    resolved_count = int(transition_counts.get("Resolved", 0))
    kpis = snapshot.payload
    health = source_health_summary(snapshot.source_status)
    health_color = "green" if health["problems"] == 0 and health["duplicate_rows"] == 0 else "amber"
    optional_health = "Optional SPC feed unavailable" if health["optional_problems"] else "Optional feeds ready"
    nco_coverage = _source_coverage(snapshot, "NCO availability")
    issue_coverage = _source_coverage(snapshot, "NCO station issues")
    igra_coverage = _source_coverage(snapshot, "IGRA daily archive")
    igra_qualifier = _source_qualifier(snapshot, "IGRA daily archive")
    archive_windows = archive_window_metrics(kpis.series, days=(7, 14, 30, 60, 90, 180, 360))
    window_90 = archive_windows[archive_windows["days"].eq(90)]
    shortfall_90 = abs(float(window_90.iloc[0]["deficit"])) if not window_90.empty else float("nan")
    percent_90 = float(window_90.iloc[0]["percent"]) if not window_90.empty else float("nan")

    nco_daily = nco_daily_ingest(snapshot.nco, snapshot.stations)
    nco_last = latest_complete_nco_date(snapshot.nco)
    if pd.isna(nco_last) and not nco_daily.empty:
        nco_last = pd.Timestamp(nco_daily["date"].max()).normalize()
    if not nco_daily.empty and pd.notna(nco_last):
        nco_daily = nco_daily[pd.to_datetime(nco_daily["date"], errors="coerce").le(nco_last)].copy()
    if nco_daily.empty:
        fallback_nco_date = _date_bounds(snapshot.nco, "cycle_dt", kpis.latest_date, default_days=365)[2]
        nco_first = nco_default_start = nco_last = fallback_nco_date
    else:
        nco_first = pd.Timestamp(nco_daily["date"].min()).normalize()
        nco_last = pd.Timestamp(nco_daily["date"].max()).normalize()
        nco_default_start = max(nco_first, nco_last - pd.Timedelta(days=364))
    nco_views = {
        "combined": nco_daily,
        "00Z": nco_daily_ingest(snapshot.nco, snapshot.stations, cycle_hours=("00",)),
        "12Z": nco_daily_ingest(snapshot.nco, snapshot.stations, cycle_hours=("12",)),
    }
    # Keep every selector on the same trailing complete-data boundary.  A
    # 00Z message can arrive during the current UTC day, but that day is not
    # complete until the next UTC date and must not enter the default year.
    if pd.notna(nco_last):
        for name, frame in list(nco_views.items()):
            if not frame.empty:
                nco_views[name] = frame[pd.to_datetime(frame["date"], errors="coerce").le(nco_last)].copy()
    nco_view_metrics = {
        name: nco_lookback_metrics(frame, windows=(7, 14, 30, 90), end_date=(frame["date"].max() if not frame.empty else nco_last))
        for name, frame in nco_views.items()
    }
    issue_dates = pd.to_datetime(snapshot.issues.get("cycle_dt", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()
    issue_span_days = max(90, int((issue_dates.max() - issue_dates.min()).days) + 1) if not issue_dates.empty else 90
    issue_history = issue_counts_by_cycle(snapshot.issues, issue_span_days)
    issue_first, issue_default_start, issue_last = _date_bounds(issue_history, "cycle_dt", kpis.latest_date, default_days=28)

    trend_data = kpis.series.copy()
    trend_data["date"] = pd.to_datetime(trend_data["date"], errors="coerce")
    trend_data["observed"] = pd.to_numeric(trend_data["observed"], errors="coerce")
    trend_data["baseline"] = pd.to_numeric(trend_data["baseline"], errors="coerce")
    valid_dates = trend_data["date"].dropna()
    first_trend_date = valid_dates.min() if not valid_dates.empty else pd.Timestamp(kpis.latest_date)
    last_trend_date = valid_dates.max() if not valid_dates.empty else pd.Timestamp(kpis.latest_date)
    default_start = max(first_trend_date, last_trend_date - pd.Timedelta(days=364))
    # The initial 1-year view uses a baseline-derived cap so older historical
    # event peaks remain available through the Full Y scale toggle without
    # flattening the current operational story.
    default_view = trend_data[trend_data["date"].ge(default_start)]
    scale_values = pd.concat([default_view["observed"], default_view["baseline"]]).dropna()
    baseline_values = default_view["baseline"].dropna()
    y_cap_low = math.floor(float(scale_values.min()) - 2) if not scale_values.empty else 0
    y_cap_high = math.ceil(float(baseline_values.max())) if not baseline_values.empty else 1

    trend_figure = archive_trend_figure(
        kpis.series,
        show_event_tags=True,
        show_workforce_events=True,
        height=500,
    )
    trend_figure.layout.annotations = ()
    trend_figure.update_layout(margin={"l": 62, "r": 34, "t": 40, "b": 48})
    trend_figure.update_xaxes(range=[default_start, last_trend_date], rangeselector=None)
    trend_figure.update_layout(yaxis={"range": [y_cap_low, y_cap_high], "autorange": False})
    trend = _plotly_fragment(trend_figure, include_runtime=True, div_id="archive-trend")
    windows = _plotly_fragment(archive_windows_figure(archive_windows, height=360, vertical=True), include_runtime=False, div_id="archive-windows")
    station_windows = ((365, "1YR"), (180, "6MO"), (90, "90D"), (30, "30D"), (7, "7D"))
    station_default_days = 30
    station_shortfall_panels: list[str] = []
    station_surplus_panels: list[str] = []
    for days, label in station_windows:
        shortfall_figure = _plotly_fragment(
            station_archive_shortfall_figure(station_deficits, height=360, days=days),
            include_runtime=False,
            div_id=f"station-shortfalls-{days}",
        )
        surplus_figure = _plotly_fragment(
            station_archive_surplus_figure(station_deficits, height=360, days=days),
            include_runtime=False,
            div_id=f"station-surpluses-{days}",
        )
        hidden = "" if days == station_default_days else " hidden"
        station_shortfall_panels.append(f'<div class="station-ranking-panel" data-window="{days}"{hidden}>{shortfall_figure}</div>')
        station_surplus_panels.append(f'<div class="station-ranking-panel" data-window="{days}"{hidden}>{surplus_figure}</div>')
    station_shortfall_panels_html = "".join(station_shortfall_panels)
    station_surplus_panels_html = "".join(station_surplus_panels)
    station_window_buttons = "".join(
        f'<button type="button" class="station-window-button{" active" if days == station_default_days else ""}" data-window="{days}" aria-pressed="{"true" if days == station_default_days else "false"}">{label}</button>'
        for days, label in station_windows
    )
    nco = _nco_heatmap_markup(
        {name: (frame, nco_view_metrics[name]) for name, frame in nco_views.items()},
        nco_first,
        nco_last,
        nco_default_start,
        nco_last,
        snapshot.stations,
        _nco_freshness(snapshot),
    )
    issue_figure = issue_category_figure(issue_history, show_smoothed_trend=True, height=350)
    issue_figure.update_xaxes(range=[issue_default_start, issue_last])
    categories = _plotly_fragment(
        issue_figure,
        include_runtime=False,
        div_id="issue-categories",
    )
    map_uri = _map_data_uri(snapshot)
    updated = pd.to_datetime(kpis.generated_at, errors="coerce")
    updated_text = (
        f"{updated.strftime('%B')} {updated.day}, {updated.year} at {updated.strftime('%H:%M')} UTC"
        if pd.notna(updated)
        else "Latest local monitor run"
    )
    build_time = dt.datetime.now(dt.timezone.utc)
    build_text = f"{build_time.strftime('%B')} {build_time.day}, {build_time.year} at {build_time.strftime('%H:%M')} UTC"
    _write_downloads(output_dir, snapshot, current_stations)

    exact_caveats = (
        "Archive records are compared with the 2021–2024 same-date baseline.",
        "NCO status reflects operational-message reporting, not confirmed IGRA archive totals.",
    )
    caveats = "".join(f"<li>{html.escape(item)}</li>" for item in exact_caveats)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>CONUS Upper-Air Data Watch | wall.cloud</title>
<meta name="description" content="A source-backed view of CONUS upper-air archive availability and NCO operational-message status.">
<meta property="og:title" content="CONUS Upper-Air Data Watch"><meta property="og:description" content="Archive availability and operational station-status diagnostics from wall.cloud.">
<meta property="og:image" content="{PUBLIC_URL}/og.png"><meta property="og:url" content="{PUBLIC_URL}/"><meta name="theme-color" content="#061521">
<style>
:root{{--bg:#061521;--panel:#0d2538;--panel2:#102c42;--line:#294960;--text:#f8fbff;--muted:#afc1d4;--blue:#59c8f5;--orange:#ff704f;--green:#52d3a2;--amber:#f6c85f}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;overflow-x:hidden}}body{{margin:0;min-width:0;overflow-x:hidden;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}}a{{color:inherit}}button,input{{font:inherit}}
.skip{{position:fixed;left:12px;top:-60px;z-index:99;background:var(--text);color:var(--bg);padding:10px 14px;border-radius:8px}}.skip:focus{{top:10px}}
.wrap{{width:min(1240px,calc(100% - 32px));margin-inline:auto;min-width:0}}nav{{position:sticky;top:0;z-index:20;background:rgba(6,21,33,.94);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}}nav .wrap{{min-height:64px;display:flex;align-items:center;justify-content:space-between;gap:20px}}.brand{{font-weight:850;letter-spacing:-.02em;color:var(--blue);text-decoration:none}}.navlinks{{display:flex;gap:22px;font-size:14px;color:var(--muted)}}.navlinks a{{text-decoration:none}}.nav-status{{font-size:12px;color:var(--green);border:1px solid var(--line);border-radius:999px;padding:5px 9px}}
.hero{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr);gap:32px;align-items:center;padding:30px 0 8px}}.eyebrow{{color:var(--blue);font-weight:800;letter-spacing:.14em;font-size:12px}}h1{{font-size:clamp(26px,3vw,38px);line-height:1.1;letter-spacing:-.035em;max-width:860px;margin:12px 0 0;overflow-wrap:break-word}}.signal{{background:linear-gradient(150deg,rgba(255,112,79,.17),var(--panel) 58%);border-color:rgba(255,112,79,.6)!important}}.signal .kpi-value{{font-size:clamp(62px,7vw,96px);line-height:.95}}.signal-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;border-top:1px solid var(--line);margin-top:18px;padding-top:16px}}.signal-grid strong{{display:block;font-size:22px}}.signal-grid span{{color:var(--muted);font-size:12px}}
.section{{padding:28px 0}}.section-head{{margin-bottom:12px}}h2{{font-size:clamp(26px,3vw,38px);letter-spacing:-.035em;line-height:1.1;margin:0}}.grid{{display:grid;gap:14px;min-width:0}}.grid>*{{min-width:0}}.two{{grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)}}.two-even{{grid-template-columns:repeat(2,minmax(0,1fr))}}.card{{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;overflow:hidden}}.status-card{{overflow:visible}}.kpi-label{{text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:800;font-size:11px}}.kpi-value{{font-size:clamp(38px,4vw,56px);font-weight:850;letter-spacing:-.05em;margin:7px 0 2px}}.problem{{color:var(--orange)}}.clean{{color:var(--green)}}.kpi-detail{{color:var(--muted);font-size:13px}}.chart-card{{padding:14px 12px 6px}}.chart-title{{padding:7px 9px 0;font-weight:800;font-size:17px}}.chart-sub{{padding:2px 9px;color:var(--muted);font-size:13px}}.js-plotly-plot,.plot-container,.svg-container{{max-width:100%!important}}.map{{width:100%;height:auto;display:block;border-radius:12px}}ul{{color:var(--muted);padding-left:20px}}li+li{{margin-top:9px}}
 .trend-controls{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:12px 9px 4px}}.preset-controls{{display:flex;gap:6px;flex-wrap:wrap}}.trend-controls button,.custom-range input,.custom-range button{{border:1px solid var(--line);background:var(--bg);color:var(--text);border-radius:8px;padding:7px 9px}}.trend-controls button,.custom-range button{{cursor:pointer;font-weight:750;font-size:12px}}.trend-controls button:hover,.trend-controls button.active,.custom-range button:hover{{background:var(--panel2);border-color:var(--blue)}}.scale-toggle{{color:var(--blue)!important}}.custom-range{{display:flex;align-items:center;gap:6px;margin-left:auto;color:var(--muted);font-size:12px}}.custom-range input{{color-scheme:dark;max-width:140px}}.custom-range button{{color:var(--blue)}}.operation-controls{{padding-top:8px}}.operation-controls .custom-range{{width:100%;justify-content:flex-end}}
 .nco-ingest-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:3px 9px 0}}.nco-ingest-head .chart-title{{padding:0}}.nco-latest{{font-weight:800;font-size:15px;margin-top:2px}}.nco-latest-detail{{color:var(--muted);font-size:11px}}.nco-view-controls{{display:flex;gap:6px;flex:0 0 auto}}.nco-view-button{{border:1px solid var(--line);background:var(--bg);color:var(--text);border-radius:8px;padding:6px 9px;cursor:pointer;font-weight:750;font-size:12px}}.nco-view-button.active,.nco-view-button:hover,.nco-view-button:focus-visible{{border-color:var(--blue);background:var(--panel2)}}.nco-lookbacks{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;padding:10px 9px 8px}}.nco-lookback{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:7px 8px;min-width:0}}.nco-lookback span,.nco-lookback small{{display:block;color:var(--muted);font-size:10px}}.nco-lookback strong{{display:block;font-size:16px;line-height:1.2}}.nco-lookback small{{color:var(--blue);margin-top:2px}}.nco-heatmap-custom{{display:flex;align-items:end;gap:7px;flex-wrap:wrap;padding:8px 9px;background:rgba(6,21,33,.45);border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px;color:var(--muted)}}.nco-heatmap-custom label{{display:flex;flex-direction:column;gap:3px}}.nco-heatmap-custom input{{color-scheme:dark;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:6px 7px}}.nco-heatmap-custom button{{border:1px solid var(--line);background:var(--bg);color:var(--blue);border-radius:7px;padding:6px 8px;cursor:pointer;font-weight:750;font-size:11px}}.nco-range-summary{{color:var(--muted);font-size:11px;padding:6px 9px 0}}.nco-heatmap-scroller{{overflow-x:auto;padding:2px 9px 0}}.nco-heatmap{{min-width:max(100%,calc(var(--nco-week-count,53) * 7px));font-size:9px}}.nco-months{{display:grid;grid-template-columns:repeat(var(--nco-week-count),minmax(0,1fr));margin-left:18px;height:17px;color:var(--muted);font-size:9px}}.nco-months span{{white-space:nowrap}}.nco-heatmap-body{{display:flex;gap:4px}}.nco-weekday-labels{{width:14px;display:grid;grid-template-rows:repeat(7,1fr);color:var(--muted);font-size:8px;text-align:right;line-height:1}}.nco-heatmap-grid{{display:grid;grid-template-columns:repeat(var(--nco-week-count),minmax(0,1fr));grid-template-rows:repeat(7,10px);gap:2px;flex:1}}.nco-cell{{display:block;width:100%;height:10px;min-width:5px;padding:0;border:0;border-radius:2px;background:var(--panel2);cursor:pointer;outline-offset:2px}}.nco-cell:hover,.nco-cell:focus-visible{{box-shadow:0 0 0 2px var(--text);z-index:2}}.nco-cell.health-healthy{{background:var(--green)}}.nco-cell.health-minor{{background:#89d58f}}.nco-cell.health-reduced{{background:var(--amber)}}.nco-cell.health-degraded{{background:var(--orange)}}.nco-cell.no-data{{background:#40566b}}.nco-heatmap-legend{{display:flex;gap:8px;flex-wrap:wrap;padding:7px 9px 2px;color:var(--muted);font-size:9px}}.nco-heatmap-legend span{{display:inline-flex;align-items:center;gap:3px;white-space:nowrap}}.nco-heatmap-legend i{{width:8px;height:8px;border-radius:2px;display:inline-block}}.health-healthy{{background:var(--green)}}.health-minor{{background:#89d58f}}.health-reduced{{background:var(--amber)}}.health-degraded{{background:var(--orange)}}.health-none{{background:#40566b}}.nco-cell-detail{{margin:7px 9px 3px;padding:8px 9px;background:var(--panel2);border:1px solid var(--line);border-radius:9px;color:var(--muted);font-size:11px;min-height:42px}}.nco-cell-detail strong{{display:block;color:var(--text);font-size:12px}}.nco-cell-detail span{{display:block}}.nco-cell-models{{color:var(--muted);font-size:10px}}
 .change-strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}.change-stat{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:13px}}.change-stat strong{{display:block;font-size:26px}}.change-stat span{{color:var(--muted);font-size:12px}}.change-stat.new strong{{color:var(--orange)}}.change-stat.resolved strong{{color:var(--green)}}
 .nco-view-controls{{border:1px solid var(--line);border-radius:9px;padding:2px;gap:0;background:var(--bg)}}.nco-view-button{{border:0;border-radius:6px;padding:5px 8px}}.nco-lookbacks-label{{padding:8px 9px 0;color:var(--muted);font-size:10px;font-weight:750;letter-spacing:.02em}}.nco-lookbacks{{gap:0;padding:6px 9px 7px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.nco-lookback{{background:transparent;border:0;border-right:1px solid var(--line);border-radius:0;padding:3px 8px}}.nco-lookback:last-child{{border-right:0}}.nco-lookback strong{{font-size:15px}}.nco-heatmap-scroller{{position:relative;overscroll-behavior-x:contain}}.nco-heatmap{{min-width:100%}}.nco-months{{margin-left:16px}}.nco-months span{{display:block;max-width:28px;overflow:visible}}.nco-heatmap-grid{{gap:1px;grid-template-rows:repeat(7,9px)}}.nco-cell{{height:9px;min-width:0;border-radius:2px}}.nco-weekday-labels{{width:12px;font-size:8px}}.nco-heatmap-legend span{{font-size:9px}}.nco-heatmap-legend span::after{{content:none!important}}.nco-cell-tooltip{{position:fixed;z-index:5;max-width:230px;padding:7px 9px;background:var(--panel2);border:1px solid var(--blue);border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.35);color:var(--text);font-size:11px;pointer-events:none}}.nco-cell-tooltip strong,.nco-cell-tooltip span{{display:block}}.nco-cell-tooltip span{{color:var(--muted);font-size:10px}}.nco-cell-detail{{position:relative;margin:6px 9px 3px;padding:7px 30px 7px 9px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--muted);font-size:11px}}.nco-cell-detail strong{{display:block;color:var(--text);font-size:12px}}.nco-cell-detail span{{display:block}}.nco-cell-models{{color:var(--muted);font-size:10px}}.nco-detail-close{{position:absolute;right:6px;top:4px;border:0;background:transparent;color:var(--muted);font-size:16px;line-height:1;cursor:pointer}}
details{{border-top:1px solid var(--line);margin-top:16px;padding-top:12px}}summary{{cursor:pointer;font-weight:800;color:var(--blue);list-style:none}}summary::-webkit-details-marker{{display:none}}summary::after{{content:" +";color:var(--muted)}}details[open] summary::after{{content:" −"}}.table-wrap{{overflow:auto;margin-top:10px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}.status,.station-status{{display:inline-block;padding:4px 8px;border-radius:999px;background:var(--panel2)}}.status.new-issue,.status.category-changed,.station-status.issue{{color:var(--orange)}}.status.resolved,.station-status.clean{{color:var(--green)}}
.review-card{{margin-top:8px;padding:12px 20px}}.review-card details{{border-top:0;margin-top:0;padding-top:0}}
 .directory-tools{{display:grid;grid-template-columns:auto minmax(180px,420px) 1fr;gap:12px;align-items:center;margin:14px 0}}.directory-tools label{{font-weight:800}}.directory-tools input{{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px 12px;outline:none}}.directory-tools input:focus{{border-color:var(--blue);box-shadow:0 0 0 3px rgba(89,200,245,.12)}}#station-count{{justify-self:end;color:var(--muted);font-size:12px}}.station-table{{max-height:420px}}.downloads{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:18px}}.download{{text-decoration:none;text-align:center;border:1px solid var(--line);border-radius:10px;padding:9px 12px;color:var(--blue);font-weight:750;font-size:13px}}.download:hover{{background:var(--panel2)}}
 .health-summary{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:2px 0 0}}.health-summary strong{{font-size:16px}}.health-badge{{border:1px solid var(--line);border-radius:999px;padding:4px 9px;font-size:11px;font-weight:800}}.health-badge.green{{color:var(--green);border-color:rgba(82,211,162,.5)}}.health-badge.amber{{color:var(--amber);border-color:rgba(246,200,95,.5)}}.health-optional{{color:var(--muted);font-size:12px;margin:4px 0 12px}}.health-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:8px}}.health-item{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:11px;min-width:0}}.health-item strong{{display:block;font-size:14px;white-space:nowrap}}.health-item span,.health-item small{{display:block;color:var(--muted);font-size:12px;margin-top:3px}}.health-item small{{font-size:11px;color:var(--amber)}}.health-item .green{{color:var(--green)}}.health-item .amber{{color:var(--amber)}}
 [hidden]{{display:none!important}}.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}.nco-detail-close:focus-visible,.nco-view-button:focus-visible{{outline:2px solid var(--blue);outline-offset:2px}}
footer{{border-top:1px solid var(--line);margin-top:28px;padding:24px 0 40px;color:var(--muted);font-size:13px}}.footer-row{{display:flex;justify-content:space-between;gap:20px}}.empty{{color:var(--muted)}}
@media(min-width:901px){{.station-search-details[open]{{position:fixed;z-index:100;top:7vh;left:50%;transform:translateX(-50%);width:min(1120px,calc(100vw - 64px));max-height:86vh;overflow:auto;margin:0;padding:22px;background:var(--panel);border:1px solid var(--blue);border-radius:18px;box-shadow:0 0 0 100vmax rgba(2,10,17,.72),0 24px 70px rgba(0,0,0,.5)}}.station-search-details[open] .station-table{{max-height:none}}}}
@media(max-width:900px){{.hero{{grid-template-columns:1fr;padding-top:32px}}.signal{{max-width:none}}.two,.two-even{{grid-template-columns:1fr}}}}
 @media(max-width:600px){{.nco-heatmap-grid{{grid-template-rows:repeat(7,8px)}}.nco-cell{{height:8px}}.nco-weekday-labels{{width:10px}}.nco-weekday-labels span{{font-size:0}}.nco-weekday-labels span:nth-child(1)::after{{content:"M";font-size:8px}}.nco-weekday-labels span:nth-child(3)::after{{content:"W";font-size:8px}}.nco-weekday-labels span:nth-child(5)::after{{content:"F";font-size:8px}}.nco-months{{font-size:8px;margin-left:14px;overflow:hidden}}.nco-months span{{max-width:24px}}.nco-months span:nth-child(even){{visibility:hidden}}.nco-heatmap-legend{{gap:6px;padding-inline:5px}}.nco-cell-tooltip{{display:none!important}}}} @media(max-width:380px){{.nco-weekday-labels{{visibility:hidden}}}}
 @media(max-width:600px){{.wrap{{width:calc(100% - 20px)}}nav .wrap{{min-height:56px}}.navlinks{{display:none}}.nav-status{{font-size:11px}}.hero{{gap:20px;padding:24px 0 8px}}h1{{font-size:clamp(26px,8vw,34px)}}.signal .kpi-value{{font-size:68px}}.section{{padding:22px 0}}.card{{padding:15px;border-radius:15px}}.chart-card{{padding:10px 4px 2px}}.custom-range{{width:100%;margin-left:0;flex-wrap:wrap}}.custom-range input{{min-width:0;flex:1}}.change-strip{{grid-template-columns:1fr 1fr 1fr}}.change-stat{{padding:10px}}.change-stat strong{{font-size:22px}}.directory-tools{{grid-template-columns:1fr}}#station-count{{justify-self:start}}.nco-lookbacks{{gap:4px;padding-inline:5px}}.nco-lookback{{padding:6px 5px}}.nco-lookback strong{{font-size:14px}}.nco-lookback small{{font-size:9px}}.nco-ingest-head{{padding-inline:5px}}.nco-heatmap-scroller{{padding-inline:5px}}.health-grid{{grid-template-columns:1fr;gap:7px}}.health-item{{display:grid;grid-template-columns:minmax(132px,.85fr) minmax(0,1.15fr);column-gap:10px;align-items:baseline;padding:9px}}.health-item strong{{white-space:normal}}.health-item small{{grid-column:2}}.downloads{{gap:6px}}.download{{padding:8px 5px;font-size:11px}}.footer-row{{display:block}}th,td{{padding:10px}}}}
</style><style>
.nco-cycle-controls{{display:flex;align-items:center;gap:5px;padding:7px 9px 0;color:var(--muted);font-size:10px}}.nco-cycle-controls span{{font-weight:750;margin-right:2px}}.nco-cycle-button{{border:1px solid var(--line);background:var(--bg);color:var(--muted);border-radius:6px;padding:4px 7px;cursor:pointer;font-size:10px;font-weight:750}}.nco-cycle-button.active,.nco-cycle-button:hover,.nco-cycle-button:focus-visible{{color:var(--text);border-color:var(--blue);background:var(--panel2)}}
 .nco-ingest-subtitle{{color:var(--muted);font-size:10px;margin-top:2px}}.nco-freshness{{margin:8px 9px 0;padding:7px 9px;border:1px solid var(--line);border-radius:8px;color:var(--muted);font-size:10px}}.nco-freshness strong{{color:var(--text);font-weight:750}}.nco-freshness.stale{{border-color:rgba(246,200,95,.65);color:var(--amber)}}
</style></head><body>
<a class="skip" href="#content">Skip to data</a>
<nav><div class="wrap"><a class="brand" href="https://wall.cloud/">wall.cloud</a><div class="navlinks"><a href="#archive">Soundings</a><a href="#stations">Stations</a><a href="#operations">Operations</a><a href="#methods">Methods</a></div><div class="nav-status">Data through {html.escape(str(kpis.latest_date))}</div></div></nav>
<main id="content" class="wrap">
<header class="hero"><div><div class="eyebrow">CONUS UPPER-AIR DATA WATCH</div><h1>Sounding data availability, clearly tracked.</h1></div>
<article class="card signal"><div class="kpi-label">Current 7-day archive gap</div><div class="kpi-value problem">{kpis.gap_percent:.1f}%</div><div class="kpi-detail">{kpis.observed:.1f} observed vs {kpis.expected:.1f} expected records per day</div><div class="signal-grid"><div><strong>{shortfall_90:.0f}</strong><span>fewer records over 90 days</span></div><div><strong>{percent_90:.1f}%</strong><span>90-day difference</span></div></div></article></header>
<section id="archive" class="section"><div class="section-head"><div class="eyebrow">SOUNDING AVAILABILITY</div><h2>Observed records versus expected</h2></div><article class="card chart-card"><div class="chart-title">Sounding availability trend</div><div class="chart-sub">Diamonds label same-date historical event maximums above 140/day. The orange dotted line marks NWS RAOB cuts.</div><div class="trend-controls"><div class="preset-controls" aria-label="Trend date range"><button type="button" class="range-button" data-days="182">6MO</button><button type="button" class="range-button active" data-days="365">1YR</button><button type="button" class="range-button" data-days="730">2YR</button><button type="button" class="range-button" data-days="10">10D</button><button type="button" class="range-button" data-days="30">30D</button><button type="button" class="range-button" data-days="60">60D</button><button type="button" class="range-button" data-days="90">90D</button><button type="button" id="nws-layoffs-range" class="event-range-button">NWS Layoffs</button><button type="button" id="scale-toggle" class="scale-toggle">Full Y scale</button></div><form id="custom-range" class="custom-range"><span>Custom</span><input id="range-start" type="date" aria-label="Custom range start" min="{first_trend_date.date().isoformat()}" max="{last_trend_date.date().isoformat()}" value="{first_trend_date.date().isoformat()}"><span>to</span><input id="range-end" type="date" aria-label="Custom range end" min="{first_trend_date.date().isoformat()}" max="{last_trend_date.date().isoformat()}" value="{last_trend_date.date().isoformat()}"><button type="submit">Apply</button></form></div>{trend}</article></section>
<section class="section"><article class="card chart-card"><div class="chart-title">Recent archive windows</div>{windows}</article><div class="grid two-even station-ranking-grid"><article class="card chart-card"><div class="chart-title">Stations ranked by archive shortfall</div><div class="chart-sub">IGRA archive records vs {html.escape(station_baseline_label)}</div><div class="station-window-controls" aria-label="Shortfall ranking period">{station_window_buttons}</div>{station_shortfall_panels_html}</article><article class="card chart-card"><div class="chart-title">Stations ranked by archive surplus</div><div class="chart-sub">IGRA archive records above {html.escape(station_baseline_label)}</div><div class="station-window-controls" aria-label="Surplus ranking period">{station_window_buttons}</div>{station_surplus_panels_html}</article></div></section>

<section id="stations" class="section"><div class="section-head"><div class="eyebrow">STATION STATUS</div><h2>Current NCO-reported issues</h2></div><div class="grid two"><article class="card"><img class="map" src="{map_uri}" alt="Miller-projection map of CONUS upper-air stations with state borders and latest NCO-reported status"></article><article class="card status-card"><div class="kpi-label">Latest mapped status</div><p class="kpi-detail">Stations with an NCO-reported issue</p><div class="kpi-value problem">{issue_count} / {active_count}</div><p class="kpi-detail">{clean_count} stations have no issue reported</p><div class="change-strip"><div class="change-stat new"><strong>{new_count}</strong><span>new or changed</span></div><div class="change-stat"><strong>{persistent_count}</strong><span>persistent</span></div><div class="change-stat resolved"><strong>{resolved_count}</strong><span>resolved</span></div></div><details class="station-search-details"><summary>Search all mapped stations</summary>{_station_directory(current_stations)}</details></article></div></section>

<section id="operations" class="section"><div class="section-head"><div class="eyebrow">OPERATIONAL MESSAGES</div><h2>NCO reported for ingest</h2></div><div class="grid two-even"><article class="card chart-card">{nco}</article><article class="card chart-card"><div class="chart-title">Reported issue categories</div><div class="chart-sub">Default view: latest 28 days</div><div class="trend-controls operation-controls"><form id="issue-custom-range" class="custom-range"><span>Custom</span><input id="issue-range-start" type="date" aria-label="Issue-category custom range start" min="{issue_first.date().isoformat()}" max="{issue_last.date().isoformat()}" value="{issue_default_start.date().isoformat()}"><span>to</span><input id="issue-range-end" type="date" aria-label="Issue-category custom range end" min="{issue_first.date().isoformat()}" max="{issue_last.date().isoformat()}" value="{issue_last.date().isoformat()}"><button type="submit">Apply</button></form></div>{categories}</article></div><article class="card review-card"><details><summary>Review station changes from the previous comparable cycle</summary>{_issue_rows(snapshot)}</details></article></section>

<section id="methods" class="section"><div class="section-head"><div><div class="eyebrow">SOURCES & METHODS</div><h2>Sources and data health</h2></div><p>Built {html.escape(build_text)}.<br>Data snapshot {html.escape(updated_text)}.</p></div><article class="card"><div class="health-summary"><div><strong>{health["required_ready"]} of {health["required_total"]} required products ready</strong></div><span class="health-badge {health_color}">{"Ready" if health["problems"] == 0 and health["duplicate_rows"] == 0 else "Review"}</span></div><p class="health-optional">{html.escape(optional_health)}</p><div class="health-grid"><div class="health-item"><strong>IGRA daily archive</strong><span>{html.escape(igra_coverage)}</span>{f'<small>{html.escape(igra_qualifier)}</small>' if igra_qualifier else ''}</div><div class="health-item"><strong>NCO ingest counts</strong><span>{html.escape(nco_coverage)}</span></div><div class="health-item"><strong>NCO issue statuses</strong><span>{html.escape(issue_coverage)}</span></div></div><ul>{caveats}</ul><p class="kpi-detail">Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages; CONUS upper-air station master.</p><div class="downloads"><a class="download" href="archive-availability.csv" download aria-label="Download archive CSV">Archive CSV</a><a class="download" href="latest-station-status.csv" download aria-label="Download station CSV">Station CSV</a><a class="download" href="nco-ingest-history.csv" download aria-label="Download NCO CSV">NCO CSV</a></div></article></section>
</main><footer><div class="wrap footer-row"><div><strong class="brand">wall.cloud</strong><br>Weather data, carefully framed.</div><div><a href="#methods">Sources and methods</a></div></div></footer>
<script>
const trendChart=document.getElementById('archive-trend');
const trendEnd=new Date('{last_trend_date.date().isoformat()}T00:00:00Z');
const rangeButtons=document.querySelectorAll('.range-button');
const nwsLayoffsButton=document.getElementById('nws-layoffs-range');
rangeButtons.forEach(button=>button.addEventListener('click',()=>{{
  const start=new Date(trendEnd);
  start.setUTCDate(start.getUTCDate()-(Number(button.dataset.days)-1));
  Plotly.relayout(trendChart,{{'xaxis.range':[start.toISOString(),trendEnd.toISOString()],'xaxis.autorange':false}});
  rangeButtons.forEach(item=>item.classList.toggle('active',item===button));
  nwsLayoffsButton?.classList.remove('active');
}}));
const customRange=document.getElementById('custom-range');
customRange.addEventListener('submit',event=>{{
  event.preventDefault();
  const start=document.getElementById('range-start');
  const end=document.getElementById('range-end');
  if(!start.value||!end.value||start.value>end.value){{start.setCustomValidity('Choose a start date on or before the end date.');start.reportValidity();return;}}
  start.setCustomValidity('');
  Plotly.relayout(trendChart,{{'xaxis.range':[start.value,end.value],'xaxis.autorange':false}});
  rangeButtons.forEach(item=>item.classList.remove('active'));
  nwsLayoffsButton?.classList.remove('active');
}});
const scaleToggle=document.getElementById('scale-toggle');
let fullYScale=false;
nwsLayoffsButton?.addEventListener('click',()=>{{
  Plotly.relayout(trendChart,{{'xaxis.range':['2025-01-01','2025-04-20'],'xaxis.autorange':false,'yaxis.range':[{y_cap_low},{y_cap_high}],'yaxis.autorange':false}});
  fullYScale=false;
  scaleToggle.textContent='Full Y scale';
  rangeButtons.forEach(item=>item.classList.remove('active'));
  nwsLayoffsButton.classList.add('active');
}});
scaleToggle.addEventListener('click',()=>{{
  fullYScale=!fullYScale;
  if(fullYScale){{Plotly.relayout(trendChart,{{'yaxis.autorange':true}});scaleToggle.textContent='Use baseline cap';}}
  else{{Plotly.relayout(trendChart,{{'yaxis.range':[{y_cap_low},{y_cap_high}],'yaxis.autorange':false}});scaleToggle.textContent='Full Y scale';}}
}});
function attachCustomDateRange(formId,startId,endId,chartId){{
  const form=document.getElementById(formId);
  const chart=document.getElementById(chartId);
  if(!form||!chart)return;
  form.addEventListener('submit',event=>{{
    event.preventDefault();
    const start=document.getElementById(startId);
    const end=document.getElementById(endId);
    if(!start.value||!end.value||start.value>end.value){{start.setCustomValidity('Choose a start date on or before the end date.');start.reportValidity();return;}}
    start.setCustomValidity('');
    Plotly.relayout(chart,{{'xaxis.range':[start.value,end.value],'xaxis.autorange':false}});
  }});
}}
attachCustomDateRange('nco-custom-range','nco-range-start','nco-range-end','nco-trend');
attachCustomDateRange('issue-custom-range','issue-range-start','issue-range-end','issue-categories');
const ncoPayloadElement=document.getElementById('nco-heatmap-payload');
if(ncoPayloadElement){{
  const ncoPayload=JSON.parse(ncoPayloadElement.textContent||'{{}}');
  const ncoByDate=new Map((ncoPayload.days||[]).map(row=>[row.date,row]));
  const ncoHeatmap=document.getElementById('nco-heatmap');
  const ncoDetail=document.getElementById('nco-cell-detail');
  const ncoSummary=document.getElementById('nco-range-summary');
  const ncoStartInput=document.getElementById('nco-heatmap-start');
  const ncoEndInput=document.getElementById('nco-heatmap-end');
  const ncoOneYear=document.getElementById('nco-one-year');
  const ncoCustomToggle=document.getElementById('nco-custom-toggle');
  const ncoCustomPanel=document.getElementById('nco-heatmap-custom');
  const ncoApply=document.getElementById('nco-heatmap-apply');
  const ncoReset=document.getElementById('nco-heatmap-reset');
  const ncoMonths=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const ncoIso=date=>date.toISOString().slice(0,10);
  const ncoEscape=value=>String(value).replace(/[&<>\"]/g,character=>({{ '&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;' }})[character]);
  const ncoPct=value=>Number.isFinite(Number(value))?Number(value).toFixed(1)+'%':'—';
  const ncoHealth=value=>{{const number=Number(value);if(!Number.isFinite(number))return 'no-data';if(number>=98)return 'health-healthy';if(number>=95)return 'health-minor';if(number>=90)return 'health-reduced';return 'health-degraded';}};
  const ncoRate=(startIso,endIso)=>{{let received=0,expected=0,days=0;for(const row of ncoPayload.days||[]){{if(row.date>=startIso&&row.date<=endIso&&Number.isFinite(Number(row.received))&&Number(row.expected)>0){{received+=Number(row.received);expected+=Number(row.expected);days++;}}}}return {{rate:expected?received/expected*100:null,days}};}};
  const ncoFormatDate=date=>new Intl.DateTimeFormat('en-US',{{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}}).format(new Date(date+'T00:00:00Z'));
  const ncoPresentSources=row=>Object.entries(row?.models||{{}}).filter(([,value])=>value!==null&&Number.isFinite(Number(value))).map(([model,value])=>model+' report: '+Number(value).toFixed(0));
  const ncoShowCell=date=>{{
    const row=ncoByDate.get(date);
    if(!row){{ncoDetail.innerHTML='<strong>'+ncoEscape(ncoFormatDate(date))+'</strong><span>No data - monitoring record unavailable.</span>';return;}}
    const sources=ncoPresentSources(row);
    ncoDetail.innerHTML='<strong>'+ncoEscape(ncoFormatDate(date))+'</strong><span>'+ncoPct(row.percent)+' · '+Number(row.received).toFixed(0)+' of '+(Number.isFinite(Number(row.expected))?Number(row.expected).toFixed(0):'—')+' expected product records</span><span class="nco-cell-models">'+ncoEscape(sources.length?sources.join(' · '):'No applicable source record')+'</span>';
  }};
  const ncoRender=(startIso,endIso)=>{{
    const start=new Date(startIso+'T00:00:00Z'),end=new Date(endIso+'T00:00:00Z');
    const monday=new Date(start);monday.setUTCDate(monday.getUTCDate()-((monday.getUTCDay()+6)%7));
    const sunday=new Date(end);sunday.setUTCDate(sunday.getUTCDate()+(6-((sunday.getUTCDay()+6)%7)));
    const weeks=Math.round((sunday-monday)/86400000/7)+1;ncoHeatmap.style.setProperty('--nco-week-count',weeks);
    const monthItems=[];const monthCursor=new Date(Date.UTC(start.getUTCFullYear(),start.getUTCMonth(),1));while(monthCursor<=end){{const labelDate=monthCursor<start?start:monthCursor;const week=Math.floor((labelDate-monday)/86400000/7);if(week>=0&&week<weeks)monthItems.push({{week,label:ncoMonths[monthCursor.getUTCMonth()]}});monthCursor.setUTCMonth(monthCursor.getUTCMonth()+1);}}
    const filteredMonths=monthItems.filter((item,index)=>index===0||item.week!==monthItems[index-1].week||(item.label!==monthItems[index-1].label||item.week-monthItems[index-1].week>=4));
    let monthLabels='';for(const item of filteredMonths)monthLabels+='<span style="grid-column:'+(item.week+1)+'">'+item.label+'</span>';
    let cells='';for(let row=0;row<7;row++)for(let column=0;column<weeks;column++){{const day=new Date(monday);day.setUTCDate(day.getUTCDate()+column*7+row);const date=ncoIso(day);if(day<start||day>end){{cells+='<span aria-hidden="true"></span>';continue;}}const value=ncoByDate.get(date);const sources=ncoPresentSources(value);const label=value?date+' '+ncoPct(value.percent)+' '+Number(value.received).toFixed(0)+' of '+(Number.isFinite(Number(value.expected))?Number(value.expected).toFixed(0):'no expected total')+' expected product records'+(sources.length?' · '+sources.join(' · '):''):date+' No data';cells+='<button type="button" class="nco-cell '+ncoHealth(value?.percent)+'" data-date="'+date+'" aria-label="'+ncoEscape(label)+'"></button>';}}
    ncoHeatmap.innerHTML='<div class="nco-months" style="--nco-week-count:'+weeks+'">'+monthLabels+'</div><div class="nco-heatmap-body"><div class="nco-weekday-labels" aria-label="Weekdays"><span>M</span><span>T</span><span>W</span><span>Th</span><span>F</span><span>Sa</span><span>Su</span></div><div class="nco-heatmap-grid" style="--nco-week-count:'+weeks+'">'+cells+'</div></div>';
    ncoHeatmap.querySelectorAll('button[data-date]').forEach(cell=>{{cell.addEventListener('focus',()=>ncoShowCell(cell.dataset.date));cell.addEventListener('click',()=>ncoShowCell(cell.dataset.date));}});
  }};
  const ncoUpdateSummary=(startIso,endIso)=>{{const current=ncoRate(startIso,endIso);const start=new Date(startIso+'T00:00:00Z'),end=new Date(endIso+'T00:00:00Z');const length=Math.round((end-start)/86400000)+1;const previousEnd=new Date(start);previousEnd.setUTCDate(previousEnd.getUTCDate()-1);const previousStart=new Date(previousEnd);previousStart.setUTCDate(previousStart.getUTCDate()-(length-1));const previous=ncoRate(ncoIso(previousStart),ncoIso(previousEnd));const delta=current.rate!==null&&previous.rate!==null?current.rate-previous.rate:null;ncoSummary.hidden=false;ncoSummary.textContent=(current.days===0?'Selected range: No monitoring data':'Selected range: '+ncoPct(current.rate))+' ('+startIso+' to '+endIso+') · Previous equal range: '+ncoPct(previous.rate)+' ('+ncoIso(previousStart)+' to '+ncoIso(previousEnd)+') · Change: '+(delta===null?'—':(delta>=0?'+':'−')+Math.abs(delta).toFixed(1)+' pp');}};
  const ncoDefaultEnd=ncoPayload.max_date;const defaultEndDate=new Date(ncoDefaultEnd+'T00:00:00Z');const defaultStartDate=new Date(defaultEndDate);defaultStartDate.setUTCDate(defaultStartDate.getUTCDate()-364);const ncoDefaultStart=ncoIso(defaultStartDate)<ncoPayload.min_date?ncoPayload.min_date:ncoIso(defaultStartDate);
  ncoRender(ncoDefaultStart,ncoDefaultEnd);
  ncoCustomToggle?.addEventListener('click',()=>{{const expanded=ncoCustomToggle.getAttribute('aria-expanded')==='true';ncoCustomToggle.setAttribute('aria-expanded',String(!expanded));ncoCustomPanel.hidden=expanded;}});
  ncoOneYear?.addEventListener('click',()=>{{ncoStartInput.value=ncoDefaultStart;ncoEndInput.value=ncoDefaultEnd;ncoSummary.hidden=true;ncoOneYear.classList.add('active');ncoCustomToggle?.classList.remove('active');ncoRender(ncoDefaultStart,ncoDefaultEnd);}});
  ncoApply?.addEventListener('click',()=>{{const start=ncoStartInput.value,end=ncoEndInput.value;const invalid=!start||!end||start>end||start<ncoPayload.min_date||end>ncoPayload.max_date;if(invalid){{ncoStartInput.setCustomValidity('Choose dates within the available range, with start on or before end.');ncoStartInput.reportValidity();return;}}ncoStartInput.setCustomValidity('');ncoOneYear?.classList.remove('active');ncoCustomToggle?.classList.add('active');ncoRender(start,end);ncoUpdateSummary(start,end);}});
  ncoReset?.addEventListener('click',()=>{{ncoStartInput.value=ncoDefaultStart;ncoEndInput.value=ncoDefaultEnd;ncoSummary.hidden=true;ncoOneYear?.classList.add('active');ncoCustomToggle?.classList.remove('active');ncoRender(ncoDefaultStart,ncoDefaultEnd);}});
}}
if(ncoPayloadElement){{
  const ncoByDateForDetails=new Map((JSON.parse(ncoPayloadElement.textContent||'{{}}').days||[]).map(row=>[row.date,row]));
  const ncoHeatmapForDetails=document.getElementById('nco-heatmap');
  const ncoTooltip=document.getElementById('nco-cell-tooltip');
  const ncoSelectedDetail=document.getElementById('nco-cell-detail');
  const ncoCustomPanelForState=document.getElementById('nco-heatmap-custom');
  const ncoCustomToggleForState=document.getElementById('nco-custom-toggle');
  const ncoOneYearForState=document.getElementById('nco-one-year');
  const ncoEscapeDetail=value=>String(value).replace(/[&<>"]/g,character=>({{ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }})[character]);
  const ncoFormatDateDetail=date=>new Intl.DateTimeFormat('en-US',{{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}}).format(new Date(date+'T00:00:00Z'));
  const ncoDetailHtml=(date,row)=>{{
    if(!row)return '<strong>'+ncoEscapeDetail(ncoFormatDateDetail(date))+'</strong><span>No data - monitoring record unavailable.</span>';
    const present=Object.entries(row.models||{{}}).filter(([,value])=>value!==null&&Number.isFinite(Number(value))).map(([model,value])=>model+' report: '+Number(value).toFixed(0));
    const sources=present.length?present.join(' · '):'No applicable source record';
    return '<strong>'+ncoEscapeDetail(ncoFormatDateDetail(date))+'</strong><span>'+(Number.isFinite(Number(row.percent))?Number(row.percent).toFixed(1)+'%':'No data')+' · '+Number(row.received).toFixed(0)+' of '+(Number.isFinite(Number(row.expected))?Number(row.expected).toFixed(0):'—')+' expected product records</span><span class="nco-cell-models">'+ncoEscapeDetail(sources)+'</span>';
  }};
  const ncoShowTooltip=cell=>{{
    if(!ncoTooltip)return;
    ncoTooltip.innerHTML=ncoDetailHtml(cell.dataset.date,ncoByDateForDetails.get(cell.dataset.date));
    const rect=cell.getBoundingClientRect();
    ncoTooltip.style.left=Math.min(Math.max(8,rect.left),Math.max(8,window.innerWidth-238))+'px';
    ncoTooltip.style.top=Math.max(8,rect.top-66)+'px';
    ncoTooltip.hidden=false;
  }};
  const ncoHideTooltip=()=>{{if(ncoTooltip)ncoTooltip.hidden=true;}};
  const ncoShowSelected=cell=>{{
    if(!ncoSelectedDetail)return;
    ncoSelectedDetail.innerHTML=ncoDetailHtml(cell.dataset.date,ncoByDateForDetails.get(cell.dataset.date))+'<button type="button" class="nco-detail-close" aria-label="Dismiss selected day details">×</button>';
    ncoSelectedDetail.hidden=false;
  }};
  const ncoBindDetailCells=()=>{{
    ncoHeatmapForDetails?.querySelectorAll('button[data-date]').forEach(cell=>{{
      if(cell.dataset.detailBound)return;
      cell.dataset.detailBound='1';
      cell.addEventListener('mouseenter',()=>ncoShowTooltip(cell));
      cell.addEventListener('mouseleave',ncoHideTooltip);
      cell.addEventListener('focus',()=>ncoShowTooltip(cell));
      cell.addEventListener('blur',ncoHideTooltip);
      cell.addEventListener('click',()=>ncoShowSelected(cell));
    }});
  }};
  ncoBindDetailCells();
  if(ncoHeatmapForDetails)new MutationObserver(ncoBindDetailCells).observe(ncoHeatmapForDetails,{{childList:true,subtree:true}});
  ncoSelectedDetail?.addEventListener('click',event=>{{if(event.target.closest('.nco-detail-close'))ncoSelectedDetail.hidden=true;}});
  const ncoSetCustomOpen=open=>{{if(ncoCustomPanelForState)ncoCustomPanelForState.hidden=!open;if(ncoCustomToggleForState){{ncoCustomToggleForState.setAttribute('aria-expanded',String(open));ncoCustomToggleForState.classList.toggle('active',open);}}ncoOneYearForState?.classList.toggle('active',!open);}};
  ncoCustomToggleForState?.addEventListener('click',()=>ncoSetCustomOpen(!ncoCustomPanelForState?.hidden));
  ncoOneYearForState?.addEventListener('click',()=>ncoSetCustomOpen(false));
  document.getElementById('nco-heatmap-reset')?.addEventListener('click',()=>ncoSetCustomOpen(false));
  ncoSetCustomOpen(false);
}}
// Cycle selector enhancement: the combined heatmap remains the default, while
// 00Z and 12Z views reuse the same compact calendar and metric strip.
if(ncoPayloadElement){{
  const ncoViewsPayload=JSON.parse(ncoPayloadElement.textContent||'{{}}').views||{{}};
  const ncoCycleHeatmap=document.getElementById('nco-heatmap');
  const ncoCycleDetail=document.getElementById('nco-cell-detail');
  const ncoCycleTooltip=document.getElementById('nco-cell-tooltip');
  const ncoCycleSummary=document.getElementById('nco-range-summary');
  const ncoCycleLatest=document.getElementById('nco-latest');
  const ncoCycleLatestDetail=document.getElementById('nco-latest-detail');
  const ncoCycleStart=document.getElementById('nco-heatmap-start');
  const ncoCycleEnd=document.getElementById('nco-heatmap-end');
  let ncoActiveCycle='combined';
  let ncoActiveStart=ncoPayloadElement.dataset.defaultStart||ncoCycleStart?.value||ncoPayloadElement.dataset.minDate;
  let ncoActiveEnd=ncoPayloadElement.dataset.defaultEnd||ncoCycleEnd?.value||ncoPayloadElement.dataset.maxDate;
  const ncoCycleRows=()=>((ncoViewsPayload[ncoActiveCycle]||{{}}).days||[]);
  const ncoCycleDateRows=()=>new Map(ncoCycleRows().map(row=>[row.date,row]));
  const ncoCycleIso=date=>date.toISOString().slice(0,10);
  const ncoCycleFmtDate=date=>new Intl.DateTimeFormat('en-US',{{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}}).format(new Date(date+'T00:00:00Z'));
  const ncoCyclePct=value=>Number.isFinite(Number(value))?Number(value).toFixed(1)+'%':'—';
  const ncoCycleHealth=value=>{{const number=Number(value);if(!Number.isFinite(number))return 'no-data';if(number>=98)return 'health-healthy';if(number>=95)return 'health-minor';if(number>=90)return 'health-reduced';return 'health-degraded';}};
  const ncoCycleRate=(start,end)=>{{let received=0,expected=0,days=0;for(const row of ncoCycleRows()){{if(row.date>=start&&row.date<=end&&Number.isFinite(Number(row.received))&&Number(row.expected)>0){{received+=Number(row.received);expected+=Number(row.expected);days++;}}}}return {{rate:expected?received/expected*100:null,days}};}};
  const ncoCycleDelta=(start,end)=>{{const current=ncoCycleRate(start,end);const a=new Date(start+'T00:00:00Z');const b=new Date(end+'T00:00:00Z');const length=Math.round((b-a)/86400000)+1;const previousEnd=new Date(a);previousEnd.setUTCDate(previousEnd.getUTCDate()-1);const previousStart=new Date(previousEnd);previousStart.setUTCDate(previousStart.getUTCDate()-(length-1));const previous=ncoCycleRate(ncoCycleIso(previousStart),ncoCycleIso(previousEnd));return {{current,previous,delta:current.rate!==null&&previous.rate!==null?current.rate-previous.rate:null,previousStart:ncoCycleIso(previousStart),previousEnd:ncoCycleIso(previousEnd)}};}};
  const ncoCycleModels=row=>Object.entries(row?.models||{{}}).filter(([,value])=>value!==null&&Number.isFinite(Number(value))).map(([model,value])=>model+' report: '+Number(value).toFixed(0));
  const ncoCycleDetailHtml=(date,row)=>{{if(!row)return '<strong>'+ncoCycleFmtDate(date)+'</strong><span>No data — monitoring record unavailable.</span>';const sources=ncoCycleModels(row);return '<strong>'+ncoCycleFmtDate(date)+'</strong><span>'+ncoCyclePct(row.percent)+' · '+Number(row.received).toFixed(0)+' of '+(Number.isFinite(Number(row.expected))?Number(row.expected).toFixed(0):'—')+' expected product records</span><span class="nco-cell-models">'+(sources.length?sources.join(' · '):'No applicable source record')+'</span>';}};
  const ncoCycleShowDetail=(date,row)=>{{if(ncoCycleDetail){{ncoCycleDetail.innerHTML=ncoCycleDetailHtml(date,row)+'<button type="button" class="nco-detail-close" aria-label="Dismiss selected day details">×</button>';ncoCycleDetail.hidden=false;}}}};
  const ncoCycleShowTooltip=(cell)=>{{if(!ncoCycleTooltip)return;ncoCycleTooltip.innerHTML=ncoCycleDetailHtml(cell.dataset.date,ncoCycleDateRows().get(cell.dataset.date));const rect=cell.getBoundingClientRect();ncoCycleTooltip.style.left=Math.min(Math.max(8,rect.left),Math.max(8,window.innerWidth-238))+'px';ncoCycleTooltip.style.top=Math.max(8,rect.top-66)+'px';ncoCycleTooltip.hidden=false;}};
  const ncoCycleBind=()=>{{ncoCycleHeatmap?.querySelectorAll('button[data-date]').forEach(cell=>{{if(cell.dataset.cycleBound)return;cell.dataset.cycleBound='1';cell.addEventListener('mouseenter',event=>{{event.stopImmediatePropagation();ncoCycleShowTooltip(cell);}},true);cell.addEventListener('mouseleave',()=>{{if(ncoCycleTooltip)ncoCycleTooltip.hidden=true;}},true);cell.addEventListener('focus',event=>{{event.stopImmediatePropagation();ncoCycleShowTooltip(cell);}},true);cell.addEventListener('blur',()=>{{if(ncoCycleTooltip)ncoCycleTooltip.hidden=true;}},true);cell.addEventListener('click',event=>{{event.stopImmediatePropagation();ncoCycleShowDetail(cell.dataset.date,ncoCycleDateRows().get(cell.dataset.date));}},true);}});}};
  const ncoCycleUpdateMetrics=()=>{{const rows=ncoCycleRows();const latest=rows.length?rows[rows.length-1]:null;if(ncoCycleLatest){{ncoCycleLatest.textContent=latest?'Latest day: '+Number(latest.received).toFixed(0)+' received across '+Number(latest.available_rows).toFixed(0)+' product records · '+ncoCyclePct(latest.percent):'Latest day: —';}}if(ncoCycleLatestDetail&&latest){{ncoCycleLatestDetail.textContent='Complete through '+ncoCycleFmtDate(latest.date)+' · 69-station reference per applicable record';}}for(const days of [7,14,30,90]){{const end=latest?.date||ncoActiveEnd;const endDate=new Date(end+'T00:00:00Z');const startDate=new Date(endDate);startDate.setUTCDate(startDate.getUTCDate()-(days-1));const result=ncoCycleDelta(ncoCycleIso(startDate),end);const current=document.getElementById('nco-metric-current-'+days);const delta=document.getElementById('nco-metric-delta-'+days);if(current)current.textContent=ncoCyclePct(result.current.rate);if(delta)delta.textContent=result.delta===null?'—':(result.delta>=0?'+':'−')+Math.abs(result.delta).toFixed(1)+' pp';}}}};
  const ncoCycleRender=(startIso,endIso)=>{{ncoActiveStart=startIso;ncoActiveEnd=endIso;const start=new Date(startIso+'T00:00:00Z');const end=new Date(endIso+'T00:00:00Z');const monday=new Date(start);monday.setUTCDate(monday.getUTCDate()-((monday.getUTCDay()+6)%7));const sunday=new Date(end);sunday.setUTCDate(sunday.getUTCDate()+(6-((sunday.getUTCDay()+6)%7)));const weeks=Math.round((sunday-monday)/86400000/7)+1;ncoCycleHeatmap.style.setProperty('--nco-week-count',weeks);const monthItems=[];const cursor=new Date(Date.UTC(start.getUTCFullYear(),start.getUTCMonth(),1));while(cursor<=end){{const labelDate=cursor<start?start:cursor;const week=Math.floor((labelDate-monday)/86400000/7);if(week>=0&&week<weeks)monthItems.push({{week,label:cursor.toLocaleString('en-US',{{month:'short',timeZone:'UTC'}})}});cursor.setUTCMonth(cursor.getUTCMonth()+1);}}const monthLabels=monthItems.filter((item,index)=>index===0||item.week!==monthItems[index-1].week||(item.label!==monthItems[index-1].label||item.week-monthItems[index-1].week>=4)).map(item=>'<span style="grid-column:'+(item.week+1)+'">'+item.label+'</span>').join('');let cells='';const rows=ncoCycleDateRows();for(let row=0;row<7;row++)for(let column=0;column<weeks;column++){{const day=new Date(monday);day.setUTCDate(day.getUTCDate()+column*7+row);const date=ncoCycleIso(day);if(day<start||day>end){{cells+='<span aria-hidden="true"></span>';continue;}}const value=rows.get(date);const sources=ncoCycleModels(value);const label=value?date+' '+ncoCyclePct(value.percent)+' '+Number(value.received).toFixed(0)+' of '+(Number.isFinite(Number(value.expected))?Number(value.expected).toFixed(0):'no expected total')+' expected product records'+(sources.length?' · '+sources.join(' · '):''):date+' No data';cells+='<button type="button" class="nco-cell '+ncoCycleHealth(value?.percent)+'" data-date="'+date+'" aria-label="'+label.replace(/"/g,'&quot;')+'"></button>';}}ncoCycleHeatmap.innerHTML='<div class="nco-months" style="--nco-week-count:'+weeks+'">'+monthLabels+'</div><div class="nco-heatmap-body"><div class="nco-weekday-labels"><span>Mon</span><span aria-hidden="true"></span><span>Wed</span><span aria-hidden="true"></span><span>Fri</span><span aria-hidden="true"></span><span aria-hidden="true"></span></div><div class="nco-heatmap-grid" style="--nco-week-count:'+weeks+'">'+cells+'</div></div>';ncoCycleBind();ncoCycleUpdateMetrics();}};
  const ncoCycleSelect=cycle=>{{ncoActiveCycle=cycle;document.querySelectorAll('.nco-cycle-button').forEach(button=>button.classList.toggle('active',button.dataset.cycleView===cycle));const rows=ncoCycleRows();const latest=rows.length?rows[rows.length-1].date:ncoActiveEnd;const useCustom=document.getElementById('nco-custom-toggle')?.classList.contains('active');ncoCycleRender(useCustom?ncoCycleStart.value:ncoActiveStart, useCustom?ncoCycleEnd.value:latest);}};
  document.querySelectorAll('.nco-cycle-button').forEach(button=>button.addEventListener('click',event=>{{event.stopImmediatePropagation();ncoCycleSelect(button.dataset.cycleView);}},true));
  document.getElementById('nco-heatmap-apply')?.addEventListener('click',event=>{{event.stopImmediatePropagation();const start=ncoCycleStart.value,end=ncoCycleEnd.value;if(!start||!end||start>end||start<ncoPayloadElement.dataset.minDate||end>ncoPayloadElement.dataset.maxDate){{ncoCycleStart.setCustomValidity('Choose dates within the available range, with start on or before end.');ncoCycleStart.reportValidity();return;}}ncoCycleStart.setCustomValidity('');document.getElementById('nco-one-year')?.classList.remove('active');document.getElementById('nco-custom-toggle')?.classList.add('active');ncoCycleRender(start,end);const result=ncoCycleDelta(start,end);if(ncoCycleSummary){{ncoCycleSummary.hidden=false;ncoCycleSummary.textContent='Selected range: '+ncoCyclePct(result.current.rate)+' · Previous equal range: '+ncoCyclePct(result.previous.rate)+' · Change: '+(result.delta===null?'—':(result.delta>=0?'+':'−')+Math.abs(result.delta).toFixed(1)+' pp');}}}},true);
   document.getElementById('nco-one-year')?.addEventListener('click',event=>{{event.stopImmediatePropagation();document.getElementById('nco-heatmap-custom').hidden=true;document.getElementById('nco-custom-toggle')?.classList.remove('active');ncoCycleSummary.hidden=true;const rows=ncoCycleRows();const end=rows.length?rows[rows.length-1].date:ncoPayloadElement.dataset.maxDate;const endDate=new Date(end+'T00:00:00Z');const startDate=new Date(endDate);startDate.setUTCDate(startDate.getUTCDate()-364);const start=ncoCycleIso(startDate)<ncoPayloadElement.dataset.minDate?ncoPayloadElement.dataset.minDate:ncoCycleIso(startDate);ncoCycleRender(start,end);}},true);
   document.getElementById('nco-heatmap-reset')?.addEventListener('click',event=>{{event.stopImmediatePropagation();document.getElementById('nco-heatmap-custom').hidden=true;document.getElementById('nco-custom-toggle')?.classList.remove('active');ncoCycleSummary.hidden=true;const rows=ncoCycleRows();const end=rows.length?rows[rows.length-1].date:ncoPayloadElement.dataset.maxDate;const endDate=new Date(end+'T00:00:00Z');const startDate=new Date(endDate);startDate.setUTCDate(startDate.getUTCDate()-364);const start=ncoCycleIso(startDate)<ncoPayloadElement.dataset.minDate?ncoPayloadElement.dataset.minDate:ncoCycleIso(startDate);ncoCycleRender(start,end);}},true);
   const ncoInitialRows=ncoCycleRows();const ncoInitialEnd=ncoInitialRows.length?ncoInitialRows[ncoInitialRows.length-1].date:(ncoCycleEnd?.value||ncoPayloadElement.dataset.maxDate);const ncoInitialDate=new Date(ncoInitialEnd+'T00:00:00Z');const ncoInitialStartDate=new Date(ncoInitialDate);ncoInitialStartDate.setUTCDate(ncoInitialStartDate.getUTCDate()-364);const ncoInitialStart=ncoCycleIso(ncoInitialStartDate)<ncoPayloadElement.dataset.minDate?ncoPayloadElement.dataset.minDate:ncoCycleIso(ncoInitialStartDate);ncoCycleRender(ncoInitialStart,ncoInitialEnd);
}}
document.querySelectorAll('.station-window-controls').forEach(control=>control.addEventListener('click',event=>{{
  const button=event.target.closest('.station-window-button');
  if(!button)return;
  const card=control.closest('.card');
  const windowDays=button.dataset.window;
  card.querySelectorAll('.station-window-button').forEach(item=>{{const active=item===button;item.classList.toggle('active',active);item.setAttribute('aria-pressed',String(active));}});
  card.querySelectorAll('.station-ranking-panel').forEach(panel=>{{panel.hidden=panel.dataset.window!==windowDays;}});
  card.querySelectorAll('.station-ranking-panel:not([hidden]) .js-plotly-plot').forEach(chart=>{{if(window.Plotly)Plotly.Plots.resize(chart);}});
}}));
const stationSearch=document.getElementById('station-search');
if(stationSearch){{stationSearch.addEventListener('input',()=>{{const query=stationSearch.value.trim().toLowerCase();let visible=0;document.querySelectorAll('.station-row').forEach(row=>{{const show=!query||row.dataset.search.includes(query);row.hidden=!show;if(show)visible++;}});document.getElementById('station-count').textContent=`${{visible}} station${{visible===1?'':'s'}}`;}});}}
</script></body></html>"""

    # The cycle-selector renderer is intentionally kept in the inline page
    # script. Normalize its weekday markup here so both render paths show the
    # complete calendar instead of the older Mon/Wed/Fri-only labels.
    page = page.replace(
        '<div class="nco-weekday-labels"><span>Mon</span><span aria-hidden="true"></span><span>Wed</span><span aria-hidden="true"></span><span>Fri</span><span aria-hidden="true"></span><span aria-hidden="true"></span></div>',
        '<div class="nco-weekday-labels" aria-label="Weekdays"><span>M</span><span>T</span><span>W</span><span>Th</span><span>F</span><span>Sa</span><span>Su</span></div>',
    )
    page = page.replace(
        '</head>',
        '<style>.station-ranking-grid{margin-top:14px}.station-window-controls{display:flex;gap:4px;flex-wrap:wrap;padding:8px 9px 2px}.station-window-button{border:1px solid var(--line);background:var(--bg);color:var(--muted);border-radius:7px;padding:5px 8px;cursor:pointer;font-size:11px;font-weight:750}.station-window-button:hover,.station-window-button.active,.station-window-button:focus-visible{color:var(--text);border-color:var(--blue);background:var(--panel2)}.station-ranking-panel[hidden]{display:none!important}.nco-ingest-card{display:flex;flex-direction:column}.nco-ingest-card .nco-heatmap-scroller{flex:1;display:flex;min-height:0}.nco-ingest-card .nco-heatmap{display:flex;flex:1;flex-direction:column;width:100%}.nco-ingest-card .nco-heatmap-body{flex:1;min-height:120px}.nco-ingest-card .nco-heatmap-grid{height:100%;grid-template-rows:repeat(7,minmax(10px,1fr))}.nco-ingest-card .nco-cell{height:auto;min-height:10px}.nco-weekday-labels{width:22px;font-size:8px}.nco-weekday-labels span{font-size:8px;white-space:nowrap}.nco-weekday-labels span::after{content:none!important}.nco-months{margin-left:26px}.nco-months span{visibility:visible!important}@media(max-width:600px){.station-window-controls{padding-inline:5px}.station-window-button{padding:5px 6px;font-size:10px}.nco-ingest-card .nco-heatmap-body{min-height:90px}.nco-weekday-labels{width:18px;font-size:7px;visibility:visible!important}.nco-weekday-labels span{font-size:7px}.nco-months{margin-left:22px;font-size:7px}.nco-months span{max-width:28px;visibility:visible!important}}</style></head>',
        1,
    )
    page = page.replace(
        '<article class="card chart-card"><div class="nco-ingest-head">',
        '<article class="card chart-card nco-ingest-card"><div class="nco-ingest-head">',
        1,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    social_image = REPO_ROOT / "outputs" / "upper_air_network_monitor" / "social" / "original_dashboard_style.png"
    if social_image.exists():
        shutil.copy2(social_image, output_dir / "og.png")
    else:
        _write_fallback_og(output_dir, snapshot)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_public_site(args.output_dir.resolve())
    print(f"Built public dashboard: {path}")


if __name__ == "__main__":
    main()
