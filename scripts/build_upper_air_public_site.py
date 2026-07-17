"""Build the standalone, static dashboard intended for soundings.wall.cloud.

Run after the monitor refresh, or by itself against the latest local outputs:
    python scripts/build_upper_air_public_site.py

The result is a dependency-free static site in ``upper-air-site/dist``. Plotly is
embedded once so the charts retain hover details and date-range controls.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
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
    nco_ingest_figure,
    station_archive_shortfall_figure,
)
from upper_air_network_monitor.dashboard_data import (
    archive_window_metrics,
    issue_counts_by_cycle,
    latest_issue_rows,
    load_dashboard_snapshot,
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


def _date_bounds(frame: pd.DataFrame, column: str, fallback: object, *, default_days: int = 90) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return source minimum, default-window start, and source maximum dates."""
    fallback_date = pd.Timestamp(fallback).normalize()
    if frame.empty or column not in frame:
        return fallback_date, fallback_date, fallback_date
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return fallback_date, fallback_date, fallback_date
    first = dates.min().normalize()
    last = dates.max().normalize()
    default_start = max(first, last - pd.Timedelta(days=default_days - 1))
    return first, default_start, last


def _source_coverage(snapshot, source: str) -> str:
    rows = snapshot.source_status[snapshot.source_status["source"].eq(source)]
    if rows.empty:
        return "unavailable"
    start = pd.to_datetime(rows.iloc[0].get("coverage_start_utc"), errors="coerce", utc=True)
    end = pd.to_datetime(rows.iloc[0].get("coverage_end_utc"), errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(end):
        return "unavailable"
    text = f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"
    if source == "IGRA daily archive" and snapshot.payload.partial_date:
        text += f" (preliminary {pd.Timestamp(snapshot.payload.partial_date).strftime('%b %d')} excluded)"
    return text


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
    health_status = "All required source products ready" if health["problems"] == 0 and health["duplicate_rows"] == 0 else "Review required source health"
    health_color = "green" if health["problems"] == 0 and health["duplicate_rows"] == 0 else "amber"
    optional_health = "Optional SPC feed unavailable" if health["optional_problems"] else "Optional feeds ready"
    nco_coverage = _source_coverage(snapshot, "NCO availability")
    issue_coverage = _source_coverage(snapshot, "NCO station issues")
    igra_coverage = _source_coverage(snapshot, "IGRA daily archive")
    archive_windows = archive_window_metrics(kpis.series, days=(7, 14, 30, 60, 90, 180, 360))
    window_90 = archive_windows[archive_windows["days"].eq(90)]
    shortfall_90 = abs(float(window_90.iloc[0]["deficit"])) if not window_90.empty else float("nan")
    percent_90 = float(window_90.iloc[0]["percent"]) if not window_90.empty else float("nan")

    nco_first, nco_default_start, nco_last = _date_bounds(snapshot.nco, "cycle_dt", kpis.latest_date, default_days=365)
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
    station_shortfalls = _plotly_fragment(
        station_archive_shortfall_figure(station_deficits, height=360),
        include_runtime=False,
        div_id="station-shortfalls",
    )
    nco_figure = nco_ingest_figure(snapshot.nco, show_smoothed_trend=True, smooth_window_days=7, height=350)
    nco_figure.update_xaxes(range=[nco_default_start, nco_last])
    nco = _plotly_fragment(nco_figure, include_runtime=False, div_id="nco-trend")
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
.change-strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}.change-stat{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:13px}}.change-stat strong{{display:block;font-size:26px}}.change-stat span{{color:var(--muted);font-size:12px}}.change-stat.new strong{{color:var(--orange)}}.change-stat.resolved strong{{color:var(--green)}}
details{{border-top:1px solid var(--line);margin-top:16px;padding-top:12px}}summary{{cursor:pointer;font-weight:800;color:var(--blue);list-style:none}}summary::-webkit-details-marker{{display:none}}summary::after{{content:" +";color:var(--muted)}}details[open] summary::after{{content:" −"}}.table-wrap{{overflow:auto;margin-top:10px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}.status,.station-status{{display:inline-block;padding:4px 8px;border-radius:999px;background:var(--panel2)}}.status.new-issue,.status.category-changed,.station-status.issue{{color:var(--orange)}}.status.resolved,.station-status.clean{{color:var(--green)}}
.review-card{{margin-top:8px;padding:12px 20px}}.review-card details{{border-top:0;margin-top:0;padding-top:0}}
.directory-tools{{display:grid;grid-template-columns:auto minmax(180px,420px) 1fr;gap:12px;align-items:center;margin:14px 0}}.directory-tools label{{font-weight:800}}.directory-tools input{{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px 12px;outline:none}}.directory-tools input:focus{{border-color:var(--blue);box-shadow:0 0 0 3px rgba(89,200,245,.12)}}#station-count{{justify-self:end;color:var(--muted);font-size:12px}}.station-table{{max-height:420px}}.downloads{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}.download{{text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:9px 12px;color:var(--blue);font-weight:750;font-size:13px}}.download:hover{{background:var(--panel2)}}
.health-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}}.health-item{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:12px}}.health-item strong{{display:block;font-size:15px}}.health-item span{{display:block;color:var(--muted);font-size:12px;margin-top:3px}}.health-item .green{{color:var(--green)}}.health-item .amber{{color:var(--amber)}}
footer{{border-top:1px solid var(--line);margin-top:28px;padding:24px 0 40px;color:var(--muted);font-size:13px}}.footer-row{{display:flex;justify-content:space-between;gap:20px}}.empty{{color:var(--muted)}}
@media(min-width:901px){{.station-search-details[open]{{position:fixed;z-index:100;top:7vh;left:50%;transform:translateX(-50%);width:min(1120px,calc(100vw - 64px));max-height:86vh;overflow:auto;margin:0;padding:22px;background:var(--panel);border:1px solid var(--blue);border-radius:18px;box-shadow:0 0 0 100vmax rgba(2,10,17,.72),0 24px 70px rgba(0,0,0,.5)}}.station-search-details[open] .station-table{{max-height:none}}}}
@media(max-width:900px){{.hero{{grid-template-columns:1fr;padding-top:32px}}.signal{{max-width:none}}.two,.two-even{{grid-template-columns:1fr}}}}
@media(max-width:600px){{.wrap{{width:calc(100% - 20px)}}nav .wrap{{min-height:56px}}.navlinks{{display:none}}.nav-status{{font-size:11px}}.hero{{gap:20px;padding:24px 0 8px}}h1{{font-size:clamp(26px,8vw,34px)}}.signal .kpi-value{{font-size:68px}}.section{{padding:22px 0}}.card{{padding:15px;border-radius:15px}}.chart-card{{padding:10px 4px 2px}}.custom-range{{width:100%;margin-left:0;flex-wrap:wrap}}.custom-range input{{min-width:0;flex:1}}.change-strip{{grid-template-columns:1fr 1fr 1fr}}.change-stat{{padding:10px}}.change-stat strong{{font-size:22px}}.directory-tools{{grid-template-columns:1fr}}#station-count{{justify-self:start}}.footer-row{{display:block}}th,td{{padding:10px}}}}
</style></head><body>
<a class="skip" href="#content">Skip to data</a>
<nav><div class="wrap"><a class="brand" href="https://wall.cloud/">wall.cloud</a><div class="navlinks"><a href="#archive">Soundings</a><a href="#stations">Stations</a><a href="#operations">Operations</a><a href="#methods">Methods</a></div><div class="nav-status">Data through {html.escape(str(kpis.latest_date))}</div></div></nav>
<main id="content" class="wrap">
<header class="hero"><div><div class="eyebrow">CONUS UPPER-AIR DATA WATCH</div><h1>Sounding data availability, clearly tracked.</h1></div>
<article class="card signal"><div class="kpi-label">Current 7-day archive gap</div><div class="kpi-value problem">{kpis.gap_percent:.1f}%</div><div class="kpi-detail">{kpis.observed:.1f} observed vs {kpis.expected:.1f} expected records per day</div><div class="signal-grid"><div><strong>{shortfall_90:.0f}</strong><span>fewer records over 90 days</span></div><div><strong>{percent_90:.1f}%</strong><span>90-day difference</span></div></div></article></header>
<section id="archive" class="section"><div class="section-head"><div class="eyebrow">SOUNDING AVAILABILITY</div><h2>Observed records versus expected</h2></div><article class="card chart-card"><div class="chart-title">Sounding availability trend</div><div class="chart-sub">Diamonds label same-date historical event maximums above 140/day. The orange dotted line marks NWS RAOB cuts.</div><div class="trend-controls"><div class="preset-controls" aria-label="Trend date range"><button type="button" class="range-button" data-days="182">6MO</button><button type="button" class="range-button active" data-days="365">1YR</button><button type="button" class="range-button" data-days="730">2YR</button><button type="button" class="range-button" data-days="10">10D</button><button type="button" class="range-button" data-days="30">30D</button><button type="button" class="range-button" data-days="60">60D</button><button type="button" class="range-button" data-days="90">90D</button><button type="button" id="nws-layoffs-range" class="event-range-button">NWS Layoffs</button><button type="button" id="scale-toggle" class="scale-toggle">Full Y scale</button></div><form id="custom-range" class="custom-range"><span>Custom</span><input id="range-start" type="date" aria-label="Custom range start" min="{first_trend_date.date().isoformat()}" max="{last_trend_date.date().isoformat()}" value="{first_trend_date.date().isoformat()}"><span>to</span><input id="range-end" type="date" aria-label="Custom range end" min="{first_trend_date.date().isoformat()}" max="{last_trend_date.date().isoformat()}" value="{last_trend_date.date().isoformat()}"><button type="submit">Apply</button></form></div>{trend}</article></section>
<section class="section"><div class="grid two-even"><article class="card chart-card"><div class="chart-title">Recent archive windows</div>{windows}</article><article class="card chart-card"><div class="chart-title">Stations ranked by 90-day archive shortfall</div><div class="chart-sub">IGRA archive records vs {html.escape(station_baseline_label)}</div>{station_shortfalls}</article></div></section>

<section id="stations" class="section"><div class="section-head"><div class="eyebrow">STATION STATUS</div><h2>Current NCO-reported issues</h2></div><div class="grid two"><article class="card"><img class="map" src="{map_uri}" alt="Miller-projection map of CONUS upper-air stations with state borders and latest NCO-reported status"></article><article class="card status-card"><div class="kpi-label">Latest mapped status</div><p class="kpi-detail">Stations with an NCO-reported issue</p><div class="kpi-value problem">{issue_count} / {active_count}</div><p class="kpi-detail">{clean_count} stations have no issue reported</p><div class="change-strip"><div class="change-stat new"><strong>{new_count}</strong><span>new or changed</span></div><div class="change-stat"><strong>{persistent_count}</strong><span>persistent</span></div><div class="change-stat resolved"><strong>{resolved_count}</strong><span>resolved</span></div></div><details class="station-search-details"><summary>Search all mapped stations</summary>{_station_directory(current_stations)}</details></article></div></section>

<section id="operations" class="section"><div class="section-head"><div class="eyebrow">OPERATIONAL MESSAGES</div><h2>NCO reported for ingest</h2></div><div class="grid two-even"><article class="card chart-card"><div class="chart-title">CONUS RAOBs reported for ingest</div><div class="chart-sub">Latest reported count: {int(kpis.nco_count or 0)} &middot; Default view: latest 1 year &middot; Smoothed line: 7 days</div><div class="trend-controls operation-controls"><div class="preset-controls" aria-label="NCO ingest date range"><button type="button" class="nco-range-button" data-days="28">1MO</button><button type="button" class="nco-range-button" data-days="182">6MO</button><button type="button" class="nco-range-button active" data-days="365">1YR</button><button type="button" class="nco-range-button" data-days="730">2YR</button></div><form id="nco-custom-range" class="custom-range"><span>Custom</span><input id="nco-range-start" type="date" aria-label="NCO custom range start" min="{nco_first.date().isoformat()}" max="{nco_last.date().isoformat()}" value="{nco_default_start.date().isoformat()}"><span>to</span><input id="nco-range-end" type="date" aria-label="NCO custom range end" min="{nco_first.date().isoformat()}" max="{nco_last.date().isoformat()}" value="{nco_last.date().isoformat()}"><button type="submit">Apply</button></form></div>{nco}</article><article class="card chart-card"><div class="chart-title">Reported issue categories</div><div class="chart-sub">Default view: latest 28 days</div><div class="trend-controls operation-controls"><form id="issue-custom-range" class="custom-range"><span>Custom</span><input id="issue-range-start" type="date" aria-label="Issue-category custom range start" min="{issue_first.date().isoformat()}" max="{issue_last.date().isoformat()}" value="{issue_default_start.date().isoformat()}"><span>to</span><input id="issue-range-end" type="date" aria-label="Issue-category custom range end" min="{issue_first.date().isoformat()}" max="{issue_last.date().isoformat()}" value="{issue_last.date().isoformat()}"><button type="submit">Apply</button></form></div>{categories}</article></div><article class="card review-card"><details><summary>Review station changes from the previous comparable cycle</summary>{_issue_rows(snapshot)}</details></article></section>

<section id="methods" class="section"><div class="section-head"><div><div class="eyebrow">SOURCES & METHODS</div><h2>Sources and data health</h2></div><p>Generated {html.escape(updated_text)}.</p></div><article class="card"><div class="eyebrow">DATA HEALTH / SOURCE COVERAGE</div><div class="health-grid"><div class="health-item"><strong class="{health_color}">{health_status}</strong><span>{health["required_ready"]} of {health["required_total"]} required products ready · {health["duplicate_rows"]} duplicate rows · {optional_health}</span></div><div class="health-item"><strong>IGRA daily archive</strong><span>{html.escape(igra_coverage)}</span></div><div class="health-item"><strong>NCO ingest counts</strong><span>{html.escape(nco_coverage)}</span></div><div class="health-item"><strong>NCO issue statuses</strong><span>{html.escape(issue_coverage)}</span></div></div><ul>{caveats}</ul><p class="kpi-detail">Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages; CONUS upper-air station master.</p><div class="downloads"><a class="download" href="archive-availability.csv" download>Download archive CSV</a><a class="download" href="latest-station-status.csv" download>Download station CSV</a><a class="download" href="nco-ingest-history.csv" download>Download NCO CSV</a></div></article></section>
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
const ncoChart=document.getElementById('nco-trend');
const ncoEnd=new Date('{nco_last.date().isoformat()}T00:00:00Z');
document.querySelectorAll('.nco-range-button').forEach(button=>button.addEventListener('click',()=>{{
  const start=new Date(ncoEnd);
  start.setUTCDate(start.getUTCDate()-(Number(button.dataset.days)-1));
  Plotly.relayout(ncoChart,{{'xaxis.range':[start.toISOString(),ncoEnd.toISOString()],'xaxis.autorange':false}});
  document.querySelectorAll('.nco-range-button').forEach(item=>item.classList.toggle('active',item===button));
}}));
document.getElementById('nco-custom-range')?.addEventListener('submit',()=>document.querySelectorAll('.nco-range-button').forEach(item=>item.classList.remove('active')));
const stationSearch=document.getElementById('station-search');
if(stationSearch){{stationSearch.addEventListener('input',()=>{{const query=stationSearch.value.trim().toLowerCase();let visible=0;document.querySelectorAll('.station-row').forEach(row=>{{const show=!query||row.dataset.search.includes(query);row.hidden=!show;if(show)visible++;}});document.getElementById('station-count').textContent=`${{visible}} station${{visible===1?'':'s'}}`;}});}}
</script></body></html>"""

    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    social_image = REPO_ROOT / "outputs" / "upper_air_network_monitor" / "social" / "original_dashboard_style.png"
    if social_image.exists():
        shutil.copy2(social_image, output_dir / "og.png")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_public_site(args.output_dir.resolve())
    print(f"Built public dashboard: {path}")


if __name__ == "__main__":
    main()
