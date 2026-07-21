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
        return "Latest day: —", "No complete NCO ingest day is available."
    latest = daily.sort_values("date").iloc[-1]
    received = float(latest.received) if pd.notna(latest.received) else float("nan")
    expected = float(latest.expected) if pd.notna(latest.expected) else float("nan")
    percent = float(latest.percent) if pd.notna(latest.percent) else float("nan")
    date_text = _display_date(latest.date)
    if pd.isna(received):
        return "Latest day: —", f"Complete through {date_text}"
    if pd.isna(expected) or not expected:
        return f"Latest day: {received:.0f} received across {int(latest.available_rows)} product records", f"Complete through {date_text}; expected inventory unavailable"
    reference = f"{reference_count}-station reference per applicable record" if reference_count else "station reference per applicable record"
    return f"Latest day: {received:.0f} received across {int(latest.available_rows)} product records · {percent:.1f}%", f"Complete through {date_text} · {reference}"


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
        '<div class="nco-ingest-subtitle">NCO operational-message availability · product records, not unique-station counts</div>'
        f'<div class="nco-latest" id="nco-latest">{html.escape(latest_text)}</div>'
        f'<div id="nco-latest-detail" class="nco-latest-detail">{html.escape(latest_detail)}</div></div>'
        '<div class="nco-view-controls"><button type="button" id="nco-one-year" class="nco-view-button active">1Y</button>'
        '<button type="button" id="nco-custom-toggle" class="nco-view-button" aria-expanded="false" aria-controls="nco-heatmap-custom">Custom</button></div></div>'
        '<div class="nco-cycle-controls" role="group" aria-label="NCO cycle view"><span>Cycle</span><button type="button" class="nco-cycle-button active" data-cycle-view="combined">Combined</button><button type="button" class="nco-cycle-button" data-cycle-view="00Z">00Z</button><button type="button" class="nco-cycle-button" data-cycle-view="12Z">12Z</button></div>'
        f'<div class="nco-freshness{" stale" if stale else ""}"><strong>Latest source record:</strong> {html.escape(latest_record_text)} · {html.escape(refresh_text)}{(" · Using retained valid data" if stale else "")}</div>'
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
  …10377 tokens truncated…dden="true"></span><span>Fri</span><span aria-hidden="true"></span><span aria-hidden="true"></span></div><div class="nco-heatmap-grid" style="--nco-week-count:'+weeks+'">'+cells+'</div></div>';
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
const stationSearch=document.getElementById('station-search');
if(stationSearch){{stationSearch.addEventListener('input',()=>{{const query=stationSearch.value.trim().toLowerCase();let visible=0;document.querySelectorAll('.station-row').forEach(row=>{{const show=!query||row.dataset.search.includes(query);row.hidden=!show;if(show)visible++;}});document.getElementById('station-count').textContent=`${{visible}} station${{visible===1?'':'s'}}`;}});}}
</script></body></html>"""

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

