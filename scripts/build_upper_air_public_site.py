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


def _archive_gap_display(value: object) -> tuple[str, str]:
    """Return a signed gap label and semantic color class.

    Positive values are archive surpluses and use the dashboard's clean
    (green) token; negative values are shortfalls and use the problem (red)
    token. Missing or non-finite values remain neutral rather than implying a
    zero gap.
    """
    try:
        gap = float(value)
    except (TypeError, ValueError):
        return "Ã¢â‚¬â€", ""
    if not math.isfinite(gap):
        return "Ã¢â‚¬â€", ""
    color_class = "clean" if gap > 0 else "problem" if gap < 0 else ""
    return f"{gap:+.1f}%", color_class


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
        category = str(row.issue_category).replace("_", " ").title() if pd.notna(row.issue_category) and row.issue_category else "Ã¢â‚¬â€"
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
    axis.text(0.06, 0.12, "soundings.wall.cloud  Ã‚Â·  wall.cloud", color="#6b8190",
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
    return f"{_display_date(start)} Ã¢â‚¬â€œ {_display_date(end)}"


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
    return latest_text, f"Last successful NCO refresh: {refresh_text} Ã‚Â· Updated {age_text}", stale


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
        return "Latest: Ã¢â‚¬â€", "No complete NCO day available"
    latest = daily.sort_values("date").iloc[-1]
    received = float(latest.received) if pd.notna(latest.received) else float("nan")
    expected = float(latest.expected) if pd.notna(latest.expected) else float("nan")
    percent = float(latest.percent) if pd.notna(latest.percent) else float("nan")
    date_text = _display_date(latest.date)
    if pd.isna(received):
        return "Latest: Ã¢â‚¬â€", f"Complete through {date_text}"
    if pd.isna(expected) or not expected:
        return f"Latest: {received:.0f} of {int(latest.available_rows)} product records", f"Complete through {date_text} Ã‚Â· expected inventory unavailable"
    return f"Latest: {received:.0f} of {int(expected)} expected product records Ã‚Â· {percent:.1f}%", f"Complete through {date_text} Ã‚Â· {int(latest.available_rows)} applicable product records"


def _nco_heatmap_markup(
    views: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    source_start: pd.Timestamp,
    source_end: pd.Timestamp,
    default_start: pd.Timestamp,
    default_end: pd.Tim×İ´âÚ$z{-®éÜj×¢6öç7Bæ6ô7–6ÆU7C×fÇVSÓäçVÖ&W"æ—4f–æ—FR„çVÖ&W"‡fÇVR’“ôçVÖ&W"‡fÇVR’çFôf—†VBƒ’²rRs¢|:.(*Î(	Òs°¢6öç7Bæ6ô7–6ÆT†VÇFƒ×fÇVSÓç·¶6öç7BçVÖ&W#ÔçVÖ&W"‡fÇVR“¶–b‚çVÖ&W"æ—4f–æ—FR†çVÖ&W"’—&WGW&âvæòÖFFs¶–b†çVÖ&W#ãÓ“‚—&WGW&âv†VÇF‚Ö†VÇF‡’s¶–b†çVÖ&W#ãÓ“R—&WGW&âv†VÇF‚ÖÖ–æ÷"s¶–b†çVÖ&W#ãÓ“—&WGW&âv†VÇF‚×&VGV6VBs¶–b†çVÖ&W#ãÓƒ—&WGW&âv†VÇF‚ÖFVw&FVBs·&WGW&âv†VÇF‚Ö7&—F–6Âs·×Ó°¢6öç7Bæ6ô7–6ÆU&FSÒ‡7F'BÆVæB“Óç·¶ÆWB&V6V—fVCÓÆW‡V7FVCÓÆF—3Ó¶f÷"†6öç7B&÷röbæ6ô7–6ÆU&÷w2‚’—·¶–b‡&÷ræFFSã×7F'Bbg&÷ræFFSÃÖVæBbdçVÖ&W"æ—4f–æ—FR„çVÖ&W"‡&÷rç&V6V—fVB’’bdçVÖ&W"‡&÷ræW‡V7FVB“ã—··&V6V—fVB³ÔçVÖ&W"‡&÷rç&V6V—fVB“¶W‡V7FVB³ÔçVÖ&W"‡&÷ræW‡V7FVB“¶F—2²³·××××&WGW&â··&FS¦W‡V7FVC÷&V6V—fVBöW‡V7FVB£¦çVÆÂÆF—7×Ó·×Ó°¢6öç7Bæ6ô7–6ÆTFVÇFÒ‡7F'BÆVæB“Óç·¶6öç7B7W'&VçCÖæ6ô7–6ÆU&FR‡7F'BÆVæB“¶6öç7BÖæWrFFR‡7F'B²uC££¢r“¶6öç7B#ÖæWrFFR†VæB²uC££¢r“¶6öç7BÆVæwFƒÔÖF‚ç&÷VæB‚†"Ö’óƒcC’³¶6öç7B&Wf–÷W4VæCÖæWrFFR†“·&Wf–÷W4VæBç6WEUD4FFR‡&Wf–÷W4VæBævWEUD4FFR‚’Ó“¶6öç7B&Wf–÷W57F'CÖæWrFFR‡&Wf–÷W4VæB“·&Wf–÷W57F'Bç6WEUD4FFR‡&Wf–÷W57F'BævWEUD4FFR‚’Ò†ÆVæwF‚Ó’“¶6öç7B&Wf–÷W3Öæ6ô7–6ÆU&FR†æ6ô7–6ÆT—6ò‡&Wf–÷W57F'B’Ææ6ô7–6ÆT—6ò‡&Wf–÷W4VæB’“·&WGW&â·¶7W'&VçBÇ&Wf–÷W2ÆFVÇF¦7W'&VçBç&FRÓÖçVÆÂbg&Wf–÷W2ç&FRÓÖçVÆÃö7W'&VçBç&FR×&Wf–÷W2ç&FS¦çVÆÂÇ&Wf–÷W57F'C¦æ6ô7–6ÆT—6ò‡&Wf–÷W57F'B’Ç&Wf–÷W4VæC¦æ6ô7–6ÆT—6ò‡&Wf–÷W4VæB—×Ó·×Ó°¢6öç7Bæ6ô7–6ÆTÖöFVÇ3×&÷sÓäö&¦V7BæVçG&–W2‡&÷sòæÖöFVÇ7ÇÇ··×Ò’æf–ÇFW"‚…²ÇfÇVUÒ“ÓçfÇVRÓÖçVÆÂbdçVÖ&W"æ—4f–æ—FR„çVÖ&W"‡fÇVR’’’æÖ‚…¶ÖöFVÂÇfÇVUÒ“ÓæÖöFVÂ²r&W÷'C¢r´çVÖ&W"‡fÇVR’çFôf—†VBƒ’“°¢6öç7Bæ6ô7–6ÆTFWF–Ä‡FÖÃÒ†FFRÇ&÷r“Óç·¶–b‚&÷r—&WGW&âsÇ7G&öæsâr¶æ6ô7–6ÆTf×DFFR†FFR’²sÂ÷7G&öæsãÇ7ãäæòFF:.(*Î(	ÒÖöæ—F÷&–ær&V6÷&BVæf–Æ&ÆRãÂ÷7ãâs¶6öç7B6÷W&6W3Öæ6ô7–6ÆTÖöFVÇ2‡&÷r“·&WGW&âsÇ7G&öæsâr¶æ6ô7–6ÆTf×DFFR†FFR’²sÂ÷7G&öæsãÇ7ãâr¶æ6ô7–6ÆU7B‡&÷rçW&6VçB’²r8,+rr´çVÖ&W"‡&÷rç&V6V—fVB’çFôf—†VBƒ’²röbr²„çVÖ&W"æ—4f–æ—FR„çVÖ&W"‡&÷ræW‡V7FVB’“ôçVÖ&W"‡&÷ræW‡V7FVB’çFôf—†VBƒ“¢|:.(*Î(	Òr’²rW‡V7FVB&öGV7B&V6÷&G3Â÷7ããÇ7â6Æ73Ò&æ6òÖ6VÆÂÖÖöFVÇ2#âr²‡6÷W&6W2æÆVæwFƒ÷6÷W&6W2æ¦ö–â‚r8,+rr“¢tæòÆ–6&ÆR6÷W&6R&V6÷&Br’²sÂ÷7ãâs·×Ó°¢6öç7Bæ6ô7–6ÆU6†÷tFWF–ÃÒ†FFRÇ&÷r“Óç·¶–b†æ6ô7–6ÆTFWF–Â—·¶æ6ô7–6ÆTFWF–Âæ–ææW$…DÔÃÖæ6ô7–6ÆTFWF–Ä‡FÖÂ†FFRÇ&÷r’²sÆ'WGFöâG—SÒ&'WGFöâ"6Æ73Ò&æ6òÖFWF–ÂÖ6Æ÷6R"&–ÖÆ&VÃÒ$F—6Ö—726VÆV7FVBF’FWF–Ç2#ì8>(	CÂö'WGFöãâs¶æ6ô7–6ÆTFWF–Âæ†–FFVãÖfÇ6S·×××Ó°¢6öç7Bæ6ô7–6ÆU6†÷uFööÇF—Ò†6VÆÂ“Óç·¶–b‚æ6ô7–6ÆUFööÇF——&WGW&ã¶æ6ô7–6ÆUFööÇF—æ–ææW$…DÔÃÖæ6ô7–6ÆTFWF–Ä‡FÖÂ†6VÆÂæFF6WBæFFRÆæ6ô7–6ÆTFFU&÷w2‚’ævWB†6VÆÂæFF6WBæFFR’“¶6öç7B&V7CÖ6VÆÂævWD&÷VæF–æt6Æ–VçE&V7B‚“¶æ6ô7–6ÆUFööÇF—ç7G–ÆRæÆVgCÔÖF‚æÖ–â„ÖF‚æÖ‚ƒ‚Ç&V7BæÆVgB’ÄÖF‚æÖ‚ƒ‚Çv–æF÷ræ–ææW%v–GF‚Ó#3‚’’²w‚s¶æ6ô7–6ÆUFööÇF—ç7G–ÆRçF÷ÔÖF‚æÖ‚ƒ‚Ç&V7BçF÷Ócb’²w‚s¶æ6ô7–6ÆUFööÇF—æ†–FFVãÖfÇ6S·×Ó°¢6öç7Bæ6ô7–6ÆT&–æCÒ‚“Óç·¶æ6ô7–6ÆT†VFÖòçVW'•6VÆV7F÷$ÆÂ‚v'WGFöå¶FFÖFFUÒr’æf÷$V6‚†6VÆÃÓç·¶–b†6VÆÂæFF6WBæ7–6ÆT&÷VæB—&WGW&ã¶6VÆÂæFF6WBæ7–6ÆT&÷VæCÒss¶6VÆÂæFDWfVçDÆ—7FVæW"‚vÖ÷W6VVçFW"rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶æ6ô7–6ÆU6†÷uFööÇF—†6VÆÂ“·×ÒÇG'VR“¶6VÆÂæFDWfVçDÆ—7FVæW"‚vÖ÷W6VÆVfRrÂ‚“Óç·¶–b†æ6ô7–6ÆUFööÇF—–æ6ô7–6ÆUFööÇF—æ†–FFVã×G'VS·×ÒÇG'VR“¶6VÆÂæFDWfVçDÆ—7FVæW"‚vfö7W2rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶æ6ô7–6ÆU6†÷uFööÇF—†6VÆÂ“·×ÒÇG'VR“¶6VÆÂæFDWfVçDÆ—7FVæW"‚v&ÇW"rÂ‚“Óç·¶–b†æ6ô7–6ÆUFööÇF—–æ6ô7–6ÆUFööÇF—æ†–FFVã×G'VS·×ÒÇG'VR“¶6VÆÂæFDWfVçDÆ—7FVæW"‚v6Æ–6²rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶æ6ô7–6ÆU6†÷tFWF–Â†6VÆÂæFF6WBæFFRÆæ6ô7–6ÆTFFU&÷w2‚’ævWB†6VÆÂæFF6WBæFFR’“·×ÒÇG'VR“·×Ò“·×Ó°¢6öç7Bæ6ô7–6ÆUWFFTÖWG&–73Ò‚“Óç·¶6öç7B&÷w3Öæ6ô7–6ÆU&÷w2‚“¶6öç7BÆFW7C×&÷w2æÆVæwFƒ÷&÷w5·&÷w2æÆVæwF‚ÓÓ¦çVÆÃ¶–b†æ6ô7–6ÆTÆFW7B—·¶æ6ô7–6ÆTÆFW7BçFW‡D6öçFVçCÖÆFW7CòtÆFW7BF“¢r´çVÖ&W"†ÆFW7Bç&V6V—fVB’çFôf—†VBƒ’²r&V6V—fVB7&÷72r´çVÖ&W"†ÆFW7Bæf–Æ&ÆU÷&÷w2’çFôf—†VBƒ’²r&öGV7B&V6÷&G28,+rr¶æ6ô7–6ÆU7B†ÆFW7BçW&6VçB“¢tÆFW7BF“¢:.(*Î(	Òs·×Ö–b†æ6ô7–6ÆTÆFW7DFWF–ÂbfÆFW7B—·¶æ6ô7–6ÆTÆFW7DFWF–ÂçFW‡D6öçFVçCÒt6ö×ÆWFRF‡&÷Vv‚r¶æ6ô7–6ÆTf×DFFR†ÆFW7BæFFR’²r8,+rc’×7FF–öâ&VfW&Væ6RW"Æ–6&ÆR&V6÷&Bs·×Öf÷"†6öç7BF—2öb³rÃBÃ3Ã“Ò—·¶6öç7BVæCÖÆFW7CòæFFWÇÆæ6ô7F—fTVæC¶6öç7BVæDFFSÖæWrFFR†VæB²uC££¢r“¶6öç7B7F'DFFSÖæWrFFR†VæDFFR“·7F'DFFRç6WEUD4FFR‡7F'DFFRævWEUD4FFR‚’Ò†F—2Ó’“¶6öç7B&W7VÇCÖæ6ô7–6ÆTFVÇF†æ6ô7–6ÆT—6ò‡7F'DFFR’ÆVæB“¶6öç7B7W'&VçCÖFö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖÖWG&–2Ö7W'&VçBÒr¶F—2“¶6öç7BFVÇFÖFö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖÖWG&–2ÖFVÇFÒr¶F—2“¶–b†7W'&VçB–7W'&VçBçFW‡D6öçFVçCÖæ6ô7–6ÆU7B‡&W7VÇBæ7W'&VçBç&FR“¶–b†FVÇF–FVÇFçFW‡D6öçFVçC×&W7VÇBæFVÇFÓÓÖçVÆÃò|:.(*Î(	Òs¢‡&W7VÇBæFVÇFãÓòr²s¢|:,¸n(	’r’´ÖF‚æ'2‡&W7VÇBæFVÇF’çFôf—†VBƒ’²rs·×××Ó°¢6öç7Bæ6ô7–6ÆU&VæFW#Ò‡7F'D—6òÆVæD—6ò“Óç·¶æ6ô7F—fU7F'C×7F'D—6ó¶æ6ô7F—fTVæCÖVæD—6ó¶6öç7B7F'CÖæWrFFR‡7F'D—6ò²uC££¢r“¶6öç7BVæCÖæWrFFR†VæD—6ò²uC££¢r“¶6öç7BÖöæF“ÖæWrFFR‡7F'B“¶ÖöæF’ç6WEUD4FFR†ÖöæF’ævWEUD4FFR‚’Ò‚†ÖöæF’ævWEUD4F’‚’³b’Sr’“¶6öç7B7VæF“ÖæWrFFR†VæB“·7VæF’ç6WEUD4FFR‡7VæF’ævWEUD4FFR‚’²ƒbÒ‚‡7VæF’ævWEUD4F’‚’³b’Sr’’“¶6öç7BvVV·3ÔÖF‚ç&÷VæB‚‡7VæF’ÖÖöæF’’óƒcCór’³¶æ6ô7–6ÆT†VFÖç7G–ÆRç6WE&÷W'G’‚rÒÖæ6ò×vVV²Ö6÷VçBrÇvVV·2“¶6öç7BÖöçF„—FV×3ÕµÓ¶6öç7B7W'6÷#ÖæWrFFR„FFRåUD2‡7F'BævWEUD4gVÆÅ–V"‚’Ç7F'BævWEUD4ÖöçF‚‚’Ã’“·v†–ÆR†7W'6÷#ÃÖVæB—·¶6öç7BÆ&VÄFFSÖ7W'6÷#Ç7F'C÷7F'C¦7W'6÷#¶6öç7BvVV³ÔÖF‚æfÆö÷"‚†Æ&VÄFFRÖÖöæF’’óƒcCór“¶–b‡vVV³ãÓbgvVV³ÇvVV·2–ÖöçF„—FV×2çW6‚‡··vVV²ÆÆ&VÃ¦7W'6÷"çFôÆö6ÆU7G&–ær‚vVâÕU2rÇ·¶ÖöçFƒ¢w6†÷'BrÇF–ÖU¦öæS¢uUD2w×Ò—×Ò“¶7W'6÷"ç6WEUD4ÖöçF‚†7W'6÷"ævWEUD4ÖöçF‚‚’³“·×Ö6öç7BÖöçF„Æ&VÇ3ÖÖöçF„—FV×2æf–ÇFW"‚†—FVÒÆ–æFW‚“Óæ–æFWƒÓÓÓÇÆ—FVÒçvVV²ÓÖÖöçF„—FV×5¶–æFW‚ÓÒçvVV·ÇÂ†—FVÒæÆ&VÂÓÖÖöçF„—FV×5¶–æFW‚ÓÒæÆ&VÇÇÆ—FVÒçvVV²ÖÖöçF„—FV×5¶–æFW‚ÓÒçvVV³ãÓB’’æÖ†—FVÓÓâsÇ7â7G–ÆSÒ&w&–BÖ6öÇVÖã¢r²†—FVÒçvVV²³’²r#âr¶—FVÒæÆ&VÂ²sÂ÷7ãâr’æ¦ö–â‚rr“¶ÆWB6VÆÇ3Òrs¶6öç7B&÷w3Öæ6ô7–6ÆTFFU&÷w2‚“¶f÷"†ÆWB&÷sÓ·&÷sÃs·&÷r²²–f÷"†ÆWB6öÇVÖãÓ¶6öÇVÖãÇvVV·3¶6öÇVÖâ²²—·¶6öç7BF“ÖæWrFFR†ÖöæF’“¶F’ç6WEUD4FFR†F’ævWEUD4FFR‚’¶6öÇVÖâ£r·&÷r“¶6öç7BFFSÖæ6ô7–6ÆT—6ò†F’“¶–b†F“Ç7F'GÇÆF“æVæB—·¶6VÆÇ2³ÒsÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ãâs¶6öçF–çVS·×Ö6öç7BfÇVS×&÷w2ævWB†FFR“¶6öç7B6÷W&6W3Öæ6ô7–6ÆTÖöFVÇ2‡fÇVR“¶6öç7BÆ&VÃ×fÇVSöFFR²rr¶æ6ô7–6ÆU7B‡fÇVRçW&6VçB’²rr´çVÖ&W"‡fÇVRç&V6V—fVB’çFôf—†VBƒ’²röbr²„çVÖ&W"æ—4f–æ—FR„çVÖ&W"‡fÇVRæW‡V7FVB’“ôçVÖ&W"‡fÇVRæW‡V7FVB’çFôf—†VBƒ“¢væòW‡V7FVBF÷FÂr’²rW‡V7FVB&öGV7B&V6÷&G2r²‡6÷W&6W2æÆVæwFƒòr8,+rr·6÷W&6W2æ¦ö–â‚r8,+rr“¢rr“¦FFR²ræòFFs¶6VÆÇ2³ÒsÆ'WGFöâG—SÒ&'WGFöâ"6Æ73Ò&æ6òÖ6VÆÂr¶æ6ô7–6ÆT†VÇF‚‡fÇVSòçW&6VçB’²r"FFÖFFSÒ"r¶FFR²r"&–ÖÆ&VÃÒ"r¶Æ&VÂç&WÆ6R‚ò"örÂrgV÷C²r’²r#ãÂö'WGFöãâs·×Öæ6ô7–6ÆT†VFÖæ–ææW$…DÔÃÒsÆF—b6Æ73Ò&æ6òÖÖöçF‡2"7G–ÆSÒ"ÒÖæ6ò×vVV²Ö6÷VçC¢r·vVV·2²r#âr¶ÖöçF„Æ&VÇ2²sÂöF—cãÆF—b6Æ73Ò&æ6òÖ†VFÖÖ&öG’#ãÆF—b6Æ73Ò&æ6ò×vVV¶F’ÖÆ&VÇ2#ãÇ7ãäÖöãÂ÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÇ7ãåvVCÂ÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÇ7ãäg&“Â÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÂöF—cãÆF—b6Æ73Ò&æ6òÖ†VFÖÖw&–B"7G–ÆSÒ"ÒÖæ6ò×vVV²Ö6÷VçC¢r·vVV·2²r#âr¶6VÆÇ2²sÂöF—cãÂöF—câs¶æ6ô7–6ÆT&–æB‚“¶æ6ô7–6ÆUWFFTÖWG&–72‚“·×Ó°¢6öç7Bæ6ô7–6ÆU6VÆV7CÖ7–6ÆSÓç·¶æ6ô7F—fT7–6ÆSÖ7–6ÆS¶Fö7VÖVçBçVW'•6VÆV7F÷$ÆÂ‚rææ6òÖ7–6ÆRÖ'WGFöâr’æf÷$V6‚†'WGFöãÓæ'WGFöâæ6Æ74Æ—7BçFövvÆR‚v7F—fRrÆ'WGFöâæFF6WBæ7–6ÆUf–WsÓÓÖ7–6ÆR’“¶6öç7B&÷w3Öæ6ô7–6ÆU&÷w2‚“¶6öç7BÆFW7C×&÷w2æÆVæwFƒ÷&÷w5·&÷w2æÆVæwF‚ÓÒæFFS¦æ6ô7F—fTVæC¶6öç7BW6T7W7FöÓÖFö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ7W7FöÒ×FövvÆRr“òæ6Æ74Æ—7Bæ6öçF–ç2‚v7F—fRr“¶æ6ô7–6ÆU&VæFW"‡W6T7W7FöÓöæ6ô7–6ÆU7F'BçfÇVS¦æ6ô7F—fU7F'BÂW6T7W7FöÓöæ6ô7–6ÆTVæBçfÇVS¦ÆFW7B“·×Ó°¢Fö7VÖVçBçVW'•6VÆV7F÷$ÆÂ‚rææ6òÖ7–6ÆRÖ'WGFöâr’æf÷$V6‚†'WGFöãÓæ'WGFöâæFDWfVçDÆ—7FVæW"‚v6Æ–6²rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶æ6ô7–6ÆU6VÆV7B†'WGFöâæFF6WBæ7–6ÆUf–Wr“·×ÒÇG'VR’“°¢Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ†VFÖÖÇ’r“òæFDWfVçDÆ—7FVæW"‚v6Æ–6²rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶6öç7B7F'CÖæ6ô7–6ÆU7F'BçfÇVRÆVæCÖæ6ô7–6ÆTVæBçfÇVS¶–b‚7F'GÇÂVæGÇÇ7F'CæVæGÇÇ7F'CÆæ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFWÇÆVæCææ6õ–ÆöDVÆVÖVçBæFF6WBæÖ„FFR—·¶æ6ô7–6ÆU7F'Bç6WD7W7FöÕfÆ–F—G’‚t6†ö÷6RFFW2v—F†–âF†Rf–Æ&ÆR&ævRÂv—F‚7F'Böâ÷"&Vf÷&RVæBâr“¶æ6ô7–6ÆU7F'Bç&W÷'EfÆ–F—G’‚“·&WGW&ã·×Öæ6ô7–6ÆU7F'Bç6WD7W7FöÕfÆ–F—G’‚rr“¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖöæR×–V"r“òæ6Æ74Æ—7Bç&VÖ÷fR‚v7F—fRr“¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ7W7FöÒ×FövvÆRr“òæ6Æ74Æ—7BæFB‚v7F—fRr“¶æ6ô7–6ÆU&VæFW"‡7F'BÆVæB“¶6öç7B&W7VÇCÖæ6ô7–6ÆTFVÇF‡7F'BÆVæB“¶–b†æ6ô7–6ÆU7VÖÖ'’—·¶æ6ô7–6ÆU7VÖÖ'’æ†–FFVãÖfÇ6S¶æ6ô7–6ÆU7VÖÖ'’çFW‡D6öçFVçCÒu6VÆV7FVB&ævS¢r¶æ6ô7–6ÆU7B‡&W7VÇBæ7W'&VçBç&FR’²r8,+r&Wf–÷W2WVÂ&ævS¢r¶æ6ô7–6ÆU7B‡&W7VÇBç&Wf–÷W2ç&FR’²r8,+r6†ævS¢r²‡&W7VÇBæFVÇFÓÓÖçVÆÃò|:.(*Î(	Òs¢‡&W7VÇBæFVÇFãÓòr²s¢|:,¸n(	’r’´ÖF‚æ'2‡&W7VÇBæFVÇF’çFôf—†VBƒ’²rr“·×××ÒÇG'VR“°¢Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖöæR×–V"r“òæFDWfVçDÆ—7FVæW"‚v6Æ–6²rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ†VFÖÖ7W7FöÒr’æ†–FFVã×G'VS¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ7W7FöÒ×FövvÆRr“òæ6Æ74Æ—7Bç&VÖ÷fR‚v7F—fRr“¶æ6ô7–6ÆU7VÖÖ'’æ†–FFVã×G'VS¶6öç7B&÷w3Öæ6ô7–6ÆU&÷w2‚“¶6öç7BVæC×&÷w2æÆVæwFƒ÷&÷w5·&÷w2æÆVæwF‚ÓÒæFFS¦æ6õ–ÆöDVÆVÖVçBæFF6WBæÖ„FFS¶6öç7BVæDFFSÖæWrFFR†VæB²uC££¢r“¶6öç7B7F'DFFSÖæWrFFR†VæDFFR“·7F'DFFRç6WEUD4FFR‡7F'DFFRævWEUD4FFR‚’Ó3cB“¶6öç7B7F'CÖæ6ô7–6ÆT—6ò‡7F'DFFR“Ææ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFSöæ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFS¦æ6ô7–6ÆT—6ò‡7F'DFFR“¶æ6ô7–6ÆU&VæFW"‡7F'BÆVæB“·×ÒÇG'VR“°¢Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ†VFÖ×&W6WBr“òæFDWfVçDÆ—7FVæW"‚v6Æ–6²rÆWfVçCÓç·¶WfVçBç7F÷–ÖÖVF–FU&÷vF–öâ‚“¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ†VFÖÖ7W7FöÒr’æ†–FFVã×G'VS¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚væ6òÖ7W7FöÒ×FövvÆRr“òæ6Æ74Æ—7Bç&VÖ÷fR‚v7F—fRr“¶æ6ô7–6ÆU7VÖÖ'’æ†–FFVã×G'VS¶6öç7B&÷w3Öæ6ô7–6ÆU&÷w2‚“¶6öç7BVæC×&÷w2æÆVæwFƒ÷&÷w5·&÷w2æÆVæwF‚ÓÒæFFS¦æ6õ–ÆöDVÆVÖVçBæFF6WBæÖ„FFS¶6öç7BVæDFFSÖæWrFFR†VæB²uC££¢r“¶6öç7B7F'DFFSÖæWrFFR†VæDFFR“·7F'DFFRç6WEUD4FFR‡7F'DFFRævWEUD4FFR‚’Ó3cB“¶6öç7B7F'CÖæ6ô7–6ÆT—6ò‡7F'DFFR“Ææ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFSöæ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFS¦æ6ô7–6ÆT—6ò‡7F'DFFR“¶æ6ô7–6ÆU&VæFW"‡7F'BÆVæB“·×ÒÇG'VR“°¢6öç7Bæ6ô–æ—F–Å&÷w3Öæ6ô7–6ÆU&÷w2‚“¶6öç7Bæ6ô–æ—F–ÄVæCÖæ6ô–æ—F–Å&÷w2æÆVæwFƒöæ6ô–æ—F–Å&÷w5¶æ6ô–æ—F–Å&÷w2æÆVæwF‚ÓÒæFFS¢†æ6ô7–6ÆTVæCòçfÇVWÇÆæ6õ–ÆöDVÆVÖVçBæFF6WBæÖ„FFR“¶6öç7Bæ6ô–æ—F–ÄFFSÖæWrFFR†æ6ô–æ—F–ÄVæB²uC££¢r“¶6öç7Bæ6ô–æ—F–Å7F'DFFSÖæWrFFR†æ6ô–æ—F–ÄFFR“¶æ6ô–æ—F–Å7F'DFFRç6WEUD4FFR†æ6ô–æ—F–Å7F'DFFRævWEUD4FFR‚’Ó3cB“¶6öç7Bæ6ô–æ—F–Å7F'CÖæ6ô7–6ÆT—6ò†æ6ô–æ—F–Å7F'DFFR“Ææ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFSöæ6õ–ÆöDVÆVÖVçBæFF6WBæÖ–äFFS¦æ6ô7–6ÆT—6ò†æ6ô–æ—F–Å7F'DFFR“¶æ6ô7–6ÆU&VæFW"†æ6ô–æ—F–Å7F'BÆæ6ô–æ—F–ÄVæB“°§×Ğ¦Fö7VÖVçBçVW'•6VÆV7F÷$ÆÂ‚rç7FF–öâ×v–æF÷rÖ6öçG&öÇ2r’æf÷$V6‚†6öçG&öÃÓæ6öçG&öÂæFDWfVçDÆ—7FVæW"‚v6Æ–6²rÆWfVçCÓç·°¢6öç7B'WGFöãÖWfVçBçF&vWBæ6Æ÷6W7B‚rç7FF–öâ×v–æF÷rÖ'WGFöâr“°¢–b‚'WGFöâ—&WGW&ã°¢6öç7B6&CÖ6öçG&öÂæ6Æ÷6W7B‚ræ6&Br“°¢6öç7Bv–æF÷tF—3Ö'WGFöâæFF6WBçv–æF÷s°¢6&BçVW'•6VÆV7F÷$ÆÂ‚rç7FF–öâ×v–æF÷rÖ'WGFöâr’æf÷$V6‚†—FVÓÓç·¶6öç7B7F—fSÖ—FVÓÓÓÖ'WGFöã¶—FVÒæ6Æ74Æ—7BçFövvÆR‚v7F—fRrÆ7F—fR“¶—FVÒç6WDGG&–'WFR‚v&–×&W76VBrÅ7G&–ær†7F—fR’“·×Ò“°¢6&BçVW'•6VÆV7F÷$ÆÂ‚rç7FF–öâ×&æ¶–ær×æVÂr’æf÷$V6‚‡æVÃÓç··æVÂæ†–FFVã×æVÂæFF6WBçv–æF÷rÓ×v–æF÷tF—3·×Ò“°¢6&BçVW'•6VÆV7F÷$ÆÂ‚rç7FF–öâ×&æ¶–ær×æVÃ¦æ÷B…¶†–FFVåÒ’æ§2×Æ÷FÇ’×Æ÷Br’æf÷$V6‚†6†'CÓç·¶–b‡v–æF÷råÆ÷FÇ’•Æ÷FÇ’åÆ÷G2ç&W6—¦R†6†'B“·×Ò“°§×Ò’“°¦6öç7B7FF–öå6V&6ƒÖFö7VÖVçBævWDVÆVÖVçD'”–B‚w7FF–öâ×6V&6‚r“°¦–b‡7FF–öå6V&6‚—··7FF–öå6V&6‚æFDWfVçDÆ—7FVæW"‚v–çWBrÂ‚“Óç·¶6öç7BVW'“×7FF–öå6V&6‚çfÇVRçG&–Ò‚’çFôÆ÷vW$66R‚“¶ÆWBf—6–&ÆSÓ¶Fö7VÖVçBçVW'•6VÆV7F÷$ÆÂ‚rç7FF–öâ×&÷rr’æf÷$V6‚‡&÷sÓç·¶6öç7B6†÷sÒVW'—ÇÇ&÷ræFF6WBç6V&6‚æ–æ6ÇVFW2‡VW'’“·&÷ræ†–FFVãÒ6†÷s¶–b‡6†÷r—f—6–&ÆR²³·×Ò“¶Fö7VÖVçBævWDVÆVÖVçD'”–B‚w7FF–öâÖ6÷VçBr’çFW‡D6öçFVçCÖG··f—6–&ÆW×Ò7FF–öâG··f—6–&ÆSÓÓÓòrs¢w2w×Ö·×Ò“·×ĞĞ£Â÷67&—CãÂö&öG“ãÂö‡FÖÃâ"" Ğ Ğ¢2F†R7–6ÆR×6VÆV7F÷"&VæFW&W"—2–çFVçF–öæÆÇ’¶WB–âF†R–æÆ–æRvP¢267&—Bâæ÷&ÖÆ—¦R—G2vVV¶F’Ö&·W†W&R6ò&÷F‚&VæFW"F‡26†÷rF†P¢26ö×ÆWFR6ÆVæF"–ç7FVBöbF†RöÆFW"ÖöâõvVBôg&’ÖöæÇ’Æ&VÇ2à¢vRÒvRç&WÆ6R€¢sÆF—b6Æ73Ò&æ6ò×vVV¶F’ÖÆ&VÇ2#ãÇ7ãäÖöãÂ÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÇ7ãåvVCÂ÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÇ7ãäg&“Â÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÇ7â&–Ö†–FFVãÒ'G'VR#ãÂ÷7ããÂöF—cârÀ¢sÆF—b6Æ73Ò&æ6ò×vVV¶F’ÖÆ&VÇ2"&–ÖÆ&VÃÒ%vVV¶F—2#ãÇ7ãäÓÂ÷7ããÇ7ãåCÂ÷7ããÇ7ãåsÂ÷7ããÇ7ãåFƒÂ÷7ããÇ7ãäcÂ÷7ããÇ7ãå6Â÷7ããÇ7ãå7SÂ÷7ããÂöF—cârÀ¢¢vRÒvRç&WÆ6R€¢sÂö†VCârÀ¢sÇ7G–ÆSâç7FF–öâ×&æ¶–ærÖw&–G¶Ö&v–â×F÷£G‡Òç7FF–öâ×v–æF÷rÖ6öçG&öÇ7¶F—7Æ“¦fÆWƒ¶v£Gƒ¶fÆW‚×w&§w&·FF–æs£‡‚—‚'‡Òç7FF–öâ×v–æF÷rÖ'WGFöç¶&÷&FW#£‚6öÆ–Bf"‚ÒÖÆ–æR“¶&6¶w&÷VæC§f"‚ÒÖ&r“¶6öÆ÷#§f"‚ÒÖ×WFVB“¶&÷&FW"×&F—W3£wƒ·FF–æs£W‚‡ƒ¶7W'6÷#§ö–çFW#¶föçB×6—¦S£ƒ¶föçB×vV–v‡C£sSÒç7FF–öâ×v–æF÷rÖ'WGFöã¦†÷fW"Âç7FF–öâ×v–æF÷rÖ'WGFöâæ7F—fRÂç7FF–öâ×v–æF÷rÖ'WGFöã¦fö7W2×f—6–&ÆW¶6öÆ÷#§f"‚Ò×FW‡B“¶&÷&FW"Ö6öÆ÷#§f"‚ÒÖ&ÇVR“¶&6¶w&÷VæC§f"‚Ò×æVÃ"—Òç7FF–öâ×&æ¶–ær×æVÅ¶†–FFVå×¶F—7Æ“¦æöæR–×÷'FçGÒææ6òÖ–ævW7BÖ6&G¶F—7Æ“¦fÆWƒ¶fÆW‚ÖF—&V7F–öã¦6öÇVÖçÒææ6òÖ–ævW7BÖ6&Bææ6òÖ†VFÖ×67&öÆÆW'¶fÆWƒ£¶F—7Æ“¦fÆWƒ¶Ö–âÖ†V–v‡C£Òææ6òÖ–ævW7BÖ6&Bææ6òÖ†VFÖ¶F—7Æ“¦fÆWƒ¶fÆWƒ£¶fÆW‚ÖF—&V7F–öã¦6öÇVÖã·v–GFƒ£WÒææ6òÖ–ævW7BÖ6&Bææ6òÖ†VFÖÖ&öG—¶fÆWƒ£¶Ö–âÖ†V–v‡C£#‡Òææ6òÖ–ævW7BÖ6&Bææ6òÖ†VFÖÖw&–G¶†V–v‡C£S¶w&–B×FV×ÆFR×&÷w3§&WVBƒrÆÖ–æÖ‚ƒ‚Ãg"’—Òææ6òÖ–ævW7BÖ6&Bææ6òÖ6VÆÇ¶†V–v‡C¦WFó¶Ö–âÖ†V–v‡C£‡Òææ6ò×vVV¶F’ÖÆ&VÇ7·v–GFƒ£#'ƒ¶föçB×6—¦S£‡‡Òææ6ò×vVV¶F’ÖÆ&VÇ27ç¶föçB×6—¦S£‡ƒ·v†—FR×76S¦æ÷w&Òææ6ò×vVV¶F’ÖÆ&VÇ27ã£¦gFW'¶6öçFVçC¦æöæR–×÷'FçGÒææ6òÖÖöçF‡7¶Ö&v–âÖÆVgC£#g‡Òææ6òÖÖöçF‡27ç·f—6–&–Æ—G“§f—6–&ÆR–×÷'FçGÔÖVF–†Ö‚×v–GFƒ£c‚—²ç7FF–öâ×v–æF÷rÖ6öçG&öÇ7·FF–ærÖ–æÆ–æS£W‡Òç7FF–öâ×v–æF÷rÖ'WGFöç·FF–æs£W‚gƒ¶föçB×6—¦S£‡Òææ6òÖ–ævW7BÖ6&Bææ6òÖ†VFÖÖ&öG—¶Ö–âÖ†V–v‡C£“‡Òææ6ò×vVV¶F’ÖÆ&VÇ7·v–GFƒ£‡ƒ¶föçB×6—¦S£wƒ·f—6–&–Æ—G“§f—6–&ÆR–×÷'FçGÒææ6ò×vVV¶F’ÖÆ&VÇ27ç¶föçB×6—¦S£w‡Òææ6òÖÖöçF‡7¶Ö&v–âÖÆVgC£#'ƒ¶föçB×6—¦S£w‡Òææ6òÖÖöçF‡27ç¶Ö‚×v–GFƒ£#‡ƒ·f—6–&–Æ—G“§f—6–&ÆR–×÷'FçG×ÓÂ÷7G–ÆSãÂö†VCârÀ¢À¢¢vRÒvRç&WÆ6R€¢sÆ'F–6ÆR6Æ73Ò&6&B6†'BÖ6&B#ãÆF—b6Æ73Ò&æ6òÖ–ævW7BÖ†VB#ârÀ¢sÆ'F–6ÆR6Æ73Ò&6&B6†'BÖ6&Bæ6òÖ–ævW7BÖ6&B#ãÆF—b6Æ73Ò&æ6òÖ–ævW7BÖ†VB#ârÀ¢À¢¢–æFW…÷F‚Ò÷WGWEöF—"ò&–æFW‚æ‡FÖÂ ¢–æFW…÷F‚çw&—FU÷FW‡B‡vRÂVæ6öF–æsÒ'WFbÓ‚"Ğ¢6ö6–Åö–ÖvRÒ$Uõõ$ôõBò&÷WGWG2"ò'WW%ö—%öæWGv÷&µöÖöæ—F÷""ò'6ö6–Â"ò&÷&–v–æÅöF6†&ö&E÷7G–ÆRçær ¢–b6ö6–Åö–ÖvRæW†—7G2‚“ ¢6‡WF–Âæ6÷“"‡6ö6–Åö–ÖvRÂ÷WGWEöF—"ò&örçær"¢VÇ6S ¢÷w&—FUöfÆÆ&6µöör†÷WGWEöF—"Â6æ6†÷B¢&WGW&â–æFW…÷F€ Ğ Ğ¦FVbÖ–â‚’ÓâæöæS Ğ¢'6W"Ò&w'6Rä&wVÖVçE'6W"†FW67&—F–öãÕõöFö5õòĞ¢'6W"æFEö&wVÖVçB‚"ÒÖ÷WGWBÖF—""ÂG—SÕF‚ÂFVfVÇCÔDTdTÅEôõUEUBĞ¢&w2Ò'6W"ç'6Uö&w2‚Ğ¢F‚Ò'V–ÆE÷V&Æ–5÷6—FR†&w2æ÷WGWEöF—"ç&W6öÇfR‚’Ğ¢&–çB†b$'V–ÇBV&Æ–2F6†&ö&C¢·F‡Ò"Ğ Ğ Ğ¦–bõöæÖUõòÓÒ%õöÖ–åõò# Ğ¢Ö–â‚Ğ