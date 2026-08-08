Exit code: 0
Wall time: 0.2 seconds
Total output lines: 1267
Output:
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
import plotly.graph_objects as go
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
        return "—", ""
    if not math.isfinite(gap):
        return "—", ""
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


SHARE_BG = "#061521"
SHARE_PANEL = "#0D2538"
SHARE_TEXT = "#F8FBFF"
SHARE_MUTED = "#AFC1D4"
SHARE_LINE = "#294960"
SHARE_BLUE = "#59C8F5"
SHARE_ORANGE = "#FF704F"
SHARE_GREEN = "#52D3A2"


def _share_box(fig: plt.Figure, x: float, y: float, width: float, height: float, *, edge: str = SHARE_LINE) -> None:
    fig.add_artist(plt.Rectangle((x, y), width, height, transform=fig.transFigure,
                                 facecolor=SHARE_PANEL, edgecolor=edge, linewidth=1.2,
                                 joinstyle="round", zorder=0))


def _share_date(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    return date.strftime("%b %d, %Y").replace(" 0", " ") if pd.notna(date) else "Latest data"


def _write_share_images(
    output_dir: Path,
    snapshot,
    archive_windows: pd.DataFrame,
    station_deficits: pd.DataFrame,
    map_uri: str,
    *,
    y_cap_low: float,
    y_cap_high: float,
    default_start: pd.Timestamp,
    last_trend_date: pd.Timestamp,
    issue_count: int,
    active_count: int,
    clean_count: int,
    new_count: int,
    persistent_count: int,
    resolved_count: int,
    nco_latest_text: str,
    nco_latest_detail: str,
) -> dict[str, Path]:
    """Render readable 1080px share images without relying on Plotly export.

    These are intentionally static, high-contrast compositions: a social image
    needs predictable typography and sizing, while the live dashboard keeps the
    interactive Plotly charts and hover details.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in ("share_archive.png", "share_operations.png", "share_station_rankings.png")}
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Archive card: one dominant number, one uncluttered trend, and a compact
    # window comparison. Fill is segmented by sign so surplus is unambiguously green.
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=SHARE_BG)
    fig.text(.07, .94, "SOUNDING AVAILABILITY", color=SHARE_BLUE, fontsize=14, fontweight="bold", va="top")
    fig.text(.07, .895, "Current 7-day archive gap", color=SHARE_TEXT, fontsize=27, fontweight="bold", va="top")
    gap = float(getattr(snapshot.payload, "gap_percent", float("nan")))
    gap_color = SHARE_GREEN if pd.notna(gap) and gap > 0 else SHARE_ORANGE
    gap_text = f"{gap:+.1f}%" if pd.notna(gap) else "—"
    fig.text(.07, .79, gap_text, color=gap_color, fontsize=68, fontweight="bold", va="top")
    fig.text(.07, .70, "Observed archive records versus the same-date baseline", color=SHARE_MUTED, fontsize=13, va="top")
    fig.text(.07, .66, f"Complete through {_share_date(getattr(snapshot.payload, 'latest_date', None))}", color=SHARE_MUTED, fontsize=12, va="top")

    _share_box(fig, .06, .345, .88, .27)
    fig.text(.09, .585, "ONE-YEAR TREND · BASELINE-CAPPED", color=SHARE_TEXT, fontsize=12, fontweight="bold", va="top")
    ax = fig.add_axes([.105, .39, .80, .16], facecolor=SHARE_PANEL)
    series = snapshot.payload.series.copy()
    if not series.empty and {"date", "observed", "baseline"}.issubset(series.columns):
        series["date"] = pd.to_datetime(series["date"], errors="coerce")
        for col in ("observed", "baseline"):
            series[col] = pd.to_numeric(series[col], errors="coerce")
        series = series.dropna(subset=["date", "observed", "baseline"]).sort_values("date")
        series = series[series["date"].between(default_start, last_trend_date)]
    if series.empty:
        ax.text(.5, .5, "Trend data unavailable", color=SHARE_MUTED, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        dates = series["date"].dt.to_pydatetime()
        observed = series["observed"].to_numpy()
        baseline = series["baseline"].to_numpy()
        ax.plot(dates, baseline, color=SHARE_MUTED, linewidth=2.0, linestyle="--", label="Baseline")
        ax.plot(dates, observed, color=SHARE_BLUE, linewidth=2.6, label="Observed")
        ax.fill_between(dates, observed, baseline, where=observed < baseline, color=SHARE_ORANGE, alpha=.28, interpolate=True)
        ax.fill_between(dates, observed, baseline, where=observed >= baseline, color=SHARE_GREEN, alpha=.25, interpolate=True)
        ax.set_ylim(y_cap_low, y_cap_high)
        ax.grid(axis="y", color=SHARE_LINE, alpha=.7, linewidth=.7)
        ax.tick_params(colors=SHARE_MUTED, labelsize=9)
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=SHARE_TEXT)
        for spine in ax.spines.values(): spine.set_color(SHARE_LINE)
        ax.set_ylabel("records/day", color=SHARE_MUTED, fontsize=9)
    fig.text(.07, .285, "Recent archive windows", color=SHARE_TEXT, fontsize=13, fontweight="bold", va="top")
    bar_ax = fig.add_axes([.10, .105, .80, .14], facecolor=SHARE_BG)
    if archive_windows.empty:
        bar_ax.text(.5, .5, "Window data unavailable", color=SHARE_MUTED, ha="center", va="center", transform=bar_ax.transAxes)
        bar_ax.set_axis_off()
    else:
        bars = archive_windows.dropna(subset=["days", "percent"]).sort_values("days", ascending=True)
        values = pd.to_numeric(bars["percent"], errors="coerce").to_numpy()
        labels = [f"{int(v)}D" for v in bars["days"]]
        colors = [SHARE_GREEN if v >= 0 else SHARE_ORANGE for v in values]
        bar_ax.bar(labels, values, color=colors, width=.65)
        bar_ax.axhline(0, color=SHARE_MUTED, linewidth=1)
        bar_ax.set_ylim(min(-10, float(values.min()) - 2), max(10, float(values.max()) + 2))
        bar_ax.tick_params(colors=SHARE_MUTED, labelsize=10)
        bar_ax.grid(axis="y", color=SHARE_LINE, alpha=.55)
        for label, value in zip(bar_ax.containers[0], values):
            bar_ax.text(label.get_x() + label.get_width()/2, value + (0.45 if value >= 0 else -0.8), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", color=SHARE_TEXT, fontsize=9, fontweight="bold")
        for spine in bar_ax.spines.values(): spine.set_visible(False)
    fig.text(.07, .045, "NOAA/NCEI IGRA v2 · data-availability diagnostic", color=SHARE_MUTED, fontsize=9)
    fig.text(.93, .045, "wall.cloud", color=SHARE_BLUE, fontsize=10, fontweight="bold", ha="right")
    fig.savefig(paths["share_archive.png"], dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Operations card: the map and the two key operational summaries are kept
    # large; no tiny embedded chart is used.
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=SHARE_BG)
    fig.text(.07, .94, "OPERATIONAL MESSAGES", color=SHARE_BLUE, fontsize=14, fontweight="bold", va="top")
    fig.text(.07, .895, "Current NCO-reported issues", color=SHARE_TEXT, fontsize=27, fontweight="bold", va="top")
    try:
        map_image = plt.imread(io.BytesIO(base64.b64decode(map_uri.split(",", 1)[1])))
        map_ax = fig.add_axes([.08, .48, .84, .34]); map_ax.imshow(map_image); map_ax.set_axis_off()
    except Exception:
        fig.text(.50, .64, "Station map unavailable", color=SHARE_MUTED, ha="center", va="center", fontsize=15)
    _share_box(fig, .07, .29, .40, .12, edge=SHARE_ORANGE)
    _share_box(fig, .53, .29, .40, .12, edge=SHARE_GREEN)
    fig.text(.095, .365, f"{issue_count} / {active_count}", color=SHARE_ORANGE, fontsize=30, fontweight="bold")
    fig.text(.095, .315, "stations with an issue", color=SHARE_MUTED, fontsize=12)
    fig.text(.555, .365, f"{clean_count}", color=SHARE_GREEN, fontsize=30, fontweight="bold")
    fig.text(.555, .315, "no issue reported", color=SHARE_MUTED, fontsize=12)
    _share_box(fig, .07, .13, .86, .11, edge=SHARE_BLUE)
    fig.text(.095, .205, "LATEST MAPPED STATUS", color=SHARE_TEXT, fontsize=11, fontweight="bold")
    fig.text(.095, .165, f"{new_count} new or changed · {persistent_count} persistent · {resolved_count} resolved", color=SHARE_MUTED, fontsize=12)
    fig.text(.07, .09, "NCO reported for ingest", color=SHARE_TEXT, fontsize=14, fontweight="bold")
    fig.text(.07, .06, nco_latest_text, color=SHARE_BLUE, fontsize=13, fontweight="bold")
    fig.text(.93, .06, nco_latest_detail, color=SHARE_MUTED, fontsize=9, ha="right")
    fig.text(.07, .025, "Operational-message reporting; not confirmed IGRA archive totals", color=SHARE_MUTED, fontsize=8.5)
    fig.text(.93, .025, "wall.cloud", color=SHARE_BLUE, fontsize=10, fontweight="bold", ha="right")
    fig.savefig(paths["share_operations.png"], dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Rankings card: two readable horizontal lists, deliberately limiting to
    # five rows so station names never collide or run off the image.
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=SHARE_BG)
    fig.text(.07, .94, "STATION RANKINGS", color=SHARE_BLUE, fontsize=14, fontweight="bold", va="top")
    fig.text(.07, .895, "Archive shortfall and surplus", color=SHARE_TEXT, fontsize=27, fontweight="bold", va="top")
    fig.text(.07, .85, "Largest station-level changes over the last 7 days", color=SHARE_MUTED, fontsize=12)
    def rank_axis(rect, title, color, surplus=False):
        ax = fig.add_axes(rect, facecolor=SHARE_PANEL)
        if station_deficits.empty or "display_label" not in station_deficits:
            ax.text(.5, .5, "Station ranking data unavailable", color=SHARE_MUTED, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); return
        frame = station_deficits.copy()
        observed_col, expected_col = "observed_7", "expected_7"
        if observed_col not in frame or expected_col not in frame:
            ax.text(.5, .5, "7-day station data unavailable", color=SHARE_MUTED, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); return
        frame[observed_col] = pd.to_numeric(frame[observed_col], errors="coerce")
        frame[expected_col] = pd.to_numeric(frame[expected_col], errors="coerce")
        frame["value"] = frame[observed_col] - frame[expected_col]
        frame = frame.dropna(subset=["display_label", "value"])
        frame = frame[frame["value"] > 0] if surplus else frame[frame["value"] < 0]
        frame["value"] = frame["value"].abs()
        frame = frame.nlargest(5, "value").sort_values("value")
        if frame.empty:
            ax.text(.5, .5, "No stations in this category", color=SHARE_MUTED, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); return
        labels = frame["display_label"].astype(str).str.slice(0, 26)
        ax.barh(labels, frame["value"], color=color, height=.58)
        ax.set_title(title, loc="left", color=SHARE_TEXT, fontsize=15, fontweight="bold", pad=14)
        ax.tick_params(colors=SHARE_MUTED, labelsize=10)
        ax.grid(axis="x", color=SHARE_LINE, alpha=.6)
        ax.set_xlabel("soundings", color=SHARE_MUTED, fontsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
    rank_axis([.09, .49, .82, .26], "Shortfall", SHARE_ORANGE, surplus=False)
    rank_axis([.09, .16, .82, .26], "Surplus", SHARE_GREEN, surplus=True)
    fig.text(.07, .06, "IGRA archive records vs the 2021–2024 same-date baseline", color=SHARE_MUTED, fontsize=9)
    fig.text(.93, .06, "wall.cloud", color=SHARE_BLUE, fontsize=10, fontweight="bold", ha="right")
    fig.savefig(paths["share_station_rankings.png"], dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


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
    dates = pd.to_datetime(frame[column], errors="coerce").dropna(…21595 tokens truncated…       '#share-archive-card #share-archive-windows{height:160px!important}'
        '#share-operations-card .share-map{height:300px!important;max-height:300px!important}'
        '#share-operations-card #share-nco-summary{height:150px!important}'
        '@media(max-width:600px){'
        '#share-archive-card{padding:8px!important}'
        '#share-archive-card .share-card-head{gap:4px}'
        '#share-archive-card .share-card-kicker{font-size:8px}'
        '#share-archive-card h3{font-size:20px}'
        '#share-archive-card p{font-size:8px}'
        '#share-archive-card .share-action{font-size:9px;padding:4px 6px}'
        '#share-archive-card .share-gap-value{font-size:48px;margin:5px 0 1px}'
        '#share-archive-card .share-gap-detail{font-size:8px}'
        '#share-archive-card .share-chart-title{font-size:10px;margin:3px 0 1px}'
        '#share-archive-card #share-archive-trend{height:80px!important}'
        '#share-archive-card #share-archive-windows{height:55px!important}'
        '#share-archive-card .share-card-source{font-size:7px}'
        '#share-operations-card{padding:8px!important}'
        '#share-operations-card .share-card-head{gap:4px}'
        '#share-operations-card .share-card-kicker{font-size:8px}'
        '#share-operations-card h3{font-size:18px}'
        '#share-operations-card p{font-size:8px}'
        '#share-operations-card .share-action{font-size:9px;padding:4px 6px}'
        '#share-operations-card .share-map{height:70px!important;max-height:70px!important;margin-top:3px}'
        '#share-operations-card .share-status-grid{gap:5px;margin-top:3px}'
        '#share-operations-card .share-status-box{padding:3px 4px;border-radius:6px}'
        '#share-operations-card .share-status-grid>div{padding-top:2px}'
        '#share-operations-card .share-status-grid strong{font-size:18px}'
        '#share-operations-card .share-status-grid span{font-size:7px}'
        '#share-operations-card .share-status-banner{margin-top:3px;padding:3px 4px}'
        '#share-operations-card .share-status-banner .share-chart-title{font-size:9px;margin:0 0 1px}'
        '#share-operations-card .share-status-note{font-size:8px}'
        '#share-operations-card .share-nco-latest{font-size:10px;margin-top:1px}'
        '#share-operations-card .share-nco-detail{font-size:7px}'
        '#share-operations-card #share-nco-summary{height:55px!important}'
        '#share-operations-card .share-card-source{font-size:7px}'
        '#share-stations-card{padding:8px!important}'
        '#share-stations-card .share-card-head{gap:4px}'
        '#share-stations-card .share-card-kicker{font-size:8px}'
        '#share-stations-card h3{font-size:18px}'
        '#share-stations-card p{font-size:8px}'
        '#share-stations-card .share-action{font-size:9px;padding:4px 6px}'
        '#share-stations-card .share-chart-title{font-size:10px;margin:3px 0 1px}'
        '#share-stations-card #share-station-shortfalls-7,#share-stations-card #share-station-surpluses-7{height:105px!important}'
        '#share-stations-card .share-card-source{font-size:7px}'
        '}'
        '</style></head>',
        1,
    )
    page = page.replace(
        '</head>',
        '<style>.share-grid{display:grid;grid-template-columns:1fr;gap:24px}.share-card{width:min(100%,1080px);aspect-ratio:1/1;margin-inline:auto;padding:28px;border-radius:24px;display:flex;flex-direction:column;justify-content:flex-start}.share-card-wide{grid-column:auto}.share-card-head{gap:24px}.share-card-kicker{font-size:18px}.share-card h3{font-size:clamp(30px,4vw,54px);margin:6px 0 5px}.share-card p{font-size:18px;line-height:1.25}.share-action{padding:11px 16px;font-size:16px}.share-gap-value{font-size:clamp(72px,8vw,124px);margin:18px 0 5px}.share-gap-detail,.share-nco-detail,.share-status-note{font-size:18px}.share-chart-title{margin:16px 0 4px;font-size:22px}.share-card-source{margin-top:8px;font-size:14px}.share-map{margin-top:12px;max-height:410px;flex:0 0 auto}.share-status-grid{gap:18px;margin-top:14px}.share-status-box{border:1px solid var(--line);border-radius:14px;padding:13px 14px;background:var(--panel2)}.share-status-issue{border-color:rgba(255,112,79,.7);background:rgba(255,112,79,.1)}.share-status-clean{border-color:rgba(82,211,162,.65);background:rgba(82,211,162,.1)}.share-status-grid>div{padding-top:10px}.share-status-grid strong{font-size:48px;line-height:1}.share-status-grid span{font-size:17px}.share-status-banner{margin-top:12px;padding:9px 12px;border-left:4px solid var(--orange);background:rgba(255,112,79,.08);border-radius:8px}.share-nco-banner{border-left-color:var(--blue);background:rgba(89,200,245,.08)}.share-status-banner .share-chart-title{margin:0 0 3px}.share-nco-latest{font-size:26px;margin-top:6px}.share-ranking-grid{gap:22px}.share-card .js-plotly-plot{width:100%!important}.share-card .plot-container,.share-card .svg-container{max-width:100%!important}@media(max-width:900px){.share-card{padding:20px}.share-card-kicker{font-size:14px}.share-card h3{font-size:clamp(26px,5vw,44px)}.share-card p{font-size:15px}.share-gap-value{font-size:clamp(62px,10vw,96px)}.share-gap-detail,.share-nco-detail,.share-status-note{font-size:15px}.share-chart-title{font-size:18px}.share-status-grid strong{font-size:38px}.share-status-grid span{font-size:14px}.share-nco-latest{font-size:22px}}@media(max-width:600px){.navlinks{display:flex;gap:8px;font-size:12px}.navlinks a:not(.nav-share){display:none}.nav-share{display:block!important}.share-card{width:100%;padding:12px;border-radius:16px}.share-card-head{gap:8px}.share-card-kicker{font-size:9px}.share-card h3{font-size:clamp(20px,6vw,30px);margin:2px 0}.share-card p{font-size:10px;line-height:1.15}.share-action{padding:5px 7px;font-size:10px}.share-gap-value{font-size:clamp(44px,14vw,70px);margin:8px 0 2px}.share-gap-detail,.share-nco-detail,.share-status-note{font-size:10px}.share-chart-title{margin:5px 0 1px;font-size:12px}.share-card-source{font-size:8px}.share-map{margin-top:5px;max-height:none;height:clamp(105px,32vw,190px)!important}.share-status-grid{gap:7px;margin-top:5px}.share-status-box{padding:5px 6px;border-radius:8px}.share-status-grid>div{padding-top:4px}.share-status-grid strong{font-size:24px}.share-status-grid span{font-size:9px}.share-status-banner{margin-top:5px;padding:4px 6px;border-left-width:2px}.share-nco-latest{font-size:14px;margin-top:2px}.share-ranking-grid{grid-template-columns:1fr;gap:3px}.share-card .js-plotly-plot{min-height:0}.share-card #share-archive-trend{height:clamp(105px,34vw,210px)!important}.share-card #share-archive-windows{height:clamp(76px,24vw,150px)!important}.share-card #share-nco-summary{height:clamp(72px,22vw,120px)!important}.share-card #share-station-shortfalls-7,.share-card #share-station-surpluses-7{height:clamp(86px,28vw,150px)!important}}</style></head>',
        1,
    )
    page = page.replace(
        '<article class="card chart-card"><div class="nco-ingest-head">',
        '<article class="card chart-card nco-ingest-card"><div class="nco-ingest-head">',
        1,
    )
    page = page.replace(
        '</head>',
        '<style>.share-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.share-card{min-width:0;background:linear-gradient(150deg,rgba(89,200,245,.08),var(--panel) 55%);border:1px solid var(--line);border-radius:20px;padding:18px;overflow:hidden;break-inside:avoid}.share-card-wide{grid-column:1/-1}.share-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px}.share-card-kicker{color:var(--blue);font-size:10px;font-weight:850;letter-spacing:.12em}.share-card h3{margin:4px 0 2px;font-size:clamp(21px,2.2vw,30px);line-height:1.1;letter-spacing:-.025em}.share-card p{margin:0;color:var(--muted);font-size:12px}.share-action{flex:0 0 auto;border:1px solid var(--line);background:var(--bg);color:var(--blue);border-radius:8px;padding:7px 10px;cursor:pointer;font-size:11px;font-weight:800}.share-action:hover,.share-action:focus-visible{border-color:var(--blue);background:var(--panel2)}.share-gap-value{font-size:clamp(46px,6vw,76px);font-weight:850;line-height:1;margin:16px 0 3px;letter-spacing:-.06em}.share-gap-detail,.share-nco-detail,.share-status-note{color:var(--muted);font-size:12px}.share-chart-title{margin:14px 0 2px;color:var(--text);font-size:12px;font-weight:800;letter-spacing:.02em}.share-chart-title-tight{margin-top:4px}.share-card-source{margin-top:8px;color:var(--muted);font-size:10px}.share-map{margin-top:14px;max-height:340px;object-fit:contain;background:var(--panel2)}.share-status-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}.share-status-grid>div{border-top:1px solid var(--line);padding-top:8px}.share-status-grid strong{display:block;font-size:22px}.share-status-grid span{color:var(--muted);font-size:11px}.share-nco-latest{font-size:16px;font-weight:800;margin-top:5px}.share-ranking-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.share-ranking-grid>div{min-width:0}.share-card .js-plotly-plot{width:100%!important}.share-card .plot-container,.share-card .svg-container{max-width:100%!important}@media(max-width:900px){.share-grid{grid-template-columns:1fr}.share-card-wide{grid-column:auto}}@media(max-width:600px){.navlinks{display:flex;gap:8px;font-size:12px}.navlinks a:not(.nav-share){display:none}.nav-share{display:block!important}.share-card{padding:14px;border-radius:16px}.share-card-head{gap:8px}.share-action{padding:6px 8px}.share-ranking-grid{grid-template-columns:1fr}.share-card .js-plotly-plot{min-height:0}}</style></head>',
        1,
    )
    page = page.replace(
        '</head>',
        '<style>.share-grid{grid-template-columns:1fr!important;gap:24px!important}.share-card{width:min(100%,1080px)!important;aspect-ratio:1/1!important;margin-inline:auto!important;padding:28px!important;border-radius:24px!important;display:flex!important;flex-direction:column!important}.share-card-wide{grid-column:auto!important}.share-card-head{gap:24px}.share-card-kicker{font-size:18px}.share-card h3{font-size:clamp(30px,4vw,54px);margin:6px 0 5px}.share-card p{font-size:18px;line-height:1.25}.share-action{padding:11px 16px;font-size:16px}.share-gap-value{font-size:clamp(72px,8vw,124px);margin:18px 0 5px}.share-gap-detail,.share-nco-detail,.share-status-note{font-size:18px}.share-chart-title{margin:16px 0 4px;font-size:22px}.share-card-source{margin-top:8px;font-size:14px}.share-map{margin-top:12px;max-height:410px}.share-status-grid{gap:18px;margin-top:14px}.share-status-box{border:1px solid var(--line);border-radius:14px;padding:13px 14px;background:var(--panel2)}.share-status-issue{border-color:rgba(255,112,79,.7);background:rgba(255,112,79,.1)}.share-status-clean{border-color:rgba(82,211,162,.65);background:rgba(82,211,162,.1)}.share-status-grid>div{padding-top:10px}.share-status-grid strong{font-size:48px;line-height:1}.share-status-grid span{font-size:17px}.share-status-banner{margin-top:12px;padding:9px 12px;border-left:4px solid var(--orange);background:rgba(255,112,79,.08);border-radius:8px}.share-nco-banner{border-left-color:var(--blue);background:rgba(89,200,245,.08)}.share-status-banner .share-chart-title{margin:0 0 3px}.share-nco-latest{font-size:26px;margin-top:6px}.share-ranking-grid{gap:22px}.share-card #share-archive-trend,.share-card #share-archive-windows,.share-card #share-nco-summary,.share-card #share-station-shortfalls-7,.share-card #share-station-surpluses-7{font-size:inherit}@media(max-width:900px){.share-card{padding:20px!important}.share-card-kicker{font-size:14px}.share-card h3{font-size:clamp(26px,5vw,44px)}.share-card p{font-size:15px}.share-gap-value{font-size:clamp(62px,10vw,96px)}.share-gap-detail,.share-nco-detail,.share-status-note{font-size:15px}.share-chart-title{font-size:18px}.share-status-grid strong{font-size:38px}.share-status-grid span{font-size:14px}.share-nco-latest{font-size:22px}}@media(max-width:600px){.navlinks{display:flex;gap:8px;font-size:12px}.navlinks a:not(.nav-share){display:none}.nav-share{display:block!important}.share-card{width:100%!important;padding:12px!important;border-radius:16px!important}.share-card-head{gap:8px}.share-card-kicker{font-size:9px}.share-card h3{font-size:clamp(20px,6vw,30px);margin:2px 0}.share-card p{font-size:10px;line-height:1.15}.share-action{padding:5px 7px;font-size:10px}.share-gap-value{font-size:clamp(44px,14vw,70px);margin:8px 0 2px}.share-gap-detail,.share-nco-detail,.share-status-note{font-size:10px}.share-chart-title{margin:5px 0 1px;font-size:12px}.share-card-source{font-size:8px}.share-map{margin-top:5px;max-height:none;height:clamp(105px,32vw,190px)!important}.share-status-grid{gap:7px;margin-top:5px}.share-status-box{padding:5px 6px;border-radius:8px}.share-status-grid>div{padding-top:4px}.share-status-grid strong{font-size:24px}.share-status-grid span{font-size:9px}.share-status-banner{margin-top:5px;padding:4px 6px;border-left-width:2px}.share-nco-latest{font-size:14px;margin-top:2px}.share-ranking-grid{grid-template-columns:1fr;gap:3px}.share-card #share-archive-trend{height:clamp(105px,34vw,210px)!important}.share-card #share-archive-windows{height:clamp(76px,24vw,150px)!important}.share-card #share-nco-summary{height:clamp(72px,22vw,120px)!important}.share-card #share-station-shortfalls-7,.share-card #share-station-surpluses-7{height:clamp(86px,28vw,150px)!important}}</style></head>',
        1,
    )
    page = page.replace(
        '</head>',
        '<style>@media(max-width:600px){nav .wrap{flex-wrap:wrap;gap:8px;padding:8px 0;align-items:center}.nav-status{order:3;width:100%;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;border-radius:12px;padding:5px 8px;font-size:10px;line-height:1.2;overflow-wrap:anywhere}.nav-status span{min-width:0;display:block}.navlinks{margin-left:auto}}</style></head>',
        1,
    )
    # Keep the primary actions beside the features they share, while the
    # dedicated Share section remains a simple image gallery.
    page = page.replace(
        '<div class="chart-title">Sounding availability trend</div>',
        '<div class="chart-title feature-title-row"><span>Sounding availability trend</span><button type="button" class="share-image-action" data-share-image="share_archive.png" aria-label="Share sounding availability image">Share image</button></div>',
        1,
    )
    page = page.replace(
        '<div class="section-head"><div class="eyebrow">STATION STATUS</div><h2>Current NCO-reported issues</h2></div>',
        '<div class="section-head"><div class="eyebrow">STATION STATUS</div><h2>Current NCO-reported issues</h2><button type="button" class="share-image-action" data-share-image="share_operations.png" aria-label="Share station operations image">Share image</button></div>',
        1,
    )
    page = page.replace(
        '<div class="chart-title">Stations ranked by archive shortfall</div>',
        '<div class="chart-title feature-title-row"><span>Stations ranked by archive shortfall</span><button type="button" class="share-image-action" data-share-image="share_station_rankings.png" aria-label="Share station rankings image">Share image</button></div>',
        1,
    )
    share_start = page.find('<section id="share"')
    if share_start >= 0:
        share_end = page.find('</section>', share_start)
        if share_end >= 0:
            share_end += len('</section>')
            page = page[:share_start] + (
                '<section id="share" class="section share-section"><div class="section-head"><div>'
                '<div class="eyebrow">SHARE IMAGES</div><h2>Downloadable data snapshots</h2></div>'
                '<p>Each button creates a readable 1080 × 1080 PNG for posting or saving.</p></div>'
                '<div class="share-grid share-image-grid">'
                '<article id="share-archive-card" class="share-card"><div class="share-card-head"><div><div class="share-card-kicker">SOUNDING AVAILABILITY</div><h3>Current 7-day archive gap</h3></div><button type="button" class="share-image-action" data-share-image="share_archive.png">Share image</button></div><img class="share-image-preview" src="share_archive.png" alt="Square sounding availability archive gap image"></article>'
                '<article id="share-operations-card" class="share-card"><div class="share-card-head"><div><div class="share-card-kicker">OPERATIONAL MESSAGES</div><h3>Current NCO-reported issues</h3></div><button type="button" class="share-image-action" data-share-image="share_operations.png">Share image</button></div><img class="share-image-preview" src="share_operations.png" alt="Square station status and NCO issue image"></article>'
                '<article id="share-stations-card" class="share-card"><div class="share-card-head"><div><div class="share-card-kicker">STATION RANKINGS</div><h3>Archive shortfall and surplus</h3></div><button type="button" class="share-image-action" data-share-image="share_station_rankings.png">Share image</button></div><img class="share-image-preview" src="share_station_rankings.png" alt="Square station archive shortfall and surplus image"></article>'
                '</div></section>'
            ) + page[share_end:]
    page = page.replace(
        '</head>',
        '<style>.feature-title-row,.section-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.share-image-action{flex:0 0 auto;border:1px solid var(--blue);background:rgba(89,200,245,.08);color:var(--blue);border-radius:9px;padding:7px 11px;cursor:pointer;font-size:11px;font-weight:800;white-space:nowrap}.share-image-action:hover,.share-image-action:focus-visible{background:var(--panel2);box-shadow:0 0 0 2px rgba(89,200,245,.2)}.share-image-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.share-image-preview{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:var(--bg);border-radius:12px;margin-top:12px}@media(max-width:900px){.share-image-grid{grid-template-columns:1fr 1fr}}@media(max-width:600px){.feature-title-row{align-items:flex-start}.section-head{align-items:flex-start;flex-wrap:wrap}.share-image-grid{grid-template-columns:1fr}.share-image-action{font-size:10px;padding:6px 8px}}</style></head>',
        1,
    )
    page = page.replace(
        '</body></html>',
        '<script>document.querySelectorAll(".share-image-action").forEach(button=>button.addEventListener("click",async()=>{const path=button.dataset.shareImage;const prior=button.textContent;try{const response=await fetch(path,{cache:"no-store"});if(!response.ok)throw new Error("image unavailable");const blob=await response.blob();const name=path.split("/").pop()||"soundings-share.png";const file=new File([blob],name,{type:"image/png"});if(navigator.canShare&&navigator.canShare({files:[file]})){await navigator.share({title:"CONUS Upper-Air Data Watch",files:[file]});}else{const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}button.textContent="Image ready";setTimeout(()=>{button.textContent=prior;},1800);}catch(_error){button.textContent="Try again";setTimeout(()=>{button.textContent=prior;},1800);}}));</script></body></html>',
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

