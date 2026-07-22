"""Create the CONUS upper-air network dashboard and social graphics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


STATION_MASTER_PATH = Path("data") / "upper_air_station_master.csv"
IGRA_DAILY_PATH = Path("outputs") / "conus_balloon_launches_by_year_daily.csv"
NCO_AVAILABILITY_PATH = Path("data") / "nco_raob_availability.csv"
NCO_ISSUES_PATH = Path("data") / "nco_raob_station_issues.csv"
SPC_AVAILABILITY_PATH = Path("data") / "spc_sounding_availability.csv"
STATION_DEFICITS_PATH = Path("outputs") / "conus_balloon_launches_station_deficits.csv"
DASHBOARD_PATH = Path("outputs") / "upper_air_network_dashboard.png"
SOCIAL_DIR = Path("outputs") / "social_upper_air"
CONUS_STATE_GEOJSON_PATH = Path("comfortwx") / "mapping" / "data" / "us_states.geojson"

SOURCE_IGRA = "Source: NOAA/NCEI IGRA v2."
SOURCE_NCO = "Source: NWS/NCEP/NCO SDM Administrative Messages."
SOURCE_SPC = "Source: SPC observed sounding page."
CAVEAT = (
    "Counts reflect available archive/operational records and may differ from actual launches "
    "due to ingest, archive, or reporting delays."
)

BG = "#f7f5ef"
INK = "#1f2933"
MUTED = "#667085"
GRID = "#d0d5dd"
BLUE = "#2563eb"
RED = "#c2410c"
ORANGE = "#f59e0b"
GREEN = "#16a34a"
GRAY = "#98a2b3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create upper-air monitor graphics.")
    parser.add_argument("--outdir", default="outputs")
    return parser.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_data(outdir: Path) -> dict[str, pd.DataFrame]:
    data = {
        "stations": read_csv(STATION_MASTER_PATH, dtype=str),
        "igra": read_csv(outdir / IGRA_DAILY_PATH.name),
        "nco": read_csv(NCO_AVAILABILITY_PATH, dtype=str),
        "issues": read_csv(NCO_ISSUES_PATH, dtype=str),
        "spc": read_csv(SPC_AVAILABILITY_PATH, dtype=str),
        "station_deficits": read_csv(outdir / STATION_DEFICITS_PATH.name),
    }
    if not data["stations"].empty:
        data["stations"]["latitude"] = pd.to_numeric(data["stations"]["latitude"], errors="coerce")
        data["stations"]["longitude"] = pd.to_numeric(data["stations"]["longitude"], errors="coerce")
    if not data["igra"].empty:
        data["igra"]["date"] = pd.to_datetime(data["igra"]["date"], errors="coerce")
        for column in ["launches", "launches_7d_avg", "baseline_5yr_avg", "difference_vs_baseline", "percent_vs_baseline"]:
            data["igra"][column] = pd.to_numeric(data["igra"].get(column), errors="coerce")
    if not data["nco"].empty:
        data["nco"]["conus_count"] = pd.to_numeric(data["nco"]["conus_count"], errors="coerce")
        data["nco"]["message_dt"] = pd.to_datetime(data["nco"]["message_time_utc"], errors="coerce", utc=True)
        data["nco"]["cycle_dt"] = pd.to_datetime(
            data["nco"]["cycle_date_utc"] + " " + data["nco"]["cycle_hour"].str.zfill(2) + ":00",
            errors="coerce",
            utc=True,
        )
    if not data["station_deficits"].empty:
        numeric_columns = [
            "latitude",
            "longitude",
            "latest_observed",
            "latest_expected",
            "latest_deficit",
            "latest_percent_difference",
            "avg_7d",
            "expected_7d",
            "avg_7d_deficit",
            "avg_7d_percent_difference",
            "observed_7",
            "expected_7",
            "deficit_7",
            "percent_7",
            "observed_30",
            "expected_30",
            "deficit_30",
            "percent_30",
            "observed_60",
            "expected_60",
            "deficit_60",
            "percent_60",
            "observed_90",
            "expected_90",
            "deficit_90",
            "percent_90",
            "observed_180",
            "expected_180",
            "deficit_180",
            "percent_180",
            "observed_365",
            "expected_365",
            "deficit_365",
            "percent_365",
            "ytd_observed",
            "ytd_expected",
            "ytd_deficit",
            "ytd_percent",
        ]
        for column in numeric_columns:
            if column in data["station_deficits"]:
                data["station_deficits"][column] = pd.to_numeric(data["station_deficits"][column], errors="coerce")
    return data


def setup_fig(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=BG)
    return fig


def source_note(ax, text: str) -> None:
    ax.text(0.0, -0.17, text, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")


def no_data(ax, title: str, source: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.text(0.5, 0.5, "Source unavailable or not parsed", ha="center", va="center", color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    source_note(ax, source)


def current_igra(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    current_year = int(df["year"].max())
    return df[df["year"] == current_year].copy()


def latest_igra_row(df: pd.DataFrame) -> pd.Series | None:
    current = current_igra(df)
    current = current.dropna(subset=["date", "launches_7d_avg"])
    if current.empty:
        return None
    return current.sort_values("date").iloc[-1]


def plot_igra_trend(ax, igra: pd.DataFrame, title: str, callout: bool = True) -> None:
    if igra.empty:
        no_data(ax, title, SOURCE_IGRA)
        return
    current = current_igra(igra).dropna(subset=["date", "launches_7d_avg"])
    baseline = current.dropna(subset=["baseline_5yr_avg"])
    if current.empty:
        no_data(ax, title, SOURCE_IGRA)
        return
    ax.plot(current["date"], current["launches_7d_avg"], color=BLUE, linewidth=3, label=f"{int(current['year'].max())} 7-day avg")
    if not baseline.empty:
        ax.plot(baseline["date"], baseline["baseline_5yr_avg"], color=INK, linestyle="--", linewidth=2.3, label="prior 5-year same-date baseline")
        below = baseline["launches_7d_avg"] < baseline["baseline_5yr_avg"]
        ax.fill_between(
            baseline["date"],
            baseline["launches_7d_avg"],
            baseline["baseline_5yr_avg"],
            where=below,
            color=RED,
            alpha=0.18,
            interpolate=True,
        )
    latest = latest_igra_row(igra)
    if callout and latest is not None and not math.isnan(float(latest.get("difference_vs_baseline", np.nan))):
        ax.scatter([latest["date"]], [latest["launches_7d_avg"]], s=50, color=RED, zorder=5)
        ax.annotate(
            f"{latest['difference_vs_baseline']:.1f} vs baseline\n{latest['percent_vs_baseline']:.1f}%",
            xy=(latest["date"], latest["launches_7d_avg"]),
            xytext=(-105, 35),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.4},
            fontsize=10,
            color=INK,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fffaf0", "edgecolor": RED, "alpha": 0.95},
        )
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel("Launches/soundings")
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    source_note(ax, SOURCE_IGRA)


def rolling_windows(igra: pd.DataFrame) -> pd.DataFrame:
    current, _excluded_partial, _reason = plotted_igra_current(igra)
    current = current.dropna(subset=["date", "launches", "baseline_5yr_avg"])
    rows = []
    if current.empty:
        return pd.DataFrame()
    latest = current["date"].max()
    for days in (30, 60, 90):
        start = latest - pd.Timedelta(days=days - 1)
        subset = current[(current["date"] >= start) & (current["date"] <= latest)]
        if subset.empty:
            continue
        observed = float(subset["launches"].sum())
        expected = float(subset["baseline_5yr_avg"].sum())
        deficit = observed - expected
        percent = (deficit / expected * 100.0) if expected else np.nan
        rows.append({"window": f"{days}d", "observed": observed, "expected": expected, "deficit": deficit, "percent": percent})
    return pd.DataFrame(rows)


def plot_windows(ax, igra: pd.DataFrame, title: str) -> None:
    windows = rolling_windows(igra)
    if windows.empty:
        no_data(ax, title, SOURCE_IGRA)
        return
    x = np.arange(len(windows))
    width = 0.36
    ax.bar(x - width / 2, windows["observed"], width, color=BLUE, label="observed")
    ax.bar(x + width / 2, windows["expected"], width, color="#111827", alpha=0.75, label="expected")
    for index, row in windows.iterrows():
        ax.text(index, max(row["observed"], row["expected"]) * 1.015, f"{row['deficit']:.0f}\n{row['percent']:.1f}%", ha="center", va="bottom", fontsize=9, color=RED)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_xticks(x, windows["window"])
    ax.set_ylabel("Total launches/soundings")
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize=9)
    source_note(ax, SOURCE_IGRA)


def plot_nco(ax, nco: pd.DataFrame, title: str) -> None:
    if nco.empty:
        no_data(ax, title, SOURCE_NCO)
        return
    recent = nco.dropna(subset=["cycle_dt", "conus_count"]).sort_values("cycle_dt").tail(40)
    if recent.empty:
        no_data(ax, title, SOURCE_NCO)
        return
    for model, group in recent.groupby("model"):
        ax.plot(group["cycle_dt"], group["conus_count"], marker="o", linewidth=2.2, label=model)
    lowest = recent.loc[recent["conus_count"].idxmin()]
    ax.scatter([lowest["cycle_dt"]], [lowest["conus_count"]], color=RED, s=45, zorder=4)
    ax.annotate(
        f"low: {int(lowest['conus_count'])}",
        xy=(lowest["cycle_dt"], lowest["conus_count"]),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel("CONUS RAOBs")
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    source_note(ax, SOURCE_NCO)


def latest_cycle(nco: pd.DataFrame) -> pd.Series | None:
    if nco.empty or "cycle_dt" not in nco:
        return None
    sort_columns = ["cycle_dt"]
    if "message_dt" in nco:
        sort_columns.append("message_dt")
    valid = nco.dropna(subset=["cycle_dt"]).sort_values(sort_columns)
    if valid.empty:
        return None
    return valid.iloc[-1]


def latest_issues_for_cycle(issues: pd.DataFrame, nco: pd.DataFrame) -> pd.DataFrame:
    latest = latest_cycle(nco)
    if latest is None or issues.empty:
        return pd.DataFrame()
    return issues[
        (issues["cycle_date_utc"] == latest["cycle_date_utc"])
        & (issues["cycle_hour"].astype(str).str.zfill(2) == str(latest["cycle_hour"]).zfill(2))
    ].copy()


def plot_issue_list(ax, issues: pd.DataFrame, nco: pd.DataFrame, title: str) -> None:
    latest_issues = latest_issues_for_cycle(issues, nco)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.axis("off")
    if latest_issues.empty:
        ax.text(0.02, 0.75, "No latest-cycle station issue rows parsed.", color=MUTED, fontsize=11)
        source_note(ax, SOURCE_NCO)
        return
    latest_issues["issue_category"] = latest_issues["issue_category"].fillna("other")
    y = 0.92
    for category, group in latest_issues.groupby("issue_category"):
        stations = ", ".join(group["station_id"].dropna().astype(str).head(9))
        extra = max(0, len(group) - 9)
        if extra:
            stations += f" +{extra} more"
        line = f"{category.replace('_', ' ').title()} ({len(group)}): {stations}"
        wrapped = textwrap.fill(line, width=46)
        ax.text(0.02, y, wrapped, fontsize=10.2, color=INK, va="top", linespacing=1.25)
        y -= 0.075 * (wrapped.count("\n") + 1) + 0.025
        if y < 0.12:
            remaining = latest_issues["issue_category"].nunique()
            ax.text(0.02, y, f"Additional categories omitted in this panel ({remaining} total).", fontsize=9.5, color=MUTED, va="top")
            break
    source_note(ax, SOURCE_NCO)


def station_statuses(stations: pd.DataFrame, issues: pd.DataFrame, nco: pd.DataFrame) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame()
    result = stations.copy()
    result["status"] = "unknown"
    result["status_color"] = GRAY
    latest = latest_cycle(nco)
    if latest is not None:
        result["status"] = "available/no issue"
        result["status_color"] = GREEN
    latest_issues = latest_issues_for_cycle(issues, nco)
    for _, issue in latest_issues.iterrows():
        station_id = str(issue.get("station_id", "")).upper()
        category = str(issue.get("issue_category", "other"))
        if category in {"no_report", "unavailable", "equipment_failure"}:
            status, color = "missing/problem", RED
        elif category in {"missing_parts", "short_sounding", "purged_data"}:
            status, color = "partial/quality issue", ORANGE
        else:
            status, color = "issue", ORANGE
        mask = result["station_id"].astype(str).str.upper() == station_id
        result.loc[mask, "status"] = status
        result.loc[mask, "status_color"] = color
    return result


def plot_map(ax, stations: pd.DataFrame, issues: pd.DataFrame, nco: pd.DataFrame, title: str) -> None:
    statuses = station_statuses(stations, issues, nco)
    if statuses.empty:
        no_data(ax, title, "Source: station master plus latest NCO issue layer.")
        return
    statuses = statuses.dropna(subset=["latitude", "longitude"])
    ax.scatter(statuses["longitude"], statuses["latitude"], c=statuses["status_color"], s=52, edgecolor="white", linewidth=0.7)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    labels = [("available/no issue", GREEN), ("missing/problem", RED), ("partial/quality issue", ORANGE), ("unknown", GRAY)]
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label) for label, color in labels]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left")
    latest = latest_cycle(nco)
    if latest is not None:
        timestamp = f"{latest['cycle_date_utc']} {str(latest['cycle_hour']).zfill(2)}Z"
        ax.text(0.99, 0.03, timestamp, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=MUTED)
    source_note(ax, "Source: station master from NCO/IGRA; latest status from NCO SDM Administrative Messages.")


def plot_timeline(ax, igra: pd.DataFrame, title: str) -> None:
    plot_igra_trend(ax, igra, title, callout=False)
    if igra.empty:
        return
    current = current_igra(igra)
    if current.empty:
        return
    date_min, date_max = current["date"].min(), current["date"].max()
    notices = [
        (pd.Timestamp("2025-03-20"), "NWS suspension notice:\nOmaha + Rapid City"),
        (pd.Timestamp("2025-04-17"), "NWS temporary reduction notice:\nselected sites"),
    ]
    for notice_date, label in notices:
        if date_min <= notice_date <= date_max:
            ax.axvline(notice_date, color=ORANGE, linewidth=1.6, linestyle=":")
            ax.text(notice_date, ax.get_ylim()[1] * 0.96, label, rotation=90, va="top", ha="right", fontsize=8.5, color=INK)


def save_social(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def add_social_title(fig, title: str) -> None:
    fig.text(0.07, 0.955, title, fontsize=24, fontweight="bold", color=INK, va="top")


def create_social_graphics(data: dict[str, pd.DataFrame], outdir: Path) -> None:
    igra, nco, issues, stations = data["igra"], data["nco"], data["issues"], data["stations"]

    fig = setup_fig(10.8, 13.5)
    add_social_title(fig, "CONUS Weather Balloon Observations Are Running Below Normal")
    ax = fig.add_axes([0.08, 0.17, 0.86, 0.68])
    plot_igra_trend(ax, igra, "", callout=True)
    fig.text(0.07, 0.055, CAVEAT, fontsize=10, color=MUTED)
    save_social(fig, outdir / "upper_air_decline_main.png")

    fig = setup_fig(10.8, 13.5)
    add_social_title(fig, "Fewer Upper-Air Observations Than Normal")
    ax = fig.add_axes([0.1, 0.18, 0.82, 0.66])
    plot_windows(ax, igra, "")
    fig.text(0.07, 0.055, CAVEAT, fontsize=10, color=MUTED)
    save_social(fig, outdir / "missing_launches_30_60_90.png")

    fig = setup_fig(10.8, 13.5)
    add_social_title(fig, "RAOBs Available for Model Ingest")
    ax = fig.add_axes([0.1, 0.18, 0.82, 0.66])
    plot_nco(ax, nco, "")
    fig.text(0.07, 0.055, CAVEAT, fontsize=10, color=MUTED)
    save_social(fig, outdir / "nco_model_ingest_status.png")

    fig = setup_fig(10.8, 13.5)
    add_social_title(fig, "Latest Problem/Missing Upper-Air Stations")
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.7])
    plot_issue_list(ax, issues, nco, "")
    fig.text(0.07, 0.055, CAVEAT, fontsize=10, color=MUTED)
    save_social(fig, outdir / "station_issues_latest.png")

    fig = setup_fig(10.8, 13.5)
    add_social_title(fig, "CONUS Upper-Air Network Availability")
    ax = fig.add_axes([0.09, 0.17, 0.84, 0.67])
    plot_map(ax, stations, issues, nco, "")
    fig.text(0.07, 0.055, CAVEAT, fontsize=10, color=MUTED)
    save_social(fig, outdir / "network_availability_map.png")

    fig = setup_fig(10.8, 13.5)
    add_social_title(fig, "Upper-Air Observation Changes in Context")
    ax = fig.add_axes([0.08, 0.17, 0.86, 0.68])
    plot_timeline(ax, igra, "")
    fig.text(0.07, 0.055, CAVEAT + " Operational notices are context markers, not causal claims.", fontsize=10, color=MUTED)
    save_social(fig, outdir / "event_timeline.png")

    create_x_post_main(data, outdir / "x_post_main.png")
    write_x_caption_options(data, outdir / "x_post_caption_options.txt")
    create_x_post_main_v2(data, outdir / "x_post_main_v2.png")
    write_x_caption_options_v2(data, outdir / "x_post_caption_options_v2.txt")
    create_x_post_main_v3(data, outdir / "x_post_main_v3.png")
    create_x_post_2025_divergence(data, outdir / "x_post_2025_divergence.png")
    create_decline_start_v2(data, outdir / "decline_start_v2.png")
    create_decline_start_v3(data, outdir / "decline_start_v3.png")
    create_cumulative_deficit_context(data, outdir / "cumulative_missing_soundings_context.png")
    create_timeline_observed_decline_context(data, outdir / "timeline_observed_decline_context.png")
    create_observed_expected_counter(data, outdir / "observed_expected_windows_context.png")
    create_station_deficit_map_context(data, outdir / "station_deficit_map_context.png")
    create_impacted_station_small_multiples(data, outdir / "impacted_station_small_multiples.png")
    create_nco_ingest_context(data, outdir / "nco_ingest_context.png")
    create_two_layer_evidence_context(data, outdir / "two_layer_evidence_context.png")
    create_operational_pipeline_explainer(data, outdir / "upper_air_operational_pipeline.png")
    create_social_hero_decline(data, outdir / "social_hero_decline.png")
    create_social_observed_expected_windows(data, outdir / "social_observed_expected_windows.png")
    create_social_cumulative_gap(data, outdir / "social_cumulative_gap.png")
    create_social_station_impacts(data, outdir / "social_station_impacts.png")
    create_social_two_layer_evidence(data, outdir / "social_two_layer_evidence.png")
    write_social_data_snapshot(data, outdir / "social_data_snapshot.json")
    write_social_editorial_manifest(data, outdir / "social_editorial_pack_manifest.txt")
    write_social_caption_pack(data, outdir / "social_caption_pack.md")
    carousel_dir = outdir / "carousel_doge_context"
    create_carousel_cover(data, carousel_dir / "carousel_01_hero.png")
    create_timeline_observed_decline_context(data, carousel_dir / "carousel_02_timeline.png")
    create_cumulative_deficit_context(data, carousel_dir / "carousel_03_cumulative_missing.png")
    create_station_deficit_map_context(data, carousel_dir / "carousel_04_station_map.png")
    create_two_layer_evidence_context(data, carousel_dir / "carousel_05_two_data_layers.png")
    write_decline_start_summary(data, outdir / "decline_start_summary.txt")


def latest_nco_row(nco: pd.DataFrame) -> pd.Series | None:
    if nco.empty or "cycle_dt" not in nco:
        return None
    sort_columns = ["cycle_dt"]
    if "message_dt" in nco:
        sort_columns.append("message_dt")
    valid = nco.dropna(subset=["cycle_dt"]).sort_values(sort_columns)
    if valid.empty:
        return None
    return valid.iloc[-1]


def create_x_post_main(data: dict[str, pd.DataFrame], path: Path) -> None:
    igra = data["igra"]
    nco = data["nco"]
    latest_igra = latest_igra_row(igra)
    latest_nco = latest_nco_row(nco)

    percent_text = "unavailable"
    date_text = "Latest IGRA date unavailable"
    avg_text = "7-day avg unavailable"
    if latest_igra is not None:
        if pd.notna(latest_igra.get("percent_vs_baseline")):
            percent_text = f"{latest_igra['percent_vs_baseline']:.1f}% vs prior 5-year same-date baseline"
        date_text = f"Latest IGRA date: {latest_igra['date'].date().isoformat()}"
        avg_text = f"7-day average: {latest_igra['launches_7d_avg']:.2f} soundings/day"

    nco_text = "Latest NCO model-ingest CONUS RAOB count: unavailable"
    if latest_nco is not None:
        nco_text = f"Latest NCO model-ingest CONUS RAOB count: {int(latest_nco['conus_count'])}"

    fig = setup_fig(10.8, 13.5)
    fig.text(
        0.07,
        0.94,
        "U.S. Upper-Air Observations\nAre Running Below Normal",
        fontsize=29,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.05,
    )
    fig.text(0.07, 0.755, percent_text, fontsize=25, fontweight="bold", color=RED)
    fig.text(0.07, 0.71, f"{date_text}  |  {avg_text}", fontsize=13.5, color=MUTED)

    ax = fig.add_axes([0.08, 0.32, 0.84, 0.31])
    plot_igra_trend(ax, igra, "", callout=False)
    ax.set_xlabel("")

    box = fig.add_axes([0.07, 0.18, 0.86, 0.085])
    box.set_facecolor("#ffffff")
    for spine in box.spines.values():
        spine.set_color("#d0d5dd")
        spine.set_linewidth(1.2)
    box.set_xticks([])
    box.set_yticks([])
    box.text(0.03, 0.55, nco_text, fontsize=16, fontweight="bold", color=INK, va="center")
    if latest_nco is not None:
        box.text(
            0.03,
            0.18,
            f"Cycle: {latest_nco['cycle_date_utc']} {str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco['model']}",
            fontsize=10.5,
            color=MUTED,
            va="center",
        )

    fig.text(
        0.07,
        0.1,
        "IGRA = archived soundings; NCO = model-ingest availability. Not a causation claim.",
        fontsize=12.5,
        color=INK,
    )
    fig.text(
        0.07,
        0.058,
        "NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages",
        fontsize=10.5,
        color=MUTED,
    )
    save_social(fig, path)


def latest_partial_day_status(igra: pd.DataFrame) -> tuple[bool, str]:
    current = current_igra(igra).dropna(subset=["date", "launches", "launches_7d_avg"])
    if len(current) < 15:
        return False, ""
    current = current.sort_values("date")
    latest = current.iloc[-1]
    prior = current.iloc[-15:-1]
    prior_median = float(prior["launches"].median())
    previous_avg = float(current.iloc[-2]["launches_7d_avg"])
    latest_launches = float(latest["launches"])
    latest_avg = float(latest["launches_7d_avg"])
    if prior_median > 0 and latest_launches < prior_median * 0.85 and latest_avg < previous_avg - 2.0:
        return True, (
            f"excluded {latest['date'].date().isoformat()} from plotted line; "
            f"daily count {latest_launches:.0f} vs prior-14-day median {prior_median:.0f}"
        )
    return False, ""


def plotted_igra_current(igra: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    current = current_igra(igra).dropna(subset=["date", "launches_7d_avg", "baseline_5yr_avg"]).sort_values("date")
    excluded, reason = latest_partial_day_status(igra)
    if excluded and len(current) > 1:
        current = current.iloc[:-1].copy()
    return current, excluded, reason


def latest_complete_igra_row(igra: pd.DataFrame) -> pd.Series | None:
    current, _excluded_partial, _reason = plotted_igra_current(igra)
    current = current.dropna(subset=["date", "launches_7d_avg", "baseline_5yr_avg"])
    if current.empty:
        return None
    return current.sort_values("date").iloc[-1]


def first_sustained_gap_date(
    current: pd.DataFrame,
    diff_column: str,
    threshold: float = -3.0,
    days: int = 7,
) -> pd.Timestamp | None:
    run = 0
    for index, value in enumerate(current[diff_column].fillna(np.inf)):
        run = run + 1 if value <= threshold else 0
        if run == days:
            return current.iloc[index - days + 1]["date"]
    return None


def short_month_day(value: pd.Timestamp) -> str:
    return f"{value.strftime('%b')} {value.day}"


def add_prior_year_baseline_columns(current: pd.DataFrame, all_rows: pd.DataFrame, years: int) -> pd.DataFrame:
    rows = current.copy()
    baselines: list[float] = []
    for _, row in rows.iterrows():
        prior = all_rows[
            (all_rows["year"] >= int(row["year"]) - years)
            & (all_rows["year"] < int(row["year"]))
            & (all_rows["month_day"] == row["month_day"])
        ]["launches_7d_avg"].dropna()
        baselines.append(float(prior.mean()) if len(prior) else np.nan)
    rows[f"baseline_{years}yr_alt"] = baselines
    rows[f"diff_{years}yr_alt"] = rows["launches_7d_avg"] - rows[f"baseline_{years}yr_alt"]
    return rows


def divergence_summary(igra: pd.DataFrame) -> dict[str, str]:
    plot_df, _excluded_partial, _reason = plotted_igra_current(igra)
    if plot_df.empty:
        return {
            "start": "unavailable",
            "detail": "Not enough IGRA rows",
            "shorter": "Shorter baselines unavailable",
        }
    start = first_sustained_gap_date(plot_df, "difference_vs_baseline", threshold=-3.0, days=7)
    all_rows = igra.copy()
    starts: list[str] = []
    for years in (1, 3):
        alt = add_prior_year_baseline_columns(plot_df, all_rows, years)
        alt_start = first_sustained_gap_date(alt, f"diff_{years}yr_alt", threshold=-3.0, days=7)
        if alt_start is not None:
            starts.append(f"{years}yr {short_month_day(alt_start)}")
    start_text = short_month_day(start) if start is not None else "unavailable"
    return {
        "start": start_text,
        "detail": "first 7-day run >=3/day below baseline",
        "shorter": "Shorter checks: " + ", ".join(starts) if starts else "Shorter checks unavailable",
    }


def analysis_rows_without_partial_latest(igra: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    current = current_igra(igra).dropna(subset=["date", "launches", "launches_7d_avg"]).sort_values("date")
    partial_date = None
    if len(current) >= 15:
        prior = current.iloc[-15:-1]
        latest = current.iloc[-1]
        if (
            float(latest["launches"]) < float(prior["launches"].median()) * 0.85
            and float(latest["launches_7d_avg"]) < float(current.iloc[-2]["launches_7d_avg"]) - 2.0
        ):
            partial_date = latest["date"]
    if partial_date is None:
        return igra.copy(), None
    return igra[igra["date"] != partial_date].copy(), partial_date


def monthly_divergence_from_pre2025(igra: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    rows, partial_date = analysis_rows_without_partial_latest(igra)
    rows = rows.copy()
    rows["month"] = rows["date"].dt.month
    rows["month_start"] = rows["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        rows.groupby(["year", "month", "month_start"])
        .agg(days=("date", "count"), launches=("launches", "sum"), daily_avg=("launches", "mean"))
        .reset_index()
    )
    baseline = (
        monthly[monthly["year"].isin([2021, 2022, 2023, 2024])]
        .groupby("month")["daily_avg"]
        .mean()
        .rename("baseline_daily_avg")
    )
    monthly = monthly.join(baseline, on="month")
    monthly["difference"] = monthly["daily_avg"] - monthly["baseline_daily_avg"]
    monthly["percent"] = monthly["difference"] / monthly["baseline_daily_avg"] * 100.0
    return monthly[monthly["year"].isin([2025, 2026])].copy(), partial_date


def first_rolling_gap_2025(igra: pd.DataFrame, threshold: float = -3.0) -> pd.Timestamp | None:
    rows, _partial_date = analysis_rows_without_partial_latest(igra)
    baseline = (
        rows[rows["year"].isin([2021, 2022, 2023, 2024])]
        .groupby("month_day")["launches"]
        .mean()
        .rename("baseline_launches")
    )
    year_rows = rows[rows["year"] == 2025].join(baseline, on="month_day").sort_values("date")
    if year_rows.empty:
        return None
    year_rows["daily_difference"] = year_rows["launches"] - year_rows["baseline_launches"]
    year_rows["rolling_30d_difference"] = year_rows["daily_difference"].rolling(30, min_periods=21).mean()
    below = year_rows[year_rows["rolling_30d_difference"] <= threshold]
    if below.empty:
        return None
    return below.iloc[0]["date"]


def rolling_percent_difference_from_pre2025(igra: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    rows, partial_date = analysis_rows_without_partial_latest(igra)
    baseline = (
        rows[rows["year"].isin([2021, 2022, 2023, 2024])]
        .groupby("month_day")["launches"]
        .mean()
        .rename("baseline_launches")
    )
    series = (
        rows[rows["year"].isin([2025, 2026])]
        .join(baseline, on="month_day")
        .sort_values("date")
        .copy()
    )
    series["observed_30d"] = series["launches"].rolling(30, min_periods=21).mean()
    series["baseline_30d"] = series["baseline_launches"].rolling(30, min_periods=21).mean()
    series["rolling_percent_difference"] = (
        (series["observed_30d"] - series["baseline_30d"]) / series["baseline_30d"] * 100.0
    )
    return series.dropna(subset=["rolling_percent_difference"]).copy(), partial_date


def detect_persistent_breakpoint(series: pd.DataFrame) -> pd.Series | None:
    if series.empty:
        return None
    values = series["rolling_percent_difference"].reset_index(drop=True)
    for position, value in enumerate(values):
        if value <= -3.0:
            following = values.iloc[position : position + 30]
            if len(following) == 30 and (following < 0).all():
                return series.iloc[position]
    return None


def monthly_percent_difference(series_rows: pd.DataFrame, year: int, month: int) -> float:
    subset = series_rows[(series_rows["year"] == year) & (series_rows["month"] == month)]
    if subset.empty:
        return float("nan")
    observed = float(subset["launches"].mean())
    baseline = float(subset["baseline_launches"].mean())
    return (observed - baseline) / baseline * 100.0 if baseline else float("nan")


def ytd_percent_difference(series_rows: pd.DataFrame, year: int) -> tuple[float, float, float, pd.Timestamp | None]:
    subset = series_rows[series_rows["year"] == year]
    if subset.empty:
        return float("nan"), float("nan"), float("nan"), None
    observed = float(subset["launches"].mean())
    baseline = float(subset["baseline_launches"].mean())
    percent = (observed - baseline) / baseline * 100.0 if baseline else float("nan")
    return percent, observed, baseline, subset["date"].max()


def decline_start_metrics(igra: pd.DataFrame) -> dict[str, object]:
    series, partial_date = rolling_percent_difference_from_pre2025(igra)
    breakpoint = detect_persistent_breakpoint(series)
    breakpoint_date = breakpoint["date"] if breakpoint is not None else None
    breakpoint_percent = (
        float(breakpoint["rolling_percent_difference"]) if breakpoint is not None else float("nan")
    )
    ytd_percent, ytd_observed, ytd_baseline, latest_complete = ytd_percent_difference(series, 2026)
    return {
        "series": series,
        "partial_date": partial_date,
        "breakpoint": breakpoint,
        "breakpoint_date": breakpoint_date,
        "breakpoint_percent": breakpoint_percent,
        "jan_2025_percent": monthly_percent_difference(series, 2025, 1),
        "mar_2025_percent": monthly_percent_difference(series, 2025, 3),
        "apr_2025_percent": monthly_percent_difference(series, 2025, 4),
        "ytd_2026_percent": ytd_percent,
        "ytd_2026_observed": ytd_observed,
        "ytd_2026_baseline": ytd_baseline,
        "latest_complete_date": latest_complete,
    }


def cumulative_deficit_since_breakpoint(igra: pd.DataFrame) -> dict[str, object]:
    metrics = decline_start_metrics(igra)
    series = metrics["series"].copy()
    breakpoint_date = metrics["breakpoint_date"]
    if series.empty or breakpoint_date is None:
        return {
            **metrics,
            "cumulative": pd.DataFrame(),
            "observed_total": float("nan"),
            "expected_total": float("nan"),
            "net_deficit": float("nan"),
            "percent_difference": float("nan"),
        }

    post = series[series["date"] >= breakpoint_date].copy()
    post["expected_daily"] = post["baseline_launches"]
    post["daily_net_deficit"] = post["expected_daily"] - post["launches"]
    post["cumulative_net_deficit"] = post["daily_net_deficit"].cumsum()

    observed_total = float(post["launches"].sum())
    expected_total = float(post["expected_daily"].sum())
    net_deficit = expected_total - observed_total
    percent_difference = (observed_total - expected_total) / expected_total * 100.0 if expected_total else float("nan")

    return {
        **metrics,
        "cumulative": post,
        "observed_total": observed_total,
        "expected_total": expected_total,
        "net_deficit": net_deficit,
        "percent_difference": percent_difference,
    }


def format_signed(value: float, suffix: str = "") -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}{suffix}"


def create_x_post_main_v2(data: dict[str, pd.DataFrame], path: Path) -> None:
    igra = data["igra"]
    nco = data["nco"]
    latest = latest_complete_igra_row(igra)
    latest_nco = latest_nco_row(nco)
    plot_df, excluded_partial, _reason = plotted_igra_current(igra)
    windows = rolling_windows(igra)
    window_90 = windows[windows["window"] == "90d"].iloc[0] if not windows.empty and (windows["window"] == "90d").any() else None

    latest_date = "unavailable"
    avg = baseline = diff = pct = np.nan
    if latest is not None:
        latest_date = latest["date"].date().isoformat()
        avg = float(latest["launches_7d_avg"])
        baseline = float(latest["baseline_5yr_avg"])
        diff = float(latest["difference_vs_baseline"])
        pct = float(latest["percent_vs_baseline"])

    nco_count = "unavailable"
    nco_cycle = "Cycle unavailable"
    if latest_nco is not None:
        nco_count = str(int(latest_nco["conus_count"]))
        nco_cycle = f"Cycle: {latest_nco['cycle_date_utc']} {str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco['model']}"

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    text = "#f8fafc"
    muted = "#a7b0c0"
    amber = "#fcd34d"
    blue = "#60a5fa"
    red = "#fb7185"
    shade = "#7f1d1d"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.94, "U.S. Upper-Air Observations", fontsize=27, fontweight="bold", color=text, va="top")
    fig.text(0.07, 0.835, f"{pct:.1f}%", fontsize=102, fontweight="bold", color=red, va="top")
    fig.text(
        0.56,
        0.812,
        "below the prior\n5-year same-date\nbaseline",
        fontsize=21,
        fontweight="bold",
        color=text,
        va="top",
        linespacing=1.08,
    )
    fig.text(
        0.07,
        0.675,
        f"Latest IGRA date: {latest_date}   |   2026 7-day avg: {avg:.2f} soundings/day",
        fontsize=13.5,
        color=muted,
    )
    fig.text(
        0.07,
        0.645,
        f"Baseline: {baseline:.2f}/day   |   Difference: {format_signed(diff)} soundings/day",
        fontsize=13.5,
        color=muted,
    )

    ax = fig.add_axes([0.07, 0.325, 0.86, 0.275], facecolor=panel)
    if not plot_df.empty:
        ax.plot(plot_df["date"], plot_df["baseline_5yr_avg"], color=amber, linewidth=3.0)
        ax.plot(plot_df["date"], plot_df["launches_7d_avg"], color=blue, linewidth=3.4)
        below = plot_df["launches_7d_avg"] < plot_df["baseline_5yr_avg"]
        ax.fill_between(
            plot_df["date"],
            plot_df["launches_7d_avg"],
            plot_df["baseline_5yr_avg"],
            where=below,
            color=shade,
            alpha=0.52,
            interpolate=True,
        )
        x_right = plot_df["date"].iloc[-1]
        ax.text(x_right + pd.Timedelta(days=5), plot_df["baseline_5yr_avg"].iloc[-1], "Prior 5-year\nbaseline", color=amber, fontsize=11, fontweight="bold", va="center")
        ax.text(x_right + pd.Timedelta(days=5), plot_df["launches_7d_avg"].iloc[-1], "2026", color=blue, fontsize=12, fontweight="bold", va="center")
        mid = plot_df.iloc[len(plot_df) // 2]
        ax.text(
            mid["date"],
            (mid["baseline_5yr_avg"] + mid["launches_7d_avg"]) / 2,
            "Missing observations\nvs baseline",
            color="#fecdd3",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
        )
        if excluded_partial:
            ax.text(
                0.985,
                0.055,
                "Latest incomplete day excluded from metrics and line",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="#e2e8f0",
                fontsize=10.0,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#1e293b",
                    "edgecolor": "#475569",
                    "alpha": 0.92,
                },
            )
        notices = [
            (pd.Timestamp("2025-03-20"), "NWS suspension notice:\nOmaha + Rapid City"),
            (pd.Timestamp("2025-04-17"), "NWS reduction notice:\nselected sites"),
        ]
        for notice_date, label in notices:
            if plot_df["date"].min() <= notice_date <= plot_df["date"].max():
                ax.axvline(notice_date, color="#cbd5e1", linewidth=1.0, linestyle=":")
                ax.text(notice_date, ax.get_ylim()[1], label, color=muted, fontsize=8, rotation=90, va="top", ha="right")
        ax.set_xlim(plot_df["date"].min() - pd.Timedelta(days=5), plot_df["date"].max() + pd.Timedelta(days=23))
    ax.tick_params(colors=muted, labelsize=10)
    ax.set_ylabel("soundings/day", color=muted, fontsize=10)
    ax.grid(True, axis="y", color="#334155", linewidth=0.8, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    card = fig.add_axes([0.07, 0.205, 0.52, 0.085], facecolor=panel2)
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_color("#334155")
    if window_90 is not None:
        deficit = float(window_90["deficit"])
        expected = float(window_90["expected"])
        observed = float(window_90["observed"])
        percent = float(window_90["percent"])
        card.text(0.04, 0.72, "Past 90 days", fontsize=12.0, fontweight="bold", color=muted, va="top")
        card.text(0.04, 0.47, f"{abs(deficit):.0f} fewer archived soundings vs baseline", fontsize=13.0, fontweight="bold", color=text, va="top")
        card.text(
            0.04,
            0.16,
            f"Observed: {observed:,.0f}  |  Expected: {expected:,.0f}  |  {percent:.1f}%",
            fontsize=10.5,
            color=muted,
            va="center",
        )

    nco_box = fig.add_axes([0.62, 0.205, 0.31, 0.085], facecolor=panel2)
    nco_box.set_xticks([])
    nco_box.set_yticks([])
    for spine in nco_box.spines.values():
        spine.set_color("#334155")
    nco_box.text(0.05, 0.7, f"Latest NCO model-ingest\nCONUS RAOB count: {nco_count}", fontsize=11.5, fontweight="bold", color=text, va="top")
    nco_box.text(0.05, 0.18, nco_cycle, fontsize=8.8, color=muted, va="center")

    fig.text(0.07, 0.151, "IGRA archive counts and NCO ingest counts are separate signals.", fontsize=12.5, color="#dbeafe")
    fig.text(0.07, 0.091, "NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages", fontsize=10.8, color=muted)
    fig.text(
        0.07,
        0.044,
        "Counts reflect available archive/operational records and may differ from actual launches due to ingest, archive, or reporting delays.\nNot a causation claim.",
        fontsize=10.2,
        color=muted,
        linespacing=1.22,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_x_post_main_v3(data: dict[str, pd.DataFrame], path: Path) -> None:
    igra = data["igra"]
    nco = data["nco"]
    latest = latest_complete_igra_row(igra)
    raw_latest = latest_igra_row(igra)
    latest_nco = latest_nco_row(nco)
    plot_df, excluded_partial, reason = plotted_igra_current(igra)
    windows = rolling_windows(igra)
    window_90 = windows[windows["window"] == "90d"].iloc[0] if not windows.empty and (windows["window"] == "90d").any() else None

    latest_date = "unavailable"
    avg = baseline = diff = pct = np.nan
    if latest is not None:
        latest_date = latest["date"].date().isoformat()
        avg = float(latest["launches_7d_avg"])
        baseline = float(latest["baseline_5yr_avg"])
        diff = float(latest["difference_vs_baseline"])
        pct = float(latest["percent_vs_baseline"])

    nco_count = "unavailable"
    nco_cycle = "Cycle unavailable"
    if latest_nco is not None:
        nco_count = str(int(latest_nco["conus_count"]))
        nco_cycle = f"Cycle: {latest_nco['cycle_date_utc']} {str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco['model']}"

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    amber = "#fcd34d"
    blue = "#60a5fa"
    red = "#fb7185"
    shade = "#7f1d1d"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.945, "U.S. Upper-Air Observations", fontsize=27, fontweight="bold", color=text, va="top")
    fig.text(
        0.07,
        0.908,
        "Archived CONUS IGRA soundings, Jan 1 through latest complete date",
        fontsize=12.5,
        color=muted,
        va="top",
    )
    fig.text(0.07, 0.838, f"{pct:.1f}%", fontsize=98, fontweight="bold", color=red, va="top")
    fig.text(
        0.56,
        0.816,
        "below the prior\n5-year same-date\nbaseline",
        fontsize=19,
        fontweight="bold",
        color=text,
        va="top",
        linespacing=1.15,
    )
    fig.text(
        0.07,
        0.682,
        f"Latest IGRA date: {latest_date}   |   2026 7-day avg: {avg:.2f} soundings/day",
        fontsize=13.0,
        color=muted,
    )
    fig.text(
        0.07,
        0.653,
        f"Baseline: {baseline:.2f}/day   |   Difference: {format_signed(diff)} soundings/day",
        fontsize=13.0,
        color=muted,
    )

    ax = fig.add_axes([0.07, 0.315, 0.86, 0.318], facecolor=panel)
    if not plot_df.empty:
        ax.plot(plot_df["date"], plot_df["baseline_5yr_avg"], color=amber, linewidth=3.0)
        ax.plot(plot_df["date"], plot_df["launches_7d_avg"], color=blue, linewidth=3.4)
        below = plot_df["launches_7d_avg"] < plot_df["baseline_5yr_avg"]
        ax.fill_between(
            plot_df["date"],
            plot_df["launches_7d_avg"],
            plot_df["baseline_5yr_avg"],
            where=below,
            color=shade,
            alpha=0.40,
            interpolate=True,
        )
        x_right = plot_df["date"].iloc[-1]
        ax.text(
            x_right + pd.Timedelta(days=7),
            plot_df["baseline_5yr_avg"].iloc[-1],
            "Prior 5-year\nbaseline",
            color=amber,
            fontsize=10.5,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.text(
            x_right + pd.Timedelta(days=7),
            plot_df["launches_7d_avg"].iloc[-1],
            "2026",
            color=blue,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="left",
        )
        mid = plot_df.iloc[len(plot_df) // 2]
        ax.text(
            mid["date"],
            (mid["baseline_5yr_avg"] + mid["launches_7d_avg"]) / 2,
            "Below baseline",
            color="#fecdd3",
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
        )
        event_year = int(plot_df["date"].dt.year.max())
        notices = [
            (pd.Timestamp(event_year, 3, 20), "NWS suspension notice"),
            (pd.Timestamp(event_year, 4, 17), "NWS reduction notice"),
        ]
        for notice_date, label in notices:
            if plot_df["date"].min() <= notice_date <= plot_df["date"].max():
                ax.axvline(notice_date, color="#94a3b8", linewidth=1.0, linestyle=":", alpha=0.75)
                ax.text(
                    notice_date + pd.Timedelta(days=1),
                    ax.get_ylim()[1] - 0.35,
                    label,
                    color=muted,
                    fontsize=7.8,
                    rotation=90,
                    va="top",
                    ha="left",
                )
        ax.set_xlim(plot_df["date"].min() - pd.Timedelta(days=5), plot_df["date"].max() + pd.Timedelta(days=32))
    ax.tick_params(colors=muted, labelsize=10)
    ax.set_ylabel("soundings/day", color=muted, fontsize=10)
    ax.grid(True, axis="y", color=border, linewidth=0.8, alpha=0.72)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    card = fig.add_axes([0.07, 0.19, 0.52, 0.095], facecolor=panel2)
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_color(border)
    if window_90 is not None:
        deficit = float(window_90["deficit"])
        expected = float(window_90["expected"])
        observed = float(window_90["observed"])
        percent = float(window_90["percent"])
        card.text(0.04, 0.79, "Past 90 days", fontsize=12.3, fontweight="bold", color=muted, va="top")
        card.text(0.04, 0.53, f"{abs(deficit):.0f} fewer archived soundings", fontsize=16.5, fontweight="bold", color=text, va="top")
        card.text(
            0.04,
            0.18,
            f"Observed {observed:,.0f} vs expected {expected:,.0f} ({percent:.1f}%)",
            fontsize=10.8,
            color=muted,
            va="center",
        )

    nco_box = fig.add_axes([0.62, 0.19, 0.31, 0.095], facecolor=panel2)
    nco_box.set_xticks([])
    nco_box.set_yticks([])
    for spine in nco_box.spines.values():
        spine.set_color(border)
    nco_box.text(0.05, 0.78, "Latest NCO model-ingest", fontsize=10.7, fontweight="bold", color=muted, va="top")
    nco_box.text(0.05, 0.54, f"{nco_count} CONUS RAOBs", fontsize=15.5, fontweight="bold", color=text, va="top")
    nco_box.text(0.05, 0.26, nco_cycle, fontsize=8.8, color=muted, va="center")
    nco_box.text(0.05, 0.08, "Separate operational ingest layer", fontsize=7.5, color="#8792a5", va="bottom")

    partial_note = ""
    if excluded_partial and raw_latest is not None:
        partial_note = f" Preliminary {raw_latest['date'].date().isoformat()} excluded as incomplete."
    fig.text(0.07, 0.128, "Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages", fontsize=10.2, color=muted)
    fig.text(
        0.07,
        0.084,
        "IGRA archive counts and NCO ingest counts are separate signals. Not a causation claim." + partial_note,
        fontsize=9.6,
        color=muted,
        wrap=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_x_post_2025_divergence(data: dict[str, pd.DataFrame], path: Path) -> None:
    igra = data["igra"]
    monthly, partial_date = monthly_divergence_from_pre2025(igra)
    onset = first_rolling_gap_2025(igra)

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    amber = "#fcd34d"
    blue = "#60a5fa"
    grid = "#334155"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.945, "When Did the Decline Start?", fontsize=28, fontweight="bold", color=text, va="top")
    fig.text(
        0.07,
        0.905,
        "30-day archived CONUS IGRA soundings vs 2021-2024 same-date baseline",
        fontsize=12.8,
        color=muted,
        va="top",
    )
    fig.text(0.07, 0.835, "March-April 2025", fontsize=46, fontweight="bold", color=red, va="top")
    fig.text(
        0.07,
        0.775,
        "The broader divergence appears before 2026, then persists into 2026.",
        fontsize=15,
        color=text,
        va="top",
    )

    timeline = fig.add_axes([0.07, 0.705, 0.86, 0.052], facecolor=panel2)
    timeline.set_xticks([])
    timeline.set_yticks([])
    for spine in timeline.spines.values():
        spine.set_color(border)
    timeline.text(
        0.02,
        0.66,
        "DOGE/NOAA cuts timeline (context): 1 Feb 4 DOGE enters NOAA HQ  |  2 Feb 27 federal layoffs via DOGE",
        color=muted,
        fontsize=8.8,
        va="center",
        ha="left",
    )
    timeline.text(
        0.02,
        0.26,
        "3 Mar 31 NWS reduced/closed balloon sites  |  4 Jun 29 DOGE NOAA contracts list",
        color=muted,
        fontsize=8.8,
        va="center",
        ha="left",
    )

    rows, partial_date = analysis_rows_without_partial_latest(igra)
    baseline_daily = (
        rows[rows["year"].isin([2021, 2022, 2023, 2024])]
        .groupby("month_day")["launches"]
        .mean()
        .rename("baseline_launches")
    )
    line_rows = rows[rows["year"].isin([2025, 2026])].join(baseline_daily, on="month_day").sort_values("date")
    line_rows["observed_30d"] = line_rows["launches"].rolling(30, min_periods=21).mean()
    line_rows["baseline_30d"] = line_rows["baseline_launches"].rolling(30, min_periods=21).mean()
    line_rows["percent_30d"] = (line_rows["observed_30d"] - line_rows["baseline_30d"]) / line_rows["baseline_30d"] * 100.0
    line_rows = line_rows.dropna(subset=["percent_30d"]).copy()

    ax = fig.add_axes([0.08, 0.36, 0.85, 0.325], facecolor=panel)
    if not line_rows.empty:
        ax.axhline(0, color="#cbd5e1", linewidth=1.0, alpha=0.9)
        ax.axhline(-3, color=red, linewidth=1.0, linestyle=":", alpha=0.78)
        ax.plot(line_rows["date"], line_rows["percent_30d"], color=blue, linewidth=3.2)
        ax.fill_between(
            line_rows["date"],
            line_rows["percent_30d"],
            0,
            where=line_rows["percent_30d"] < 0,
            color="#7f1d1d",
            alpha=0.38,
            interpolate=True,
        )
        ax.text(
            line_rows["date"].iloc[-1] + pd.Timedelta(days=13),
            line_rows["percent_30d"].iloc[-1],
            "30-day\nobserved",
            color=blue,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.text(
            pd.Timestamp("2025-09-01"),
            -4.7,
            "Below 2021-2024 baseline",
            color="#fecdd3",
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
        )
        if onset is not None:
            ax.axvline(onset, color="#e2e8f0", linewidth=1.4, linestyle="--", alpha=0.85)
            ax.text(
                onset + pd.Timedelta(days=8),
                -7.35,
                f"First sustained\n30-day gap:\n{short_month_day(onset)}",
                color="#e2e8f0",
                fontsize=9.2,
                va="bottom",
                ha="left",
            )
        events = [
            (pd.Timestamp("2025-02-04"), "1", 0.72),
            (pd.Timestamp("2025-02-27"), "2", 0.45),
            (pd.Timestamp("2025-03-31"), "3", 0.18),
            (pd.Timestamp("2025-06-29"), "4", 0.72),
        ]
        for event_date, label, y_frac in events:
            if line_rows["date"].min() <= event_date <= line_rows["date"].max():
                ax.axvline(event_date, color="#94a3b8", linewidth=0.9, linestyle=":", alpha=0.66)
                ax.text(
                    event_date,
                    y_frac,
                    label,
                    transform=ax.get_xaxis_transform(),
                    color=muted,
                    fontsize=8.5,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    bbox={
                        "boxstyle": "circle,pad=0.18",
                        "facecolor": panel2,
                        "edgecolor": "#94a3b8",
                        "linewidth": 0.8,
                        "alpha": 0.95,
                    },
                )
        ax.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), line_rows["date"].max() + pd.Timedelta(days=52))
        ax.set_ylim(min(-8.2, float(line_rows["percent_30d"].min()) - 0.8), 1.1)
    ax.set_ylabel("% vs baseline", color=muted, fontsize=10)
    ax.tick_params(colors=muted, labelsize=10)
    ax.grid(True, axis="y", color=grid, linewidth=0.8, alpha=0.68)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    card_2025 = fig.add_axes([0.07, 0.235, 0.4, 0.105], facecolor=panel2)
    card_2025.set_xticks([])
    card_2025.set_yticks([])
    for spine in card_2025.spines.values():
        spine.set_color(border)
    summary_2025 = monthly[monthly["year"] == 2025]
    if not summary_2025.empty:
        jan = summary_2025[summary_2025["month"] == 1]["percent"].iloc[0]
        mar = summary_2025[summary_2025["month"] == 3]["percent"].iloc[0]
        apr = summary_2025[summary_2025["month"] == 4]["percent"].iloc[0]
        card_2025.text(0.05, 0.78, "2025 shift", fontsize=12, fontweight="bold", color=muted, va="top")
        card_2025.text(0.05, 0.51, f"Jan near baseline ({jan:+.1f}%)", fontsize=13, fontweight="bold", color=text, va="top")
        card_2025.text(0.05, 0.22, f"Mar {mar:+.1f}%  |  Apr {apr:+.1f}%", fontsize=11, color=muted, va="center")

    card_2026 = fig.add_axes([0.5, 0.235, 0.43, 0.105], facecolor=panel2)
    card_2026.set_xticks([])
    card_2026.set_yticks([])
    for spine in card_2026.spines.values():
        spine.set_color(border)
    summary_2026 = monthly[monthly["year"] == 2026]
    if not summary_2026.empty:
        ytd = summary_2026["launches"].sum() / summary_2026["days"].sum()
        baseline = (summary_2026["baseline_daily_avg"] * summary_2026["days"]).sum() / summary_2026["days"].sum()
        pct = (ytd - baseline) / baseline * 100.0
        card_2026.text(0.05, 0.78, "2026 continuation", fontsize=12, fontweight="bold", color=muted, va="top")
        card_2026.text(0.05, 0.51, f"YTD through latest complete date: {pct:+.1f}%", fontsize=12.7, fontweight="bold", color=text, va="top")
        card_2026.text(0.05, 0.22, f"{ytd:.1f}/day vs {baseline:.1f}/day baseline", fontsize=10.8, color=muted, va="center")

    note = ""
    if partial_date is not None:
        note = f" Note: {partial_date.date().isoformat()} excluded from the 2026 rolling/YTD comparison as preliminary."
    fig.text(0.07, 0.16, "Interpretation: 2026 is lower, but the broader divergence begins in 2025.", fontsize=13.2, color="#dbeafe")
    fig.text(0.07, 0.105, "Sources: NOAA/NCEI IGRA v2.", fontsize=10.5, color=muted)
    fig.text(
        0.07,
        0.068,
        "Counts reflect available archive records and may differ from actual launches due to ingest, archive, or reporting delays. Not a causation claim." + note,
        fontsize=9.5,
        color=muted,
        wrap=True,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_decline_start_v2(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = decline_start_metrics(data["igra"])
    series = metrics["series"]
    breakpoint = metrics["breakpoint"]
    breakpoint_date = metrics["breakpoint_date"]
    partial_date = metrics["partial_date"]

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    blue = "#60a5fa"
    grid = "#334155"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(
        0.07,
        0.95,
        "When Did U.S. Upper-Air Observations\nBegin Falling Below Normal?",
        fontsize=24.5,
        fontweight="bold",
        color=text,
        va="top",
        linespacing=1.05,
    )
    fig.text(
        0.07,
        0.885,
        "30-day archived CONUS IGRA soundings vs 2021-2024 same-date baseline",
        fontsize=12.5,
        color=muted,
        va="top",
    )

    if breakpoint_date is not None:
        answer = f"Persistent divergence begins:\n{breakpoint_date.strftime('%b')} {breakpoint_date.day}, {breakpoint_date.year}"
    else:
        answer = "Persistent divergence:\nnot detected"
        print("WARNING: No persistent below-baseline breakpoint detected.", flush=True)
    fig.text(0.07, 0.82, answer, fontsize=33, fontweight="bold", color=red, va="top", linespacing=1.08)

    ax = fig.add_axes([0.08, 0.365, 0.85, 0.355], facecolor=panel)
    if not series.empty:
        ax.axhline(0, color="#e2e8f0", linewidth=1.4, alpha=0.95)
        ax.text(
            series["date"].min() + pd.Timedelta(days=12),
            0.22,
            "Expected level",
            color="#e2e8f0",
            fontsize=9.5,
            fontweight="bold",
            va="bottom",
        )
        ax.plot(series["date"], series["rolling_percent_difference"], color=blue, linewidth=3.2)
        if breakpoint_date is not None:
            shaded = series[series["date"] >= breakpoint_date]
            ax.fill_between(
                shaded["date"],
                shaded["rolling_percent_difference"],
                0,
                where=shaded["rolling_percent_difference"] < 0,
                color="#7f1d1d",
                alpha=0.42,
                interpolate=True,
            )
            ax.axvline(breakpoint_date, color=red, linewidth=1.4, linestyle="--", alpha=0.95)
            ax.text(
                breakpoint_date + pd.Timedelta(days=10),
                min(-6.8, float(series["rolling_percent_difference"].min()) + 0.6),
                "First sustained\nbelow-baseline period",
                color=text,
                fontsize=10.3,
                fontweight="bold",
                va="bottom",
                ha="left",
            )
        for event_date, label in [
            (pd.Timestamp("2025-03-20"), "NWS suspension notice"),
            (pd.Timestamp("2025-04-17"), "NWS reduction notice"),
        ]:
            if series["date"].min() <= event_date <= series["date"].max():
                ax.axvline(event_date, color="#94a3b8", linewidth=0.9, linestyle=":", alpha=0.62)
                ax.text(
                    event_date + pd.Timedelta(days=3),
                    0.90,
                    label,
                    transform=ax.get_xaxis_transform(),
                    color=muted,
                    fontsize=7.9,
                    rotation=90,
                    va="top",
                    ha="left",
                )
        latest = series.iloc[-1]
        ax.text(
            latest["date"] + pd.Timedelta(days=13),
            latest["rolling_percent_difference"],
            "30-day observed\nvs baseline",
            color=blue,
            fontsize=10.7,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), series["date"].max() + pd.Timedelta(days=58))
        ax.set_ylim(min(-8.5, float(series["rolling_percent_difference"].min()) - 0.7), 1.2)
    ax.set_ylabel("Difference from 2021-2024 average (%)", color=muted, fontsize=9.8)
    ax.tick_params(colors=muted, labelsize=10)
    ax.grid(True, axis="y", color=grid, linewidth=0.8, alpha=0.68)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    card_1 = fig.add_axes([0.07, 0.225, 0.4, 0.095], facecolor=panel2)
    card_1.set_xticks([])
    card_1.set_yticks([])
    for spine in card_1.spines.values():
        spine.set_color(border)
    detected_text = (
        f"Detected: {breakpoint_date.strftime('%b')} {breakpoint_date.day}, {breakpoint_date.year}"
        if breakpoint_date is not None
        else "Detected: none"
    )
    card_1.text(0.05, 0.78, "2025 shift", fontsize=12.0, fontweight="bold", color=muted, va="top")
    card_1.text(0.05, 0.52, "First sustained below-baseline period", fontsize=12.0, fontweight="bold", color=text, va="top")
    card_1.text(0.05, 0.22, detected_text, fontsize=11.0, color=muted, va="center")

    card_2 = fig.add_axes([0.5, 0.225, 0.43, 0.095], facecolor=panel2)
    card_2.set_xticks([])
    card_2.set_yticks([])
    for spine in card_2.spines.values():
        spine.set_color(border)
    ytd_percent = float(metrics["ytd_2026_percent"])
    ytd_observed = float(metrics["ytd_2026_observed"])
    ytd_baseline = float(metrics["ytd_2026_baseline"])
    card_2.text(0.05, 0.78, "2026 continuation", fontsize=12.0, fontweight="bold", color=muted, va="top")
    card_2.text(0.05, 0.52, f"YTD through latest complete date: {ytd_percent:+.1f}%", fontsize=12.0, fontweight="bold", color=text, va="top")
    card_2.text(0.05, 0.22, f"{ytd_observed:.1f}/day vs {ytd_baseline:.1f}/day baseline", fontsize=10.5, color=muted, va="center")

    footer = "Counts reflect available archive records and may differ from actual launches due to ingest, archive, or reporting delays. Not a causation claim."
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    fig.text(0.07, 0.145, "Source: NOAA/NCEI IGRA v2.", fontsize=10.4, color=muted)
    fig.text(0.07, 0.102, footer, fontsize=9.5, color=muted, wrap=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_decline_start_v3(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = decline_start_metrics(data["igra"])
    series = metrics["series"]
    breakpoint_date = metrics["breakpoint_date"]
    partial_date = metrics["partial_date"]

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    blue = "#60a5fa"
    grid = "#334155"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(
        0.07,
        0.95,
        "When Did U.S. Upper-Air Observations\nBegin Falling Below Normal?",
        fontsize=24.5,
        fontweight="bold",
        color=text,
        va="top",
        linespacing=1.05,
    )
    fig.text(
        0.07,
        0.885,
        "30-day archived CONUS IGRA soundings vs 2021-2024 same-date baseline",
        fontsize=12.5,
        color=muted,
        va="top",
    )

    if breakpoint_date is not None:
        answer = f"Persistent divergence begins: {breakpoint_date.strftime('%b')} {breakpoint_date.day}, {breakpoint_date.year}"
    else:
        answer = "Persistent divergence not detected"
        print("WARNING: No persistent below-baseline breakpoint detected.", flush=True)
    fig.text(0.07, 0.815, answer, fontsize=28.5, fontweight="bold", color=red, va="top")

    ax = fig.add_axes([0.08, 0.315, 0.85, 0.405], facecolor=panel)
    if not series.empty:
        ax.axhline(0, color="#e2e8f0", linewidth=1.35, alpha=0.95)
        ax.text(
            series["date"].min() + pd.Timedelta(days=10),
            0.22,
            "Expected level",
            color="#e2e8f0",
            fontsize=9.8,
            fontweight="bold",
            va="bottom",
        )
        ax.plot(series["date"], series["rolling_percent_difference"], color=blue, linewidth=3.25)
        if breakpoint_date is not None:
            shaded = series[series["date"] >= breakpoint_date]
            ax.fill_between(
                shaded["date"],
                shaded["rolling_percent_difference"],
                0,
                where=shaded["rolling_percent_difference"] < 0,
                color="#7f1d1d",
                alpha=0.40,
                interpolate=True,
            )
            ax.axvline(breakpoint_date, color=red, linewidth=1.35, linestyle="--", alpha=0.95)
        for event_date, label, x_offset in [
            (pd.Timestamp("2025-03-20"), "Mar 20, 2025\nNWS suspension notice", -45),
            (pd.Timestamp("2025-04-17"), "Apr 17, 2025\nNWS reduction notice", 34),
        ]:
            if series["date"].min() <= event_date <= series["date"].max():
                ax.axvline(event_date, color="#94a3b8", linewidth=0.9, linestyle=":", alpha=0.58)
                ax.annotate(
                    label,
                    xy=(event_date, 0.05),
                    xycoords="data",
                    xytext=(x_offset, 34),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.9},
                    color="#dbeafe",
                    fontsize=8.7,
                    ha="center",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": panel2,
                        "edgecolor": border,
                        "alpha": 0.96,
                    },
                    clip_on=False,
                )
        latest = series.iloc[-1]
        ax.text(
            latest["date"] + pd.Timedelta(days=13),
            latest["rolling_percent_difference"],
            "30-day observed\nvs baseline",
            color=blue,
            fontsize=10.7,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), series["date"].max() + pd.Timedelta(days=58))
        ax.set_ylim(min(-8.5, float(series["rolling_percent_difference"].min()) - 0.7), 1.55)
    ax.set_ylabel("Difference from 2021-2024 average (%)", color=muted, fontsize=9.8)
    ax.tick_params(colors=muted, labelsize=10)
    ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.46)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    card_1 = fig.add_axes([0.07, 0.19, 0.4, 0.09], facecolor=panel2)
    card_1.set_xticks([])
    card_1.set_yticks([])
    for spine in card_1.spines.values():
        spine.set_color(border)
    detected_text = (
        f"Detected: {breakpoint_date.strftime('%b')} {breakpoint_date.day}, {breakpoint_date.year}"
        if breakpoint_date is not None
        else "Detected: none"
    )
    card_1.text(0.05, 0.78, "2025 shift", fontsize=12.0, fontweight="bold", color=muted, va="top")
    card_1.text(0.05, 0.52, "Data-driven breakpoint", fontsize=11.7, fontweight="bold", color=text, va="top")
    card_1.text(0.05, 0.22, detected_text, fontsize=10.8, color=muted, va="center")

    card_2 = fig.add_axes([0.5, 0.19, 0.43, 0.09], facecolor=panel2)
    card_2.set_xticks([])
    card_2.set_yticks([])
    for spine in card_2.spines.values():
        spine.set_color(border)
    ytd_percent = float(metrics["ytd_2026_percent"])
    ytd_observed = float(metrics["ytd_2026_observed"])
    ytd_baseline = float(metrics["ytd_2026_baseline"])
    card_2.text(0.05, 0.78, "2026 continuation", fontsize=12.0, fontweight="bold", color=muted, va="top")
    card_2.text(0.05, 0.52, f"YTD through latest complete date: {ytd_percent:+.1f}%", fontsize=11.7, fontweight="bold", color=text, va="top")
    card_2.text(0.05, 0.22, f"{ytd_observed:.1f}/day vs {ytd_baseline:.1f}/day baseline", fontsize=10.2, color=muted, va="center")

    footer = (
        "Counts reflect available archive records and may differ from actual launches due to ingest, archive, "
        "or reporting delays. Not a causation claim."
    )
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    fig.text(0.07, 0.125, "Source: NOAA/NCEI IGRA v2.", fontsize=10.0, color=muted)
    fig.text(0.07, 0.082, textwrap.fill(footer, width=135), fontsize=9.2, color=muted, linespacing=1.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_cumulative_deficit_context(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = cumulative_deficit_since_breakpoint(data["igra"])
    post = metrics["cumulative"]
    breakpoint_date = metrics["breakpoint_date"]
    latest_complete = metrics["latest_complete_date"]
    partial_date = metrics["partial_date"]
    observed_total = float(metrics["observed_total"])
    expected_total = float(metrics["expected_total"])
    net_deficit = float(metrics["net_deficit"])
    percent_difference = float(metrics["percent_difference"])

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    red_dark = "#7f1d1d"
    amber = "#fcd34d"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.95, "The Missing Soundings Added Up", fontsize=30, fontweight="bold", color=text, va="top")
    fig.text(
        0.07,
        0.91,
        "Archived CONUS IGRA soundings after the data-driven Apr 3, 2025 breakpoint",
        fontsize=12.7,
        color=muted,
        va="top",
    )

    if post.empty or breakpoint_date is None or latest_complete is None:
        fig.text(0.07, 0.78, "Cumulative deficit unavailable", fontsize=34, fontweight="bold", color=red)
    else:
        fig.text(0.07, 0.835, f"{net_deficit:,.0f}", fontsize=92, fontweight="bold", color=red, va="top")
        fig.text(
            0.55,
            0.815,
            "fewer archived\nsoundings vs baseline",
            fontsize=23,
            fontweight="bold",
            color=text,
            va="top",
            linespacing=1.12,
        )
        date_range = f"{breakpoint_date.date().isoformat()} to {latest_complete.date().isoformat()}"
        fig.text(
            0.07,
            0.715,
            f"Observed {observed_total:,.0f}  |  Expected {expected_total:,.0f}  |  {percent_difference:+.1f}%  |  {date_range}",
            fontsize=12.6,
            color=muted,
            va="top",
        )

    ax = fig.add_axes([0.08, 0.34, 0.85, 0.34], facecolor=panel)
    if not post.empty:
        ax.plot(post["date"], post["cumulative_net_deficit"], color=red, linewidth=3.6)
        ax.fill_between(
            post["date"],
            post["cumulative_net_deficit"],
            0,
            color=red_dark,
            alpha=0.36,
        )
        ax.axhline(0, color="#e2e8f0", linewidth=1.1, alpha=0.8)
        ax.axvline(breakpoint_date, color=red, linewidth=1.25, linestyle="--", alpha=0.95)
        ax.text(
            breakpoint_date + pd.Timedelta(days=9),
            max(120, float(post["cumulative_net_deficit"].max()) * 0.08),
            "Apr 3\nbreakpoint",
            color="#fecdd3",
            fontsize=9.2,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        latest = post.iloc[-1]
        ax.scatter([latest["date"]], [latest["cumulative_net_deficit"]], color=red, s=42, zorder=4)
        ax.text(
            latest["date"] + pd.Timedelta(days=13),
            latest["cumulative_net_deficit"],
            "Cumulative net\nshortfall",
            color=red,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )
        for event_date, label in [
            (pd.Timestamp("2025-02-04"), "Feb 4\nDOGE at NOAA HQ"),
            (pd.Timestamp("2025-02-27"), "Feb 27\nNOAA layoffs reported"),
            (pd.Timestamp("2025-03-20"), "Mar 20\nNWS suspension notice"),
            (pd.Timestamp("2025-04-17"), "Apr 17\nNWS reduction notice"),
            (pd.Timestamp("2025-06-29"), "Jun 29\nDOGE NOAA contract list"),
        ]:
            if pd.Timestamp("2025-01-01") <= event_date <= post["date"].max():
                ax.axvline(event_date, color="#94a3b8", linewidth=0.8, linestyle=":", alpha=0.38)
        ax.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), post["date"].max() + pd.Timedelta(days=58))
        ax.set_ylim(0, max(3200, float(post["cumulative_net_deficit"].max()) * 1.16))
    ax.set_ylabel("Cumulative net fewer archived soundings", color=muted, fontsize=9.8)
    ax.tick_params(colors=muted, labelsize=10)
    ax.grid(True, axis="y", color=border, linewidth=0.65, alpha=0.43)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    card = fig.add_axes([0.07, 0.205, 0.86, 0.105], facecolor=panel2)
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_color(border)
    card.text(0.03, 0.78, "Context timeline", fontsize=11.5, fontweight="bold", color=muted, va="top")
    card.text(0.23, 0.78, "Markers are context only; not a causation claim.", fontsize=8.7, color=muted, va="top")
    timeline_items = [
        ("Feb 4", "DOGE at\nNOAA HQ"),
        ("Feb 27", "NOAA layoffs\nreported"),
        ("Mar 20", "NWS suspension\nnotice"),
        ("Apr 17", "NWS reduction\nnotice"),
        ("Jun 29", "DOGE NOAA\ncontract list"),
    ]
    x_positions = [0.03, 0.22, 0.42, 0.62, 0.81]
    for (date_label, event_label), x in zip(timeline_items, x_positions):
        card.text(x, 0.48, date_label, fontsize=12.4, fontweight="bold", color=amber, va="top")
        card.text(x, 0.25, event_label, fontsize=8.6, color=text, va="top", linespacing=1.12)

    basis = fig.add_axes([0.07, 0.125, 0.86, 0.055], facecolor=bg)
    basis.set_axis_off()
    basis.text(
        0.0,
        0.8,
        "Method: daily CONUS IGRA archive count minus 2021-2024 same-date average, cumulated from the detected breakpoint.",
        fontsize=10.4,
        color="#dbeafe",
        va="top",
    )
    footer = (
        "Sources: NOAA/NCEI IGRA v2; public reporting and NWS notices for context markers. "
        "Counts reflect available archive records and may differ from actual launches due to ingest, archive, or reporting delays. "
        "Not a causation claim."
    )
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    fig.text(0.07, 0.055, textwrap.fill(footer, width=145), fontsize=8.5, color=muted, linespacing=1.18)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def complete_current_windows(igra: pd.DataFrame, windows: tuple[int, ...] = (30, 60, 90, 180)) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    rows, partial_date = analysis_rows_without_partial_latest(igra)
    current = rows[rows["year"] == rows["year"].max()].dropna(subset=["date", "launches", "baseline_5yr_avg"]).copy()
    if current.empty:
        return pd.DataFrame(), partial_date
    latest = current["date"].max()
    records = []
    for days in windows:
        start = latest - pd.Timedelta(days=days - 1)
        subset = current[(current["date"] >= start) & (current["date"] <= latest)]
        if subset.empty:
            continue
        observed = float(subset["launches"].sum())
        expected = float(subset["baseline_5yr_avg"].sum())
        deficit = observed - expected
        records.append(
            {
                "window": f"{days}d",
                "days": days,
                "observed": observed,
                "expected": expected,
                "deficit": deficit,
                "percent": deficit / expected * 100.0 if expected else np.nan,
                "latest": latest,
            }
        )
    return pd.DataFrame(records), partial_date


def context_events() -> list[tuple[pd.Timestamp, str]]:
    return [
        (pd.Timestamp("2025-02-04"), "DOGE at NOAA HQ"),
        (pd.Timestamp("2025-02-27"), "NOAA layoffs reported"),
        (pd.Timestamp("2025-03-20"), "NWS suspension notice"),
        (pd.Timestamp("2025-04-17"), "NWS reduction notice"),
        (pd.Timestamp("2025-06-29"), "DOGE NOAA contract list"),
    ]


def dark_note_footer(fig, text: str, y: float = 0.052, width: int = 145) -> None:
    fig.text(0.07, y, textwrap.fill(text, width=width), fontsize=8.5, color="#a7b0c0", linespacing=1.18)


def editorial_palette() -> dict[str, str]:
    return {
        "bg": "#fbfaf6",
        "panel": "#ffffff",
        "ink": "#18212f",
        "muted": "#647084",
        "grid": "#d8dee8",
        "blue": "#1f6aa5",
        "red": "#c84a32",
        "red_soft": "#f2d7cf",
        "amber": "#c98217",
        "green": "#2b8c65",
        "border": "#c9d2df",
        "land": "#e7ecef",
        "water": "#f5f8fb",
    }


def editorial_fig():
    colors = editorial_palette()
    return plt.figure(figsize=(10.8, 13.5), facecolor=colors["bg"])


def save_editorial(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=editorial_palette()["bg"])
    plt.close(fig)


def editorial_footer(fig, text: str, y: float = 0.055, width: int = 145) -> None:
    colors = editorial_palette()
    fig.text(0.07, y, textwrap.fill(text, width=width), fontsize=8.8, color=colors["muted"], linespacing=1.18)


def style_editorial_axis(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    colors = editorial_palette()
    ax.set_facecolor(colors["panel"])
    ax.tick_params(colors=colors["muted"], labelsize=10)
    grid_axis = "x" if xgrid and not ygrid else "y"
    if xgrid and ygrid:
        ax.grid(True, color=colors["grid"], linewidth=0.75, alpha=0.7)
    else:
        ax.grid(True, axis=grid_axis, color=colors["grid"], linewidth=0.75, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(colors["border"])


def draw_editorial_conus_base(ax) -> None:
    colors = editorial_palette()
    ax.set_facecolor(colors["water"])
    if CONUS_STATE_GEOJSON_PATH.exists():
        try:
            geojson = json.loads(CONUS_STATE_GEOJSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            geojson = {"features": []}
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not isinstance(geometry, dict):
                continue
            for ring in iter_geometry_rings(geometry):
                xs = [float(point[0]) for point in ring]
                ys = [float(point[1]) for point in ring]
                if max(xs) < -126 or min(xs) > -66 or max(ys) < 24 or min(ys) > 50:
                    continue
                ax.fill(xs, ys, facecolor=colors["land"], edgecolor=colors["border"], linewidth=0.45, zorder=0)
                ax.plot(xs, ys, color="#9aa8b8", linewidth=0.28, alpha=0.95, zorder=1)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(colors["border"])


def latest_station_status_rows(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    stations = active_conus_station_frame(data.get("stations", pd.DataFrame()))
    nco = data.get("nco", pd.DataFrame())
    issues = data.get("issues", pd.DataFrame())
    if stations.empty:
        return pd.DataFrame(), pd.DataFrame(), "latest cycle unavailable"
    status_rows = station_statuses(stations, issues, nco).dropna(subset=["latitude", "longitude"]).copy()
    impacted = status_rows[~status_rows["status"].isin(["available/no issue", "unknown"])].copy()
    latest = latest_cycle(nco)
    cycle_text = "latest cycle unavailable"
    if latest is not None:
        cycle_text = f"{latest['cycle_date_utc']} {str(latest['cycle_hour']).zfill(2)}Z {latest['model']}"
    return status_rows, impacted, cycle_text


def social_palette() -> dict[str, str]:
    """High-contrast palette for the 1080x1350 social graphic series."""
    return {
        "bg": "#071827",
        "panel": "#102b45",
        "panel_alt": "#143957",
        "ink": "#f8fbff",
        "muted": "#b5c5d8",
        "grid": "#355775",
        "blue": "#62c6ff",
        "red": "#ff7657",
        "red_soft": "#743844",
        "amber": "#ffd166",
        "green": "#5ed3a5",
        "border": "#31506d",
        "land": "#173752",
        "water": "#0b2238",
    }


def social_fig():
    return plt.figure(figsize=(10.8, 13.5), facecolor=social_palette()["bg"])


def save_social_series(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=social_palette()["bg"])
    plt.close(fig)


def social_card(fig, x: float, y: float, width: float, height: float, *, alt: bool = False) -> None:
    colors = social_palette()
    fig.patches.append(
        Rectangle(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            facecolor=colors["panel_alt"] if alt else colors["panel"],
            edgecolor=colors["border"],
            linewidth=0.9,
            zorder=-1,
        )
    )


def social_eyebrow(fig, text: str) -> None:
    colors = social_palette()
    fig.text(0.07, 0.955, text.upper(), fontsize=10.8, fontweight="bold", color=colors["blue"], va="top")


def style_social_axis(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    colors = social_palette()
    ax.set_facecolor(colors["panel"])
    ax.tick_params(colors=colors["muted"], labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(colors["border"])
    if xgrid and ygrid:
        ax.grid(True, color=colors["grid"], linewidth=0.7, alpha=0.65)
    elif xgrid:
        ax.grid(True, axis="x", color=colors["grid"], linewidth=0.7, alpha=0.65)
    elif ygrid:
        ax.grid(True, axis="y", color=colors["grid"], linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def social_footer(fig, text: str, *, y: float = 0.057, width: int = 142) -> None:
    colors = social_palette()
    fig.text(0.07, y, textwrap.fill(text, width=width), fontsize=8.15, color=colors["muted"], va="top", linespacing=1.18)


def draw_social_conus_base(ax) -> None:
    colors = social_palette()
    ax.set_facecolor(colors["water"])
    if CONUS_STATE_GEOJSON_PATH.exists():
        try:
            geojson = json.loads(CONUS_STATE_GEOJSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            geojson = {"features": []}
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not isinstance(geometry, dict):
                continue
            for ring in iter_geometry_rings(geometry):
                xs = [float(point[0]) for point in ring]
                ys = [float(point[1]) for point in ring]
                if max(xs) < -126 or min(xs) > -66 or max(ys) < 24 or min(ys) > 50:
                    continue
                ax.fill(xs, ys, facecolor=colors["land"], edgecolor=colors["border"], linewidth=0.45, zorder=0)
                ax.plot(xs, ys, color="#52708c", linewidth=0.28, alpha=0.9, zorder=1)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(colors["border"])


def finite_or_none(value: object) -> float | None:
    try:
        value_as_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_as_float if math.isfinite(value_as_float) else None


def rounded_or_none(value: object, digits: int = 2) -> float | None:
    finite_value = finite_or_none(value)
    return round(finite_value, digits) if finite_value is not None else None


def social_data_snapshot(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Collect every published social claim in one JSON-serializable object."""
    latest = latest_complete_igra_row(data["igra"])
    raw_latest = latest_igra_row(data["igra"])
    windows, partial_date = complete_current_windows(data["igra"], windows=(30, 60, 90, 180))
    cumulative = cumulative_deficit_since_breakpoint(data["igra"])
    statuses, impacted, cycle_text = latest_station_status_rows(data)
    latest_nco = latest_nco_row(data["nco"])

    archive_date = latest["date"].date() if latest is not None else None
    nco_date = pd.Timestamp(latest_nco["cycle_dt"]).date() if latest_nco is not None else None
    nco_lag_days = (archive_date - nco_date).days if archive_date is not None and nco_date is not None else None
    window_rows = []
    for _, row in windows.iterrows():
        window_rows.append(
            {
                "days": int(row["days"]),
                "observed": rounded_or_none(row["observed"]),
                "expected": rounded_or_none(row["expected"]),
                "deficit": rounded_or_none(row["deficit"]),
                "percent_difference": rounded_or_none(row["percent"]),
            }
        )
    return {
        "series": "CONUS upper-air social data watch",
        "archive": {
            "source": "NOAA/NCEI IGRA v2",
            "latest_complete_date": archive_date.isoformat() if archive_date is not None else None,
            "excluded_incomplete_date": partial_date.date().isoformat() if partial_date is not None else None,
            "latest_raw_date": raw_latest["date"].date().isoformat() if raw_latest is not None else None,
            "baseline": "2021-2024 same-date average",
            "seven_day_observed_per_day": rounded_or_none(latest["launches_7d_avg"]) if latest is not None else None,
            "seven_day_expected_per_day": rounded_or_none(latest["baseline_5yr_avg"]) if latest is not None else None,
            "seven_day_percent_difference": rounded_or_none(latest["percent_vs_baseline"]) if latest is not None else None,
            "windows": window_rows,
        },
        "cumulative_shortfall": {
            "breakpoint_date": cumulative["breakpoint_date"].date().isoformat() if cumulative["breakpoint_date"] is not None else None,
            "fewer_archived_soundings": rounded_or_none(cumulative["net_deficit"]),
            "percent_difference": rounded_or_none(cumulative["percent_difference"]),
        },
        "nco_snapshot": {
            "source": "NWS/NCEP/NCO SDM Administrative Messages",
            "cycle": cycle_text,
            "conus_raobs_available_for_ingest": int(latest_nco["conus_count"]) if latest_nco is not None else None,
            "impacted_active_conus_stations": int(len(impacted)),
            "mapped_active_conus_stations": int(len(statuses)),
            "days_older_than_latest_complete_igra": nco_lag_days,
            "interpretation": "A separate operational-message snapshot; do not combine it with IGRA archive counts or treat it as a causal explanation.",
        },
        "publishing_note": "Counts reflect available archive or operational records and may differ from actual launches because of ingest, archive, or reporting delays. Not a causation claim.",
    }


def write_social_data_snapshot(data: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(social_data_snapshot(data), indent=2) + "\n", encoding="utf-8")


def create_social_hero_decline(data: dict[str, pd.DataFrame], path: Path) -> None:
    colors = social_palette()
    latest = latest_complete_igra_row(data["igra"])
    plot_df, excluded_partial, _reason = plotted_igra_current(data["igra"])
    windows, _partial_date = complete_current_windows(data["igra"], windows=(90,))
    metrics = cumulative_deficit_since_breakpoint(data["igra"])

    pct = float(latest["percent_vs_baseline"]) if latest is not None else float("nan")
    latest_date = latest["date"].date().isoformat() if latest is not None else "latest complete date unavailable"
    avg = float(latest["launches_7d_avg"]) if latest is not None else float("nan")
    baseline = float(latest["baseline_5yr_avg"]) if latest is not None else float("nan")
    window_90 = windows.iloc[0] if not windows.empty else None

    fig = social_fig()
    social_eyebrow(fig, "Upper-air data watch  /  archive counts")
    fig.text(0.07, 0.905, "THE WEATHER-BALLOON\nDATA GAP IS STILL GROWING", fontsize=31.5, fontweight="bold", color=colors["ink"], va="top", linespacing=0.96)
    fig.text(0.07, 0.828, "The latest complete U.S. archive data remain below the seasonal norm.", fontsize=13.0, color=colors["muted"], va="top")
    social_card(fig, 0.07, 0.62, 0.86, 0.157)
    fig.text(0.10, 0.747, f"{pct:.1f}%", fontsize=58, fontweight="bold", color=colors["red"], va="top")
    fig.text(0.43, 0.738, "7-day archive average\nvs 2021-2024 same-date baseline", fontsize=14.2, fontweight="bold", color=colors["ink"], va="top", linespacing=1.16)
    fig.text(0.10, 0.644, f"AS OF {latest_date}  •  {avg:.1f}/DAY OBSERVED  •  {baseline:.1f}/DAY EXPECTED", fontsize=9.4, color=colors["muted"], va="top")

    social_card(fig, 0.07, 0.505, 0.415, 0.088, alt=True)
    social_card(fig, 0.515, 0.505, 0.415, 0.088, alt=True)
    deficit_90 = abs(float(window_90["deficit"])) if window_90 is not None else float("nan")
    fig.text(0.095, 0.57, f"{deficit_90:,.0f}" if math.isfinite(deficit_90) else "—", fontsize=25, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.095, 0.542, "FEWER ARCHIVED SOUNDINGS\nIN THE LAST 90 DAYS", fontsize=8.8, fontweight="bold", color=colors["muted"], va="top", linespacing=1.1)
    cumulative_total = float(metrics["net_deficit"])
    fig.text(0.54, 0.57, f"{cumulative_total:,.0f}" if math.isfinite(cumulative_total) else "—", fontsize=25, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.54, 0.542, "CUMULATIVE SHORTFALL\nSINCE APRIL 2025", fontsize=8.8, fontweight="bold", color=colors["muted"], va="top", linespacing=1.1)

    ax = fig.add_axes([0.07, 0.20, 0.86, 0.257])
    style_social_axis(ax)
    if not plot_df.empty:
        ax.plot(plot_df["date"], plot_df["baseline_5yr_avg"], color=colors["muted"], linewidth=2.0, linestyle="--", label="Expected baseline")
        ax.plot(plot_df["date"], plot_df["launches_7d_avg"], color=colors["blue"], linewidth=3.0, label="Observed 7-day average")
        below = plot_df["launches_7d_avg"] < plot_df["baseline_5yr_avg"]
        ax.fill_between(plot_df["date"], plot_df["launches_7d_avg"], plot_df["baseline_5yr_avg"], where=below, color=colors["red_soft"], alpha=0.72, interpolate=True)
        ax.set_xlim(plot_df["date"].min() - pd.Timedelta(days=5), plot_df["date"].max() + pd.Timedelta(days=10))
        ax.set_ylabel("soundings / day", color=colors["muted"], fontsize=9.3)
        ax.legend(loc="lower left", frameon=False, fontsize=8.8, labelcolor=colors["muted"], ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.text(0.07, 0.473, "2026 DAILY ARCHIVE TREND  •  SHADED AREA = GAP BELOW EXPECTED", fontsize=8.9, fontweight="bold", color=colors["muted"], va="bottom")
    footer = "Source: NOAA/NCEI IGRA v2. Counts are archived records, not a count of confirmed launches. They may reflect ingest, archive, or reporting delays; this is not a causation claim."
    if excluded_partial:
        footer += " Latest incomplete day excluded."
    social_footer(fig, footer)
    save_social_series(fig, path)


def create_social_observed_expected_windows(data: dict[str, pd.DataFrame], path: Path) -> None:
    colors = social_palette()
    windows, partial_date = complete_current_windows(data["igra"], windows=(30, 60, 90, 180))
    fig = social_fig()
    social_eyebrow(fig, "Archive trend check  /  four time windows")
    fig.text(0.07, 0.905, "EVERY RECENT WINDOW\nIS BELOW EXPECTED", fontsize=32, fontweight="bold", color=colors["ink"], va="top", linespacing=0.96)
    fig.text(0.07, 0.828, "The gap holds whether the comparison looks back one month or six.", fontsize=13.0, color=colors["muted"], va="top")

    social_card(fig, 0.07, 0.655, 0.86, 0.115)
    latest_text = "date unavailable"
    if not windows.empty:
        latest_text = windows["latest"].iloc[0].date().isoformat()
        strongest = windows.loc[windows["percent"].idxmin()]
        fig.text(0.10, 0.74, f"{float(strongest['percent']):.1f}%", fontsize=35, fontweight="bold", color=colors["red"], va="top")
        fig.text(0.37, 0.736, f"largest recent gap\n({int(strongest['days'])}-day window)", fontsize=13.0, fontweight="bold", color=colors["ink"], va="top", linespacing=1.15)
        fig.text(0.10, 0.675, f"WINDOWS ENDING {latest_text}  •  BASELINE = 2021-2024 SAME DATE", fontsize=8.9, color=colors["muted"], va="top")

    ax = fig.add_axes([0.11, 0.245, 0.78, 0.34])
    style_social_axis(ax, xgrid=True, ygrid=False)
    if not windows.empty:
        chart_rows = windows.sort_values("days").reset_index(drop=True)
        y = np.arange(len(chart_rows))
        values = chart_rows["percent"].to_numpy(dtype=float)
        ax.barh(y, values, height=0.55, color=colors["red"], alpha=0.94)
        ax.axvline(0, color=colors["ink"], linewidth=1.2, alpha=0.85)
        left_limit = min(-1.0, float(np.nanmin(values)) * 1.65)
        ax.set_xlim(left_limit, 1.0)
        for index, row in chart_rows.iterrows():
            value = float(row["percent"])
            ax.text(value - 0.08, index, f"{value:.1f}%", ha="right", va="center", fontsize=15, fontweight="bold", color=colors["ink"])
            ax.text(value - 0.08, index + 0.26, f"{abs(float(row['deficit'])):,.0f} fewer", ha="right", va="center", fontsize=8.6, color=colors["muted"])
        ax.set_yticks(y, [f"{int(days)} DAYS" for days in chart_rows["days"]], fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("percent difference from expected archive total", color=colors["muted"], fontsize=9.3)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    fig.text(0.07, 0.605, "PERCENT BELOW THE SAME-DATE BASELINE", fontsize=8.9, fontweight="bold", color=colors["muted"], va="bottom")
    footer = "Source: NOAA/NCEI IGRA v2. Each window compares archive counts with the 2021-2024 same-date average. Consistent shortfalls describe the archive data; they do not identify a cause."
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    social_footer(fig, footer)
    save_social_series(fig, path)


def create_social_cumulative_gap(data: dict[str, pd.DataFrame], path: Path) -> None:
    colors = social_palette()
    metrics = cumulative_deficit_since_breakpoint(data["igra"])
    post = metrics["cumulative"]
    breakpoint_date = metrics["breakpoint_date"]
    latest_complete = metrics["latest_complete_date"]
    partial_date = metrics["partial_date"]
    net_deficit = float(metrics["net_deficit"])
    percent_difference = float(metrics["percent_difference"])

    fig = social_fig()
    social_eyebrow(fig, "Archive trend check  /  cumulative shortfall")
    fig.text(0.07, 0.905, "THIS ISN'T ONE\nBAD WEEK OF DATA", fontsize=32, fontweight="bold", color=colors["ink"], va="top", linespacing=0.96)
    fig.text(0.07, 0.828, "The difference from the expected archive total has accumulated since spring 2025.", fontsize=13.0, color=colors["muted"], va="top")
    if not post.empty and breakpoint_date is not None and latest_complete is not None:
        social_card(fig, 0.07, 0.64, 0.86, 0.125)
        fig.text(0.10, 0.746, f"{net_deficit:,.0f}", fontsize=43, fontweight="bold", color=colors["red"], va="top")
        fig.text(0.47, 0.74, "fewer archived soundings\nsince the detected breakpoint", fontsize=13.2, fontweight="bold", color=colors["ink"], va="top", linespacing=1.15)
        fig.text(0.10, 0.67, f"{percent_difference:+.1f}% VS EXPECTED  •  APR 3, 2025 TO {latest_complete.date().isoformat()}", fontsize=8.9, color=colors["muted"], va="top")

    ax = fig.add_axes([0.07, 0.225, 0.86, 0.35])
    style_social_axis(ax)
    if not post.empty:
        ax.plot(post["date"], post["cumulative_net_deficit"], color=colors["red"], linewidth=3.7)
        ax.fill_between(post["date"], post["cumulative_net_deficit"], 0, color=colors["red_soft"], alpha=0.72)
        ax.axhline(0, color=colors["muted"], linewidth=1.0, alpha=0.7)
        if breakpoint_date is not None:
            ax.axvline(breakpoint_date, color=colors["amber"], linewidth=1.3, linestyle="--")
            ax.text(breakpoint_date + pd.Timedelta(days=8), max(80, float(post["cumulative_net_deficit"].max()) * 0.08), "Apr 3\nbreakpoint", fontsize=8.6, color=colors["amber"], fontweight="bold", va="bottom")
        latest = post.iloc[-1]
        ax.scatter([latest["date"]], [latest["cumulative_net_deficit"]], color=colors["amber"], s=48, zorder=4)
        ax.text(latest["date"] - pd.Timedelta(days=8), latest["cumulative_net_deficit"], f"{latest['cumulative_net_deficit']:,.0f}", ha="right", va="bottom", fontsize=10.5, fontweight="bold", color=colors["ink"])
        ax.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), post["date"].max() + pd.Timedelta(days=22))
        ax.set_ylim(0, max(3200, float(post["cumulative_net_deficit"].max()) * 1.13))
    ax.set_ylabel("cumulative fewer archived soundings", color=colors["muted"], fontsize=9.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.text(0.07, 0.595, "RUNNING DIFFERENCE FROM THE 2021-2024 SAME-DATE BASELINE", fontsize=8.9, fontweight="bold", color=colors["muted"], va="bottom")
    footer = "Source: NOAA/NCEI IGRA v2. The running gap is archive count minus the 2021-2024 same-date average, accumulated from a data-driven breakpoint. It is not a causation claim."
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    social_footer(fig, footer)
    save_social_series(fig, path)


def create_social_station_impacts(data: dict[str, pd.DataFrame], path: Path) -> None:
    colors = social_palette()
    statuses, impacted, cycle_text = latest_station_status_rows(data)
    latest = latest_nco_row(data["nco"])
    fig = social_fig()
    social_eyebrow(fig, "Operational snapshot  /  NCO message layer")
    fig.text(0.07, 0.905, "A NETWORK SNAPSHOT,\nNOT THE WHOLE EXPLANATION", fontsize=30.5, fontweight="bold", color=colors["ink"], va="top", linespacing=0.96)
    fig.text(0.07, 0.828, "Reported station status is useful context—but it is a separate data stream from the archive trend.", fontsize=12.8, color=colors["muted"], va="top")
    social_card(fig, 0.07, 0.645, 0.86, 0.112)
    fig.text(0.10, 0.735, f"{len(impacted)}", fontsize=42, color=colors["red"], fontweight="bold", va="top")
    fig.text(0.27, 0.727, "active CONUS stations\nwith a reported issue status", fontsize=13.3, color=colors["ink"], fontweight="bold", va="top", linespacing=1.15)
    cycle_caption = cycle_text.upper() if cycle_text != "latest cycle unavailable" else "CYCLE UNAVAILABLE"
    fig.text(0.10, 0.672, f"NCO MESSAGE SNAPSHOT: {cycle_caption}", fontsize=8.9, color=colors["muted"], va="top")

    ax = fig.add_axes([0.07, 0.23, 0.86, 0.355])
    draw_social_conus_base(ax)
    if not statuses.empty:
        available = statuses[statuses["status"].isin(["available/no issue", "unknown"])]
        ax.scatter(available["longitude"], available["latitude"], s=28, c=colors["green"], edgecolor=colors["bg"], linewidth=0.5, alpha=0.72, zorder=3)
    if not impacted.empty:
        color_map = {"missing/problem": colors["red"], "partial/quality issue": colors["amber"], "issue": colors["amber"]}
        impacted = impacted.copy()
        impacted["plot_color"] = impacted["status"].map(color_map).fillna(colors["amber"])
        ax.scatter(impacted["longitude"], impacted["latitude"], s=92, c=impacted["plot_color"], edgecolor=colors["ink"], linewidth=0.7, alpha=0.98, zorder=4)
        label_rows = impacted.sort_values(["status", "station_id"]).head(18)
        for _, row in label_rows.iterrows():
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            ha = "right" if lon > -78 else "left"
            dx = -0.55 if ha == "right" else 0.55
            ax.text(
                lon + dx,
                lat + 0.18,
                str(row["station_id"]),
                color=colors["ink"],
                fontsize=7.9,
                fontweight="bold",
                ha=ha,
                va="center",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.16", "fc": colors["panel"], "ec": colors["border"], "alpha": 0.94},
            )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["red"], markeredgecolor=colors["ink"], markersize=8, label="missing / problem"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["amber"], markeredgecolor=colors["ink"], markersize=8, label="partial / quality"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["green"], markeredgecolor=colors["bg"], markersize=8, label="available / no issue"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, facecolor=colors["panel"], edgecolor=colors["border"], labelcolor=colors["muted"], fontsize=8.2)
    nco_note = "cycle unavailable"
    if latest is not None:
        nco_note = f"{latest['cycle_date_utc']} {str(latest['cycle_hour']).zfill(2)}Z {latest['model']}"
    fig.text(0.07, 0.603, "MAP READ: markers are reported status labels in one NCO message cycle—not a measure of the archive deficit.", fontsize=8.8, fontweight="bold", color=colors["muted"], va="bottom")
    social_footer(fig, f"Source: NWS/NCEP/NCO SDM Administrative Messages and the station master. Snapshot from {nco_note}; it may lag the IGRA archive series. Reported status does not establish why archive counts differ from baseline.")
    save_social_series(fig, path)


def create_social_two_layer_evidence(data: dict[str, pd.DataFrame], path: Path) -> None:
    colors = social_palette()
    metrics = decline_start_metrics(data["igra"])
    series = metrics["series"]
    nco = data["nco"]
    latest = latest_complete_igra_row(data["igra"])
    latest_nco = latest_nco_row(nco)
    pct = float(latest["percent_vs_baseline"]) if latest is not None else float("nan")
    latest_date = latest["date"].date().isoformat() if latest is not None else "unavailable"
    nco_count = int(latest_nco["conus_count"]) if latest_nco is not None else None

    nco_cycle = "unavailable"
    nco_lag_note = ""
    if latest_nco is not None:
        nco_cycle = f"{latest_nco['cycle_date_utc']} {str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco['model']}"
        if latest is not None:
            lag_days = (latest["date"].date() - pd.Timestamp(latest_nco["cycle_dt"]).date()).days
            if lag_days > 0:
                nco_lag_note = f" • {lag_days} DAYS OLDER THAN THE LATEST COMPLETE IGRA DATE"

    fig = social_fig()
    social_eyebrow(fig, "Data literacy  /  archive and operations")
    fig.text(0.07, 0.905, "TWO DATA STREAMS.\nDON'T CONFLATE THEM.", fontsize=32, fontweight="bold", color=colors["ink"], va="top", linespacing=0.96)
    fig.text(0.07, 0.828, "Both are useful. Neither can be used as a shortcut to explain the other.", fontsize=13.0, color=colors["muted"], va="top")

    social_card(fig, 0.07, 0.655, 0.86, 0.117)
    fig.text(0.10, 0.743, "IGRA ARCHIVE", fontsize=10.3, fontweight="bold", color=colors["blue"], va="top")
    fig.text(0.10, 0.712, f"{pct:.1f}%", fontsize=29, fontweight="bold", color=colors["red"], va="top")
    fig.text(0.31, 0.708, f"7-day count vs baseline\nlatest complete: {latest_date}", fontsize=10.5, color=colors["ink"], va="top", linespacing=1.15)
    fig.text(0.61, 0.743, "NCO MESSAGES", fontsize=10.3, fontweight="bold", color=colors["amber"], va="top")
    nco_text = f"{nco_count}" if nco_count is not None else "—"
    fig.text(0.61, 0.712, nco_text, fontsize=29, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.73, 0.708, "CONUS RAOBs\nfor ingest in this cycle", fontsize=10.2, color=colors["ink"], va="top", linespacing=1.15)
    fig.text(0.10, 0.672, f"SEPARATE DEFINITIONS • SEPARATE UPDATE CLOCKS{nco_lag_note}", fontsize=8.05, color=colors["muted"], va="top")

    ax1 = fig.add_axes([0.07, 0.43, 0.86, 0.16])
    style_social_axis(ax1)
    if not series.empty:
        ax1.axhline(0, color=colors["muted"], linewidth=1.0, alpha=0.72)
        ax1.plot(series["date"], series["rolling_percent_difference"], color=colors["red"], linewidth=2.8)
        ax1.fill_between(series["date"], series["rolling_percent_difference"], 0, where=series["rolling_percent_difference"] < 0, color=colors["red_soft"], alpha=0.68)
        ax1.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), series["date"].max() + pd.Timedelta(days=15))
    ax1.set_ylabel("IGRA vs base (%)", color=colors["muted"], fontsize=8.8)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.text(0.07, 0.607, "1  /  ARCHIVE COUNTS: A CONTINUOUS SERIES COMPARED WITH ITS SEASONAL BASELINE", fontsize=8.8, fontweight="bold", color=colors["muted"], va="bottom")

    ax2 = fig.add_axes([0.07, 0.195, 0.86, 0.15])
    style_social_axis(ax2)
    if not nco.empty:
        recent = nco.dropna(subset=["cycle_dt", "conus_count"]).sort_values("cycle_dt")
        if not recent.empty:
            recent = recent[recent["cycle_dt"] >= recent["cycle_dt"].max() - pd.Timedelta(days=21)]
        for model, group in recent.groupby("model"):
            ax2.plot(group["cycle_dt"], group["conus_count"], marker="o", markersize=5.0, linestyle="none", color=colors["amber"] if model == "NAM" else colors["blue"], label=model)
        ax2.legend(loc="lower left", frameon=False, fontsize=8.3, labelcolor=colors["muted"], ncol=2)
    ax2.set_ylabel("NCO CONUS RAOBs", color=colors["muted"], fontsize=8.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    fig.text(0.07, 0.36, f"2  /  NCO MESSAGES: DISCRETE MODEL-INGEST RECAPS • SNAPSHOT {nco_cycle.upper()}", fontsize=8.8, fontweight="bold", color=colors["muted"], va="bottom")
    social_footer(fig, "Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages. IGRA archive counts and NCO ingest recaps use separate definitions and update schedules. Do not add the values together or treat one as proof of what caused the other.")
    save_social_series(fig, path)


def write_social_editorial_manifest(data: dict[str, pd.DataFrame], path: Path) -> None:
    snapshot = social_data_snapshot(data)
    archive = snapshot["archive"]
    cumulative = snapshot["cumulative_shortfall"]
    windows = {row["days"]: row for row in archive["windows"]}
    latest_date = archive["latest_complete_date"] or "unavailable"
    pct = archive["seven_day_percent_difference"]
    pct_text = f"{pct:.1f}%" if pct is not None else "unavailable"
    window_90 = windows.get(90)
    window_text = ""
    if window_90 is not None and window_90["deficit"] is not None and window_90["percent_difference"] is not None:
        window_text = f"The past 90 days contained {abs(window_90['deficit']):,.0f} fewer archived soundings ({window_90['percent_difference']:.1f}%)."
    cumulative_text = ""
    if cumulative["fewer_archived_soundings"] is not None and cumulative["breakpoint_date"] is not None:
        cumulative_text = f"Since {cumulative['breakpoint_date']}: {cumulative['fewer_archived_soundings']:,.0f} fewer archived soundings."
    files = [
        ("social_hero_decline.png", "Lead post. One-screen summary of the archive gap, 90-day shortfall, and cumulative context."),
        ("social_observed_expected_windows.png", "Evidence slide. Percent comparison keeps the shared signal legible across 30/60/90/180-day windows."),
        ("social_cumulative_gap.png", "Trend slide. Shows that the shortfall has accumulated since the data-driven breakpoint."),
        ("social_station_impacts.png", "Operational context. A clearly dated NCO station-status snapshot, not an explanation of the archive trend."),
        ("social_two_layer_evidence.png", "Method slide. Prevents false equivalence between IGRA archive data and NCO model-ingest messages."),
    ]
    caption = (
        f"The latest complete U.S. upper-air archive data are running {pct_text} versus the 2021-2024 same-date baseline as of {latest_date}. "
        f"{window_text} {cumulative_text} The charts show an archive signal, not a confirmed count of launches or a causal explanation."
    )
    if archive["excluded_incomplete_date"] is not None:
        caption += f" Latest incomplete date excluded: {archive['excluded_incomplete_date']}."
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Upper-air social publishing pack\n")
        handle.write("==============================\n\n")
        handle.write(f"Archive data through: {latest_date}\n")
        handle.write(f"Archive baseline: {archive['baseline']}\n")
        handle.write(f"NCO snapshot: {snapshot['nco_snapshot']['cycle']}\n\n")
        handle.write("Recommended sequence\n")
        handle.write("--------------------\n")
        for filename, purpose in files:
            handle.write(f"- {filename}: {purpose}\n")
        handle.write("\nSuggested lead caption\n")
        handle.write("----------------------\n")
        handle.write(textwrap.fill(caption, width=92))
        handle.write("\n\nPublishing guardrails\n")
        handle.write("---------------------\n")
        handle.write("- Keep the source/caveat footer visible when cropping or reposting.\n")
        handle.write("- Describe the IGRA trend as archive counts; do not call it a confirmed launch count.\n")
        handle.write("- Keep the NCO snapshot in its own slide/caption and do not present it as causal proof.\n")


def write_social_caption_pack(data: dict[str, pd.DataFrame], path: Path) -> None:
    """Write ready-to-post copy and alt text from the exact published data snapshot."""
    snapshot = social_data_snapshot(data)
    archive = snapshot["archive"]
    cumulative = snapshot["cumulative_shortfall"]
    nco = snapshot["nco_snapshot"]
    pct = archive["seven_day_percent_difference"]
    pct_text = f"{pct:.1f}%" if pct is not None else "unavailable"
    window_90 = next((row for row in archive["windows"] if row["days"] == 90), None)
    shortfall_90 = "unavailable"
    if window_90 is not None and window_90["deficit"] is not None:
        shortfall_90 = f"{abs(window_90['deficit']):,.0f}"
    cumulative_total = cumulative["fewer_archived_soundings"]
    cumulative_text = f"{cumulative_total:,.0f}" if cumulative_total is not None else "unavailable"
    archive_date = archive["latest_complete_date"] or "unavailable"
    breakpoint_date = cumulative["breakpoint_date"] or "the detected breakpoint"
    nco_count = nco["conus_raobs_available_for_ingest"]
    nco_count_text = str(nco_count) if nco_count is not None else "unavailable"
    nco_lag = nco["days_older_than_latest_complete_igra"]
    nco_lag_text = f" ({nco_lag} days older than the archive series)" if nco_lag and nco_lag > 0 else ""

    copy = f"""# Upper-air social copy\n\nData basis: IGRA archive through {archive_date}; baseline is the 2021-2024 same-date average. The latest incomplete archive date ({archive['excluded_incomplete_date'] or 'none'}) is excluded.\n\n## 1. Lead post — social_hero_decline.png\n\nThe latest complete U.S. upper-air archive data are {pct_text} versus the seasonal baseline. In the past 90 days, the archive contained {shortfall_90} fewer soundings; since {breakpoint_date}, the cumulative gap is {cumulative_text}.\n\nThat is an archive-count signal, not a confirmed launch count or proof of why the difference exists.\n\nAlt text: A dark-blue graphic states that the weather-balloon data gap is still growing. It reports a {pct_text} seven-day archive difference from the 2021-2024 same-date baseline, plus the 90-day and cumulative shortfalls. A line chart shows observed 2026 archive counts below the expected baseline.\n\n## 2. Evidence slide — social_observed_expected_windows.png\n\nThis isn't a one-window result: 30-, 60-, 90-, and 180-day archive comparisons are all below their same-date baselines.\n\nAlt text: A horizontal bar chart shows negative percentage differences for four recent archive windows, each below the expected baseline.\n\n## 3. Trend slide — social_cumulative_gap.png\n\nThe relevant question isn't only whether a single week is down. The running archive shortfall has grown to {cumulative_text} soundings since {breakpoint_date}.\n\nAlt text: A line and shaded-area chart shows the cumulative archive shortfall rising from zero after the April 2025 breakpoint to {cumulative_text}.\n\n## 4. Operational context — social_station_impacts.png\n\nThis map is a dated operations snapshot: {nco['impacted_active_conus_stations']} active CONUS stations carry a reported issue status in NCO's {nco['cycle']} message. It is context, not a causal explanation for the archive trend.\n\nAlt text: A map of the continental United States marks reported station issue statuses in red or amber and available stations in green.\n\n## 5. Method slide — social_two_layer_evidence.png\n\nIGRA archive counts and NCO model-ingest recaps are different data streams. The NCO snapshot reports {nco_count_text} CONUS RAOBs available for ingest{nco_lag_text}; it should not be added to, or used to explain, the archive-count series.\n\nAlt text: A two-panel explainer distinguishes the longer IGRA archive trend from the shorter NCO model-ingest message series and states that the values must not be conflated.\n\n## Publishing guardrails\n\n- Preserve the source and caveat footer in every repost/crop.\n- Do not attribute the archive signal to a specific action, policy, station issue, or reporting gap without direct evidence.\n- Refresh the NCO slides before reuse; their update cadence differs from the archive series.\n"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(copy, encoding="utf-8", newline="\n")


def create_timeline_observed_decline_context(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = decline_start_metrics(data["igra"])
    series = metrics["series"]
    breakpoint_date = metrics["breakpoint_date"]
    partial_date = metrics["partial_date"]

    bg = "#0b1220"
    panel = "#111827"
    panel2 = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    blue = "#60a5fa"
    red = "#fb7185"
    amber = "#fcd34d"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.95, "Upper-Air Decline in Context", fontsize=31, fontweight="bold", color=text, va="top")
    fig.text(0.07, 0.91, "30-day archived CONUS IGRA soundings vs 2021-2024 same-date baseline", fontsize=12.8, color=muted, va="top")
    if breakpoint_date is not None:
        fig.text(0.07, 0.845, f"Data-driven breakpoint: {breakpoint_date.strftime('%b')} {breakpoint_date.day}, {breakpoint_date.year}", fontsize=26, fontweight="bold", color=red, va="top")

    ax = fig.add_axes([0.08, 0.36, 0.85, 0.39], facecolor=panel)
    if not series.empty:
        ax.axhline(0, color="#e2e8f0", linewidth=1.2, alpha=0.92)
        ax.text(series["date"].min() + pd.Timedelta(days=12), 0.22, "Expected level", fontsize=9.5, color="#e2e8f0", fontweight="bold")
        ax.plot(series["date"], series["rolling_percent_difference"], color=blue, linewidth=3.2)
        if breakpoint_date is not None:
            shaded = series[series["date"] >= breakpoint_date]
            ax.fill_between(shaded["date"], shaded["rolling_percent_difference"], 0, where=shaded["rolling_percent_difference"] < 0, color="#7f1d1d", alpha=0.34)
            ax.axvline(breakpoint_date, color=red, linewidth=1.2, linestyle="--", alpha=0.95)
        y_top = 1.25
        for index, (event_date, label) in enumerate(context_events()):
            if series["date"].min() <= event_date <= series["date"].max():
                ax.axvline(event_date, color="#94a3b8", linewidth=0.85, linestyle=":", alpha=0.48)
                if index in {0, 1, 4}:
                    ax.text(event_date, y_top - 0.27 * (index % 2), event_date.strftime("%b %d"), fontsize=8.3, color=amber, ha="center", va="bottom", fontweight="bold")
        latest = series.iloc[-1]
        ax.text(latest["date"] + pd.Timedelta(days=13), latest["rolling_percent_difference"], "30-day observed\nvs baseline", color=blue, fontsize=10.6, fontweight="bold", va="center")
        ax.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), series["date"].max() + pd.Timedelta(days=58))
        ax.set_ylim(min(-8.8, float(series["rolling_percent_difference"].min()) - 0.7), 1.55)
    ax.set_ylabel("Difference from 2021-2024 average (%)", color=muted, fontsize=9.8)
    ax.tick_params(colors=muted, labelsize=10)
    ax.grid(True, axis="y", color=border, linewidth=0.65, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    card = fig.add_axes([0.07, 0.195, 0.86, 0.105], facecolor=panel2)
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_color(border)
    card.text(0.03, 0.76, "DOGE/NOAA/NWS context markers", fontsize=11.5, fontweight="bold", color=muted, va="top")
    for (event_date, label), x in zip(context_events(), [0.03, 0.22, 0.42, 0.62, 0.81]):
        card.text(x, 0.49, f"{event_date.strftime('%b')} {event_date.day}", fontsize=12.0, color=amber, fontweight="bold", va="top")
        card.text(x, 0.25, textwrap.fill(label, width=16), fontsize=8.5, color=text, va="top", linespacing=1.12)

    footer = "Sources: NOAA/NCEI IGRA v2; NWS notices and public reporting for context markers. Not a causation claim."
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    dark_note_footer(fig, footer)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_observed_expected_counter(data: dict[str, pd.DataFrame], path: Path) -> None:
    windows, partial_date = complete_current_windows(data["igra"])
    bg = "#0b1220"
    panel = "#111827"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    blue = "#60a5fa"
    amber = "#fcd34d"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.95, "Observed vs Expected Soundings", fontsize=31, fontweight="bold", color=text, va="top")
    fig.text(0.07, 0.91, "Archived CONUS IGRA totals through the latest complete date", fontsize=12.8, color=muted, va="top")

    ax = fig.add_axes([0.09, 0.27, 0.84, 0.55], facecolor=panel)
    if not windows.empty:
        y = np.arange(len(windows))
        ax.barh(y + 0.18, windows["expected"], height=0.32, color="#334155", edgecolor="#64748b", label="Expected baseline")
        ax.barh(y - 0.18, windows["observed"], height=0.32, color=blue, edgecolor=blue, label="Observed")
        for index, row in windows.iterrows():
            x = max(row["observed"], row["expected"]) * 1.015
            ax.text(x, index, f"{row['deficit']:.0f} ({row['percent']:.1f}%)", color=red, fontsize=12, fontweight="bold", va="center")
        ax.set_yticks(y, windows["window"])
        ax.invert_yaxis()
        ax.set_xlim(0, float(windows[["observed", "expected"]].max().max()) * 1.24)
        ax.set_xlabel("Archived soundings", color=muted)
        ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=muted)
        latest = windows["latest"].iloc[0]
        fig.text(0.07, 0.845, f"Latest complete date: {latest.date().isoformat()}", fontsize=21, color=amber, fontweight="bold", va="top")
    ax.tick_params(colors=muted, labelsize=11)
    ax.grid(True, axis="x", color=border, linewidth=0.65, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_color(border)
    footer = "Source: NOAA/NCEI IGRA v2. Expected equals prior 5-year same-date baseline. Not a causation claim."
    if partial_date is not None:
        footer += f" Latest incomplete date excluded: {partial_date.date().isoformat()}."
    dark_note_footer(fig, footer)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def iter_geometry_rings(geometry: dict[str, object]) -> list[list[list[float]]]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        return [ring for polygon in [coordinates] for ring in polygon if ring]
    if geom_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon if ring]
    return []


def draw_conus_base(ax, panel: str, border: str, muted: str) -> None:
    ax.set_facecolor(panel)
    if CONUS_STATE_GEOJSON_PATH.exists():
        try:
            geojson = json.loads(CONUS_STATE_GEOJSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            geojson = {"features": []}
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not isinstance(geometry, dict):
                continue
            for ring in iter_geometry_rings(geometry):
                xs = [float(point[0]) for point in ring]
                ys = [float(point[1]) for point in ring]
                if max(xs) < -126 or min(xs) > -66 or max(ys) < 24 or min(ys) > 50:
                    continue
                ax.fill(xs, ys, facecolor="#142033", edgecolor=border, linewidth=0.45, alpha=0.78, zorder=0)
                ax.plot(xs, ys, color="#64748b", linewidth=0.35, alpha=0.58, zorder=1)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", color=muted)
    ax.set_ylabel("Latitude", color=muted)
    ax.tick_params(colors=muted, labelsize=9)
    ax.grid(True, color=border, linewidth=0.45, alpha=0.28, zorder=-1)
    for spine in ax.spines.values():
        spine.set_color(border)


def active_conus_station_frame(stations: pd.DataFrame) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame()
    rows = stations.copy()
    if "active_expected" in rows:
        rows = rows[rows["active_expected"].astype(str).str.lower() == "true"].copy()
    if "country" in rows:
        rows = rows[rows["country"].astype(str).str.upper().isin({"US", ""})].copy()
    rows["latitude"] = pd.to_numeric(rows.get("latitude"), errors="coerce")
    rows["longitude"] = pd.to_numeric(rows.get("longitude"), errors="coerce")
    return rows.dropna(subset=["latitude", "longitude"]).copy()


def draw_latest_station_issue_map(
    ax,
    data: dict[str, pd.DataFrame],
    *,
    panel: str,
    border: str,
    text: str,
    muted: str,
    red: str,
    orange: str,
    blue: str,
) -> tuple[int, str]:
    draw_conus_base(ax, panel, border, muted)
    stations = active_conus_station_frame(data.get("stations", pd.DataFrame()))
    nco = data.get("nco", pd.DataFrame())
    issues = data.get("issues", pd.DataFrame())
    if stations.empty:
        ax.text(-96, 37, "Station metadata unavailable", color=muted, fontsize=13, ha="center", va="center")
        return 0, "latest cycle unavailable"

    status_rows = station_statuses(stations, issues, nco)
    status_rows = status_rows.dropna(subset=["latitude", "longitude"]).copy()
    available = status_rows[status_rows["status"].isin(["available/no issue", "unknown"])]
    impacted = status_rows[~status_rows["status"].isin(["available/no issue", "unknown"])].copy()

    ax.scatter(
        available["longitude"],
        available["latitude"],
        s=34,
        c=blue,
        edgecolor="#dbeafe",
        linewidth=0.45,
        alpha=0.58,
        zorder=3,
    )
    if not impacted.empty:
        color_map = {"missing/problem": red, "partial/quality issue": orange, "issue": orange}
        impacted["plot_color"] = impacted["status"].map(color_map).fillna(orange)
        ax.scatter(
            impacted["longitude"],
            impacted["latitude"],
            s=86,
            c=impacted["plot_color"],
            edgecolor="#f8fafc",
            linewidth=0.8,
            alpha=0.96,
            zorder=4,
        )
        label_rows = impacted.sort_values(["status", "station_id"]).head(18)
        for _, row in label_rows.iterrows():
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            ha = "right" if lon > -78 else "left"
            dx = -0.55 if ha == "right" else 0.55
            ax.text(
                lon + dx,
                lat + 0.18,
                str(row["station_id"]),
                color=text,
                fontsize=8.0,
                fontweight="bold",
                ha=ha,
                va="center",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.14", "fc": "#0b1220", "ec": "none", "alpha": 0.72},
            )

    latest = latest_cycle(nco)
    cycle_text = "latest cycle unavailable"
    if latest is not None:
        cycle_text = f"{latest['cycle_date_utc']} {str(latest['cycle_hour']).zfill(2)}Z {latest['model']}"
    return len(impacted), cycle_text


def create_station_deficit_map_context(data: dict[str, pd.DataFrame], path: Path) -> None:
    deficits = data.get("station_deficits", pd.DataFrame()).copy()
    bg = "#0b1220"
    panel = "#111827"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    orange = "#f59e0b"
    blue = "#60a5fa"
    gray = "#94a3b8"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    if deficits.empty:
        fig.text(0.07, 0.95, "Latest Upper-Air Station Issues", fontsize=31, fontweight="bold", color=text, va="top")
        fig.text(0.07, 0.91, "NCO issue-layer fallback map with station labels", fontsize=12.8, color=muted, va="top")
    else:
        fig.text(0.07, 0.95, "Where the Archive Shows Gaps", fontsize=31, fontweight="bold", color=text, va="top")
        fig.text(0.07, 0.91, "Station-level 90-day IGRA observed vs expected archive counts", fontsize=12.8, color=muted, va="top")
    ax = fig.add_axes([0.08, 0.25, 0.85, 0.57], facecolor=panel)
    draw_conus_base(ax, panel, border, muted)
    if not deficits.empty:
        deficits = deficits.dropna(subset=["latitude", "longitude", "percent_90"])
        colors = np.where(deficits["percent_90"] <= -50, red, np.where(deficits["percent_90"] <= -10, orange, np.where(deficits["percent_90"] < 0, blue, gray)))
        sizes = np.clip(np.abs(deficits["deficit_90"].fillna(0)) * 1.2 + 35, 35, 260)
        ax.scatter(deficits["longitude"], deficits["latitude"], c=colors, s=sizes, edgecolor="#e2e8f0", linewidth=0.7, alpha=0.94, zorder=4)
        top = deficits.sort_values("deficit_90").head(6)
        for _, row in top.iterrows():
            label = str(row["station_id"])[-5:]
            ax.text(row["longitude"] + 0.55, row["latitude"] + 0.18, label, color=text, fontsize=8.0, fontweight="bold", zorder=5)
        severe = int((deficits["percent_90"] <= -50).sum())
        moderate = int(((deficits["percent_90"] > -50) & (deficits["percent_90"] <= -10)).sum())
        fig.text(0.07, 0.845, f"{severe} stations at least 50% below expected over 90 days", fontsize=22, color=red, fontweight="bold", va="top")
        station_word = "station" if moderate == 1 else "stations"
        fig.text(0.07, 0.812, f"{moderate} more {station_word} 10-50% below expected", fontsize=12.5, color=muted, va="top")
    else:
        impacted_count, cycle_text = draw_latest_station_issue_map(
            ax,
            data,
            panel=panel,
            border=border,
            text=text,
            muted=muted,
            red=red,
            orange=orange,
            blue=blue,
        )
        fig.text(0.07, 0.845, f"{impacted_count} latest-cycle station issue markers", fontsize=22, color=red, fontweight="bold", va="top")
        fig.text(0.07, 0.812, f"NCO issue fallback shown because station-level IGRA deficit rows are unavailable. Cycle: {cycle_text}", fontsize=11.5, color=muted, va="top")
    if deficits.empty:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=red, markeredgecolor="#e2e8f0", markersize=9, label="missing/problem"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=orange, markeredgecolor="#e2e8f0", markersize=9, label="partial/quality issue"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=blue, markeredgecolor="#e2e8f0", markersize=9, label="available/no issue"),
        ]
    else:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=red, markeredgecolor="#e2e8f0", markersize=9, label="50%+ below"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=orange, markeredgecolor="#e2e8f0", markersize=9, label="10-50% below"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=blue, markeredgecolor="#e2e8f0", markersize=9, label="0-10% below"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=gray, markeredgecolor="#e2e8f0", markersize=9, label="near/above expected"),
        ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.2, labelcolor=muted)
    footer = (
        "Source: NWS/NCEP/NCO SDM Administrative Messages and station master. Fallback issue-layer map; not a causation claim."
        if deficits.empty
        else "Source: NOAA/NCEI IGRA v2 station archive counts. Simple latitude/longitude map; not a causation claim."
    )
    dark_note_footer(fig, footer, y=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_impacted_station_small_multiples(data: dict[str, pd.DataFrame], path: Path) -> None:
    deficits = data.get("station_deficits", pd.DataFrame()).copy()
    bg = "#0b1220"
    panel = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    blue = "#60a5fa"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    if deficits.empty:
        fig.text(0.07, 0.95, "Latest Impacted Stations Are Visible", fontsize=30, fontweight="bold", color=text, va="top")
        fig.text(0.07, 0.91, "NCO issue-layer fallback map with station labels", fontsize=12.8, color=muted, va="top")
    else:
        fig.text(0.07, 0.95, "Station-Level Reductions Are Visible", fontsize=30, fontweight="bold", color=text, va="top")
        fig.text(0.07, 0.91, "Largest 90-day archived-sounding shortfalls vs expected station baseline", fontsize=12.8, color=muted, va="top")
    if deficits.empty:
        ax = fig.add_axes([0.08, 0.24, 0.85, 0.58], facecolor=panel)
        impacted_count, cycle_text = draw_latest_station_issue_map(
            ax,
            data,
            panel="#111827",
            border=border,
            text=text,
            muted=muted,
            red=red,
            orange="#f59e0b",
            blue=blue,
        )
        fig.text(0.07, 0.845, f"{impacted_count} latest-cycle impacted stations labeled", fontsize=22, color=red, fontweight="bold", va="top")
        fig.text(0.07, 0.812, f"NCO issue fallback shown because station-level IGRA deficit rows are unavailable. Cycle: {cycle_text}", fontsize=11.5, color=muted, va="top")
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=red, markeredgecolor="#e2e8f0", markersize=9, label="missing/problem"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#f59e0b", markeredgecolor="#e2e8f0", markersize=9, label="partial/quality issue"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=blue, markeredgecolor="#e2e8f0", markersize=9, label="available/no issue"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.2, labelcolor=muted)
    else:
        top = deficits.dropna(subset=["deficit_90"]).sort_values("deficit_90").head(8).reset_index(drop=True)
        for index, row in top.iterrows():
            col = index % 2
            r = index // 2
            ax = fig.add_axes([0.07 + col * 0.46, 0.69 - r * 0.145, 0.40, 0.105], facecolor=panel)
            for spine in ax.spines.values():
                spine.set_color(border)
            ax.set_xticks([])
            ax.set_yticks([])
            expected = float(row["expected_90"])
            observed = float(row["observed_90"])
            pct = float(row["percent_90"])
            deficit = float(row["deficit_90"])
            label = str(row["station_id"])[-5:]
            name = str(row.get("name", "")).replace(";", ",").title()
            ax.text(0.04, 0.82, f"{label}  {name[:26]}", color=text, fontsize=10.2, fontweight="bold", va="top", transform=ax.transAxes)
            ax.text(0.04, 0.55, f"{abs(deficit):.0f} fewer over 90 days", color=red, fontsize=13.2, fontweight="bold", va="top", transform=ax.transAxes)
            ax.text(0.04, 0.28, f"Observed {observed:.0f} vs expected {expected:.0f} ({pct:.0f}%)", color=muted, fontsize=9.2, va="top", transform=ax.transAxes)
            ax.barh([0], [expected], color="#334155", height=0.18)
            ax.barh([0], [observed], color=blue, height=0.18)
            ax.set_xlim(0, max(expected, observed) * 1.08 if max(expected, observed) else 1)
            ax.set_ylim(-0.45, 0.45)
    footer = (
        "Source: NWS/NCEP/NCO SDM Administrative Messages and station master. Fallback issue-layer map; not a causation claim."
        if deficits.empty
        else "Source: NOAA/NCEI IGRA v2 station archive counts. Top stations are ranked by 90-day deficit; not a causation claim."
    )
    dark_note_footer(fig, footer, y=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_nco_ingest_context(data: dict[str, pd.DataFrame], path: Path) -> None:
    nco = data["nco"].copy()
    bg = "#0b1220"
    panel = "#111827"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    blue = "#60a5fa"
    amber = "#fcd34d"
    red = "#fb7185"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.95, "Model-Ingest Availability Is a Separate Signal", fontsize=28, fontweight="bold", color=text, va="top")
    fig.text(0.07, 0.91, "NCO SDM Administrative Messages: CONUS RAOBs available for NAM/GFS ingest", fontsize=12.4, color=muted, va="top")
    ax = fig.add_axes([0.08, 0.32, 0.85, 0.45], facecolor=panel)
    if not nco.empty:
        recent = nco.dropna(subset=["cycle_dt", "conus_count"]).sort_values(["cycle_dt", "model"]).copy()
        for model, group in recent.groupby("model"):
            color = blue if model == "GFS" else amber
            ax.plot(group["cycle_dt"], group["conus_count"], marker="o", linewidth=2.8, color=color, label=model)
        latest = latest_nco_row(nco)
        if latest is not None:
            latest_text = f"{int(latest['conus_count'])} CONUS RAOBs"
            cycle_text = f"{latest['cycle_date_utc']} {str(latest['cycle_hour']).zfill(2)}Z {latest['model']}"
            fig.text(0.07, 0.845, latest_text, fontsize=44, color=red, fontweight="bold", va="top")
            fig.text(0.56, 0.825, f"latest parsed cycle\n{cycle_text}", fontsize=18, color=text, fontweight="bold", va="top")
        lowest = recent.loc[recent["conus_count"].idxmin()]
        ax.scatter([lowest["cycle_dt"]], [lowest["conus_count"]], color=red, s=55, zorder=5)
        ax.text(lowest["cycle_dt"], lowest["conus_count"] + 0.35, f"low {int(lowest['conus_count'])}", color=red, fontsize=10.5, fontweight="bold", ha="center", va="bottom")
        ax.set_ylim(max(0, float(recent["conus_count"].min()) - 2.0), float(recent["conus_count"].max()) + 2.0)
    ax.set_ylabel("CONUS RAOBs available for ingest", color=muted)
    ax.tick_params(colors=muted, labelsize=10)
    ax.grid(True, axis="y", color=border, linewidth=0.65, alpha=0.42)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=muted)
    for spine in ax.spines.values():
        spine.set_color(border)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    fig.text(0.08, 0.24, "NCO ingest counts are not the same as IGRA archive counts.", fontsize=16, color="#dbeafe", fontweight="bold")
    dark_note_footer(fig, "Source: NWS/NCEP/NCO SDM Administrative Messages. Parsed counts reflect model-ingest availability for listed cycles.", y=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_two_layer_evidence_context(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = decline_start_metrics(data["igra"])
    series = metrics["series"]
    nco = data["nco"]
    bg = "#0b1220"
    panel = "#111827"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    blue = "#60a5fa"
    red = "#fb7185"
    amber = "#fcd34d"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.95, "Two Data Layers Point to Operational Stress", fontsize=27, fontweight="bold", color=text, va="top")
    fig.text(0.07, 0.91, "IGRA archive counts and NCO model-ingest messages are separate evidence layers", fontsize=12.5, color=muted, va="top")
    ax1 = fig.add_axes([0.08, 0.46, 0.85, 0.30], facecolor=panel)
    if not series.empty:
        ax1.axhline(0, color="#e2e8f0", linewidth=1.0, alpha=0.9)
        ax1.plot(series["date"], series["rolling_percent_difference"], color=blue, linewidth=2.8)
        ax1.fill_between(series["date"], series["rolling_percent_difference"], 0, where=series["rolling_percent_difference"] < 0, color="#7f1d1d", alpha=0.30)
        ax1.text(series["date"].iloc[-1] + pd.Timedelta(days=12), series["rolling_percent_difference"].iloc[-1], "IGRA 30-day\narchive gap", color=blue, fontsize=10.5, fontweight="bold", va="center")
        ax1.set_xlim(pd.Timestamp("2025-01-01") - pd.Timedelta(days=10), series["date"].max() + pd.Timedelta(days=58))
    ax1.set_ylabel("IGRA vs baseline (%)", color=muted)
    ax1.tick_params(colors=muted, labelsize=9)
    ax1.grid(True, axis="y", color=border, linewidth=0.6, alpha=0.4)
    for spine in ax1.spines.values():
        spine.set_color(border)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    ax2 = fig.add_axes([0.08, 0.21, 0.85, 0.18], facecolor=panel)
    if not nco.empty:
        recent = nco.dropna(subset=["cycle_dt", "conus_count"]).sort_values("cycle_dt")
        for model, group in recent.groupby("model"):
            ax2.plot(group["cycle_dt"], group["conus_count"], marker="o", linewidth=2.4, color=amber if model == "NAM" else red, label=model)
        ax2.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=muted)
    ax2.set_ylabel("NCO CONUS RAOBs", color=muted)
    ax2.tick_params(colors=muted, labelsize=9)
    ax2.grid(True, axis="y", color=border, linewidth=0.6, alpha=0.4)
    for spine in ax2.spines.values():
        spine.set_color(border)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    fig.text(0.08, 0.815, "Layer 1: archived soundings are below the same-date baseline", fontsize=13.5, color=text, fontweight="bold")
    fig.text(0.08, 0.415, "Layer 2: NCO messages report model-ingest availability by cycle", fontsize=13.5, color=text, fontweight="bold")
    dark_note_footer(fig, "Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages. Separate signals; not a causation claim.", y=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_operational_pipeline_explainer(data: dict[str, pd.DataFrame], path: Path) -> None:
    bg = "#0b1220"
    panel = "#172033"
    border = "#334155"
    text = "#f8fafc"
    muted = "#a7b0c0"
    blue = "#60a5fa"
    red = "#fb7185"
    amber = "#fcd34d"
    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.95, "Why Upper-Air Observations Matter", fontsize=31, fontweight="bold", color=text, va="top")
    fig.text(0.07, 0.91, "Weather-balloon data help initialize the vertical structure of forecast models", fontsize=12.8, color=muted, va="top")
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.63], facecolor=bg)
    ax.set_axis_off()
    steps = [
        ("1", "Balloon launch", "Temperature, moisture,\nwind with height"),
        ("2", "Vertical profile", "A real atmosphere sample\nabove one station"),
        ("3", "Model ingest", "RAOBs available for\nNAM/GFS initialization"),
        ("4", "Forecast analysis", "Initial conditions for\nweather prediction"),
    ]
    y_positions = [0.78, 0.58, 0.38, 0.18]
    for (num, title, detail), y in zip(steps, y_positions):
        ax.add_patch(Rectangle((0.08, y - 0.07), 0.84, 0.13, facecolor=panel, edgecolor=border, linewidth=1.2))
        ax.text(0.12, y, num, color=amber, fontsize=26, fontweight="bold", va="center")
        ax.text(0.22, y + 0.026, title, color=text, fontsize=19, fontweight="bold", va="center")
        ax.text(0.22, y - 0.035, detail, color=muted, fontsize=11.2, va="center", linespacing=1.15)
        if y != y_positions[-1]:
            ax.annotate("", xy=(0.50, y - 0.13), xytext=(0.50, y - 0.075), arrowprops={"arrowstyle": "->", "color": blue, "lw": 1.8})
    ax.text(0.08, 0.04, "Where gaps can appear:", color=red, fontsize=16, fontweight="bold", va="bottom")
    ax.text(0.08, 0.005, "launch reductions, missing reports, ingest problems, or archive delays", color=text, fontsize=13, va="bottom")
    metrics = cumulative_deficit_since_breakpoint(data["igra"])
    if not math.isnan(float(metrics["net_deficit"])):
        fig.text(0.07, 0.845, f"{float(metrics['net_deficit']):,.0f} fewer archived soundings vs baseline", fontsize=22, color=red, fontweight="bold", va="top")
        fig.text(0.07, 0.813, "since Apr 3, 2025", fontsize=15, color=muted, va="top")
    dark_note_footer(fig, "Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages. This explains the data pathway, not a measured forecast-error attribution.", y=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def create_carousel_cover(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = cumulative_deficit_since_breakpoint(data["igra"])
    bg = "#0b1220"
    text = "#f8fafc"
    muted = "#a7b0c0"
    red = "#fb7185"
    fig = plt.figure(figsize=(10.8, 13.5), facecolor=bg)
    fig.text(0.07, 0.93, "U.S. Upper-Air Observations\nAre Below Normal", fontsize=34, fontweight="bold", color=text, va="top", linespacing=1.05)
    fig.text(0.07, 0.73, f"{float(metrics['net_deficit']):,.0f}", fontsize=112, color=red, fontweight="bold", va="top")
    fig.text(0.07, 0.57, "fewer archived soundings\nthan the 2021-2024 same-date baseline", fontsize=27, color=text, fontweight="bold", va="top", linespacing=1.12)
    fig.text(0.07, 0.40, "Following early-2025 DOGE/NOAA/NWS operational changes, the IGRA archive shows a persistent shortfall. The timing is context, not proof of causation.", fontsize=16, color=muted, va="top", wrap=True)
    dark_note_footer(fig, "Source: NOAA/NCEI IGRA v2. Counts may differ from actual launches due to ingest, archive, or reporting delays. Not a causation claim.", y=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=bg)
    plt.close(fig)


def write_decline_start_summary(data: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = decline_start_metrics(data["igra"])
    breakpoint_date = metrics["breakpoint_date"]
    partial_date = metrics["partial_date"]
    latest_complete = metrics["latest_complete_date"]
    lines = {
        "detection_rule": (
            "30-day rolling percent difference vs 2021-2024 same-date baseline; "
            "first date <= -3.0%; requires next 30 consecutive valid days below 0%"
        ),
        "detected_breakpoint_date": breakpoint_date.date().isoformat() if breakpoint_date is not None else "",
        "breakpoint_rolling_percent": f"{float(metrics['breakpoint_percent']):.2f}" if breakpoint_date is not None else "",
        "2025 Jan average percent difference": f"{float(metrics['jan_2025_percent']):.2f}",
        "2025 Mar average percent difference": f"{float(metrics['mar_2025_percent']):.2f}",
        "2025 Apr average percent difference": f"{float(metrics['apr_2025_percent']):.2f}",
        "2026 YTD percent difference": f"{float(metrics['ytd_2026_percent']):.2f}",
        "latest_complete_date": latest_complete.date().isoformat() if latest_complete is not None else "",
        "excluded_preliminary_date": partial_date.date().isoformat() if partial_date is not None else "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in lines.items():
            handle.write(f"{key}: {value}\n")


def write_x_caption_options(data: dict[str, pd.DataFrame], path: Path) -> None:
    latest_igra = latest_complete_igra_row(data["igra"])
    latest_nco = latest_nco_row(data["nco"])
    percent = "below"
    date_text = "the latest IGRA date"
    avg_text = "the latest 7-day average"
    nco_count = "the latest"
    if latest_igra is not None:
        if pd.notna(latest_igra.get("percent_vs_baseline")):
            percent_value = float(latest_igra["percent_vs_baseline"])
            if percent_value < 0:
                percent = f"{abs(percent_value):.1f}% below"
            else:
                percent = f"{percent_value:.1f}% above"
        date_text = latest_igra["date"].date().isoformat()
        avg_text = f"{latest_igra['launches_7d_avg']:.2f}/day"
    if latest_nco is not None:
        nco_count = str(int(latest_nco["conus_count"]))

    captions = [
        (
            "neutral",
            f"U.S. upper-air observations are running {percent} the prior 5-year same-date IGRA baseline as of {date_text}. "
            f"The latest NCO SDM message shows {nco_count} CONUS RAOBs available for model ingest. This coincides with operational changes, but this chart is not a causation claim.",
        ),
        (
            "urgent",
            f"The upper-air network is a core input to forecasts. As of {date_text}, archived CONUS soundings are {percent} the prior 5-year same-date baseline, with a 7-day average of {avg_text}. "
            f"NCO model-ingest availability is a separate operational layer: {nco_count} CONUS RAOBs in the latest parsed message.",
        ),
        (
            "meteorologist-focused",
            f"For forecasters: IGRA archived CONUS soundings are {percent} the prior 5-year same-date baseline as of {date_text}. "
            f"The latest NCO SDM layer reports {nco_count} CONUS RAOBs available for model ingest. Treat IGRA archive counts and NCO ingest counts as separate signals.",
        ),
        (
            "public-interest",
            f"Weather balloons still matter. The latest IGRA archive data show U.S. upper-air observations running {percent} the prior 5-year same-date baseline as of {date_text}. "
            f"That trend follows announced reductions at selected sites, but this graphic does not claim those notices caused the full observed decline.",
        ),
        (
            "technical/data-focused",
            f"Data note: IGRA counts one archived sounding/header record per launch; the latest 7-day average is {avg_text} on {date_text}, {percent} the prior 5-year same-date baseline. "
            f"NCO SDM messages are parsed separately for model-ingest availability; latest CONUS count = {nco_count}.",
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for label, text in captions:
            handle.write(f"{label}:\n{text}\n\n")


def write_x_caption_options_v2(data: dict[str, pd.DataFrame], path: Path) -> None:
    latest_igra = latest_complete_igra_row(data["igra"])
    latest_nco = latest_nco_row(data["nco"])
    date_text = "latest IGRA date"
    pct_text = "below baseline"
    nco_text = "latest NCO count unavailable"
    if latest_igra is not None and pd.notna(latest_igra.get("percent_vs_baseline")):
        date_text = latest_igra["date"].date().isoformat()
        pct_text = f"{abs(float(latest_igra['percent_vs_baseline'])):.1f}% below baseline"
    if latest_nco is not None:
        nco_text = f"NCO ingest count: {int(latest_nco['conus_count'])} CONUS RAOBs"

    captions = [
        (
            "neutral",
            f"U.S. upper-air observations are {pct_text} in IGRA as of {date_text}. {nco_text}. IGRA and NCO are separate data layers; this is an observed decline, not a causation claim.",
        ),
        (
            "meteorologist-focused",
            f"For forecasters: IGRA archived soundings are {pct_text} as of {date_text}; {nco_text}. This coincides with operational changes, but IGRA archive counts and NCO ingest counts are separate signals.",
        ),
        (
            "public-interest",
            f"Weather balloon observations are running {pct_text} in the latest IGRA data. This follows announced reductions at selected sites, but does not prove causation. NCO ingest availability is shown separately.",
        ),
        (
            "technical/data-focused",
            f"Data check: IGRA 7-day archived CONUS soundings are {pct_text} on {date_text}. {nco_text}. IGRA = archive layer; NCO = model-ingest layer. No causation claim.",
        ),
        (
            "thread-starter",
            f"Upper-air observations are showing an observed decline in IGRA: {pct_text} as of {date_text}. Next: compare archive counts, NCO ingest availability, and station-level issue recaps as separate layers.",
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for label, text in captions:
            handle.write(f"{label} ({len(text)} chars):\n{text}\n\n")


def create_dashboard(data: dict[str, pd.DataFrame], path: Path) -> None:
    colors = social_palette()
    latest = latest_complete_igra_row(data["igra"])
    plot_df, excluded_partial, _reason = plotted_igra_current(data["igra"])
    windows, partial_date = complete_current_windows(data["igra"], windows=(30, 60, 90, 180))
    metrics = cumulative_deficit_since_breakpoint(data["igra"])
    statuses, impacted, cycle_text = latest_station_status_rows(data)
    latest_nco = latest_nco_row(data["nco"])

    latest_date = latest["date"].date().isoformat() if latest is not None else "unavailable"
    archive_pct = float(latest["percent_vs_baseline"]) if latest is not None else float("nan")
    archive_observed = float(latest["launches_7d_avg"]) if latest is not None else float("nan")
    archive_expected = float(latest["baseline_5yr_avg"]) if latest is not None else float("nan")
    window_90 = windows[windows["days"] == 90].iloc[0] if not windows.empty and (windows["days"] == 90).any() else None
    deficit_90 = abs(float(window_90["deficit"])) if window_90 is not None else float("nan")
    pct_90 = float(window_90["percent"]) if window_90 is not None else float("nan")
    net_deficit = float(metrics["net_deficit"])
    latest_nco_count = int(latest_nco["conus_count"]) if latest_nco is not None else None
    fig = plt.figure(figsize=(16, 10), facecolor=colors["bg"])
    fig.text(0.05, 0.952, "CONUS UPPER-AIR DATA WATCH", fontsize=27, fontweight="bold", color=colors["ink"], va="top")
    fig.text(0.05, 0.916, "Archive trend, seasonal baseline, and a dated operational-status snapshot", fontsize=12.6, color=colors["muted"], va="top")
    social_card(fig, 0.05, 0.755, 0.35, 0.115)
    social_card(fig, 0.415, 0.755, 0.30, 0.115, alt=True)
    social_card(fig, 0.73, 0.755, 0.22, 0.115)
    fig.text(0.07, 0.842, f"{archive_pct:.1f}%" if math.isfinite(archive_pct) else "N/A", fontsize=31, fontweight="bold", color=colors["red"], va="top")
    fig.text(0.20, 0.835, "7-day archive average\nvs same-date baseline", fontsize=10.8, fontweight="bold", color=colors["ink"], va="top", linespacing=1.16)
    fig.text(0.07, 0.777, f"AS OF {latest_date}  •  {archive_observed:.1f}/DAY VS {archive_expected:.1f}/DAY", fontsize=8.2, color=colors["muted"], va="top")
    fig.text(0.435, 0.842, f"{deficit_90:,.0f}" if math.isfinite(deficit_90) else "N/A", fontsize=31, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.555, 0.835, "fewer archived soundings\nin the past 90 days", fontsize=10.8, fontweight="bold", color=colors["ink"], va="top", linespacing=1.16)
    fig.text(0.435, 0.777, f"{pct_90:.1f}% VS EXPECTED" if math.isfinite(pct_90) else "WINDOW UNAVAILABLE", fontsize=8.2, color=colors["muted"], va="top")
    fig.text(0.75, 0.842, f"{len(impacted)}", fontsize=31, fontweight="bold", color=colors["red"], va="top")
    fig.text(0.825, 0.835, "reported\nissue statuses", fontsize=9.7, fontweight="bold", color=colors["ink"], va="top", linespacing=1.16)
    nco_count_text = f"{latest_nco_count}" if latest_nco_count is not None else "N/A"
    fig.text(0.825, 0.797, f"{nco_count_text} RAOBs for ingest", fontsize=9.3, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.75, 0.777, cycle_text.upper(), fontsize=7.7, color=colors["muted"], va="top")

    ax_trend = fig.add_axes([0.05, 0.42, 0.55, 0.25])
    style_social_axis(ax_trend)
    ax_trend.set_title("ARCHIVED SOUNDINGS VS SEASONAL BASELINE", loc="left", fontsize=11.5, fontweight="bold", color=colors["ink"], pad=10)
    if not plot_df.empty:
        ax_trend.plot(plot_df["date"], plot_df["baseline_5yr_avg"], color=colors["muted"], linewidth=2.0, linestyle="--", label="expected baseline")
        ax_trend.plot(plot_df["date"], plot_df["launches_7d_avg"], color=colors["blue"], linewidth=3.0, label="observed 7-day average")
        below = plot_df["launches_7d_avg"] < plot_df["baseline_5yr_avg"]
        ax_trend.fill_between(plot_df["date"], plot_df["launches_7d_avg"], plot_df["baseline_5yr_avg"], where=below, color=colors["red_soft"], alpha=0.7, interpolate=True)
        ax_trend.set_xlim(plot_df["date"].min() - pd.Timedelta(days=5), plot_df["date"].max() + pd.Timedelta(days=10))
        ax_trend.legend(loc="lower left", frameon=False, fontsize=8.6, labelcolor=colors["muted"], ncol=2)
    ax_trend.set_ylabel("soundings / day", color=colors["muted"], fontsize=9)
    ax_trend.xaxis.set_major_locator(mdates.MonthLocator())
    ax_trend.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax_windows = fig.add_axes([0.66, 0.43, 0.29, 0.24])
    style_social_axis(ax_windows, xgrid=True, ygrid=False)
    ax_windows.set_title("GAP ACROSS RECENT WINDOWS", loc="left", fontsize=11.5, fontweight="bold", color=colors["ink"], pad=10)
    if not windows.empty:
        window_chart = windows.sort_values("days").reset_index(drop=True)
        y = np.arange(len(window_chart))
        values = window_chart["percent"].to_numpy(dtype=float)
        ax_windows.barh(y, values, height=0.54, color=colors["red"], alpha=0.94)
        ax_windows.axvline(0, color=colors["ink"], linewidth=1.1)
        ax_windows.set_xlim(min(-1.0, float(np.nanmin(values)) * 1.55), 1.0)
        for index, row in window_chart.iterrows():
            value = float(row["percent"])
            ax_windows.text(value - 0.06, index, f"{value:.1f}%", ha="right", va="center", fontsize=10.2, fontweight="bold", color=colors["ink"])
        ax_windows.set_yticks(y, [f"{int(days)}D" for days in window_chart["days"]], fontweight="bold")
        ax_windows.invert_yaxis()
        ax_windows.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax_windows.set_xlabel("difference from expected", color=colors["muted"], fontsize=8.7)

    ax_map = fig.add_axes([0.05, 0.105, 0.55, 0.24])
    draw_social_conus_base(ax_map)
    ax_map.set_title("CONUS STATION-STATUS SNAPSHOT", loc="left", fontsize=11.5, fontweight="bold", color=colors["ink"], pad=8)
    if not statuses.empty:
        available = statuses[statuses["status"].isin(["available/no issue", "unknown"])]
        ax_map.scatter(available["longitude"], available["latitude"], s=25, c=colors["green"], edgecolor=colors["bg"], linewidth=0.45, alpha=0.76, zorder=3)
    if not impacted.empty:
        impact_colors = {"missing/problem": colors["red"], "partial/quality issue": colors["amber"], "issue": colors["amber"]}
        ax_map.scatter(impacted["longitude"], impacted["latitude"], s=65, c=impacted["status"].map(impact_colors).fillna(colors["amber"]), edgecolor=colors["ink"], linewidth=0.7, zorder=4)
    map_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["green"], markeredgecolor=colors["bg"], markersize=7, label="available / no issue"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["red"], markeredgecolor=colors["ink"], markersize=7, label="missing / problem"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["amber"], markeredgecolor=colors["ink"], markersize=7, label="partial / quality"),
    ]
    ax_map.legend(handles=map_handles, loc="lower left", frameon=True, facecolor=colors["panel"], edgecolor=colors["border"], labelcolor=colors["muted"], fontsize=7.8, ncol=3)

    social_card(fig, 0.66, 0.105, 0.29, 0.265, alt=True)
    fig.text(0.68, 0.345, "DATA STATUS", fontsize=10.2, fontweight="bold", color=colors["blue"], va="top")
    fig.text(0.68, 0.313, "ARCHIVE", fontsize=8.5, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.68, 0.293, "Daily archive records compared with the\n2021-2024 same-date baseline.", fontsize=9.0, color=colors["ink"], va="top", linespacing=1.16)
    fig.text(0.68, 0.247, "NCO", fontsize=8.5, fontweight="bold", color=colors["amber"], va="top")
    fig.text(0.68, 0.227, "One operational message cycle; its counts\nare not comparable to the IGRA archive total.", fontsize=9.0, color=colors["ink"], va="top", linespacing=1.16)
    data_status = f"Archive latest: {latest_date}"
    if partial_date is not None:
        data_status += f" | {partial_date.date().isoformat()} excluded as incomplete"
    fig.text(0.68, 0.157, data_status, fontsize=7.9, color=colors["muted"], va="top")
    fig.text(0.68, 0.132, f"NCO cycle: {cycle_text.upper()}", fontsize=7.9, color=colors["muted"], va="top")

    footer = "Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM Administrative Messages; station master. Archive and operational records can differ from confirmed launches because of ingest, archive, or reporting delays. Not a causation claim."
    if partial_date is not None or excluded_partial:
        footer += f" Latest incomplete archive date excluded: {(partial_date.date().isoformat() if partial_date is not None else 'yes')}."
    social_footer(fig, footer, y=0.047, width=205)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=colors["bg"])
    plt.close(fig)


def print_x_v2_validation(data: dict[str, pd.DataFrame], social_dir: Path) -> None:
    x_path = social_dir / "x_post_main_v3.png"
    if x_path.exists():
        image = plt.imread(x_path)
        height, width = image.shape[:2]
        print(f"x_post_main_v3 dimensions: {width}x{height}")
    latest = latest_complete_igra_row(data["igra"])
    if latest is not None:
        print(f"Latest complete IGRA date used: {latest['date'].date().isoformat()}")
    excluded, reason = latest_partial_day_status(data["igra"])
    print(f"Partial latest day excluded from plotted line: {'yes' if excluded else 'no'}")
    if reason:
        print(f"Partial-day rule: {reason}")
    divergence = divergence_summary(data["igra"])
    print(
        f"Divergence start estimate: {divergence['start']} "
        f"({divergence['detail']}; {divergence['shorter']})"
    )
    windows = rolling_windows(data["igra"])
    if not windows.empty:
        print("30/60/90-day deficits:")
        for _, row in windows.iterrows():
            print(
                f"  {row['window']}: observed {row['observed']:.0f}, "
                f"expected {row['expected']:.0f}, difference {row['deficit']:.0f}, "
                f"percent {row['percent']:.1f}%"
            )
    latest_nco = latest_nco_row(data["nco"])
    if latest_nco is not None:
        print(
            f"Latest NCO cycle/count: {latest_nco['cycle_date_utc']} "
            f"{str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco['model']} / "
            f"{int(latest_nco['conus_count'])}"
        )
    print(f"x_post_main_v3 output: {x_path}")
    divergence_path = social_dir / "x_post_2025_divergence.png"
    if divergence_path.exists():
        image = plt.imread(divergence_path)
        height, width = image.shape[:2]
        print(f"x_post_2025_divergence dimensions: {width}x{height}")
        print(f"x_post_2025_divergence output: {divergence_path}")
    onset = first_rolling_gap_2025(data["igra"])
    if onset is not None:
        print(f"2025 divergence onset estimate: {onset.date().isoformat()}")
    decline_path = social_dir / "decline_start_v2.png"
    if decline_path.exists():
        image = plt.imread(decline_path)
        height, width = image.shape[:2]
        print(f"decline_start_v2 dimensions: {width}x{height}")
        print(f"decline_start_v2 output: {decline_path}")
    decline_v3_path = social_dir / "decline_start_v3.png"
    if decline_v3_path.exists():
        image = plt.imread(decline_v3_path)
        height, width = image.shape[:2]
        print(f"decline_start_v3 dimensions: {width}x{height}")
        print(f"decline_start_v3 output: {decline_v3_path}")
    cumulative_path = social_dir / "cumulative_missing_soundings_context.png"
    if cumulative_path.exists():
        image = plt.imread(cumulative_path)
        height, width = image.shape[:2]
        print(f"cumulative_missing_soundings_context dimensions: {width}x{height}")
        print(f"cumulative_missing_soundings_context output: {cumulative_path}")
        cumulative_metrics = cumulative_deficit_since_breakpoint(data["igra"])
        print(
            "Cumulative missing soundings since breakpoint: "
            f"observed {float(cumulative_metrics['observed_total']):.0f}, "
            f"expected {float(cumulative_metrics['expected_total']):.0f}, "
            f"net fewer {float(cumulative_metrics['net_deficit']):.0f}, "
            f"percent {float(cumulative_metrics['percent_difference']):.1f}%"
        )
    for output_name in [
        "timeline_observed_decline_context.png",
        "observed_expected_windows_context.png",
        "station_deficit_map_context.png",
        "impacted_station_small_multiples.png",
        "nco_ingest_context.png",
        "two_layer_evidence_context.png",
        "upper_air_operational_pipeline.png",
        "social_hero_decline.png",
        "social_observed_expected_windows.png",
        "social_cumulative_gap.png",
        "social_station_impacts.png",
        "social_two_layer_evidence.png",
        "carousel_doge_context/carousel_01_hero.png",
        "carousel_doge_context/carousel_02_timeline.png",
        "carousel_doge_context/carousel_03_cumulative_missing.png",
        "carousel_doge_context/carousel_04_station_map.png",
        "carousel_doge_context/carousel_05_two_data_layers.png",
    ]:
        output_path = social_dir / output_name
        if output_path.exists():
            image = plt.imread(output_path)
            height, width = image.shape[:2]
            print(f"{output_name} dimensions: {width}x{height}")
            print(f"{output_name} output: {output_path}")
    decline_metrics = decline_start_metrics(data["igra"])
    breakpoint_date = decline_metrics["breakpoint_date"]
    if breakpoint_date is None:
        print("WARNING: No persistent below-baseline breakpoint detected.")
    else:
        print(f"Detected breakpoint date: {breakpoint_date.date().isoformat()}")
    latest_complete = decline_metrics["latest_complete_date"]
    if latest_complete is not None:
        print(f"Latest complete date: {latest_complete.date().isoformat()}")
    print(f"2026 YTD percent difference: {float(decline_metrics['ytd_2026_percent']):.1f}%")
    partial_date = decline_metrics["partial_date"]
    if partial_date is not None:
        print(f"Excluded preliminary date: {partial_date.date().isoformat()}")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    dashboard_path = outdir / DASHBOARD_PATH.name
    social_dir = outdir / SOCIAL_DIR.name
    data = load_data(outdir)
    create_dashboard(data, dashboard_path)
    create_social_graphics(data, social_dir)
    print(f"Dashboard output: {dashboard_path}")
    print(f"Social graphics output: {social_dir}")
    print_x_v2_validation(data, social_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
