"""Generate the mobile-first CONUS Upper-Air Data Watch social suite.

Usage
-----
Run ``python scripts/build_upper_air_social_graphics.py`` after the existing
``run_upper_air_monitor.py --refresh`` pipeline.  This module only reads the
monitor's CSV products, writes a portable JSON manifest, then renders graphics
from that manifest.  The manifest-only path lets a design refresh happen
without re-downloading archive or NCO data.

Design contract
---------------
These graphics describe archive and operational-message *availability* only.
They never claim a launch-count, model-ingest, model-skill, or causal impact.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


# Central visual configuration. Exact pixel exports make the suite predictable
# in social schedulers while the high DPI preserves type and map detail.
@dataclass(frozen=True)
class Theme:
    background: str = "#061521"
    panel: str = "#0D2538"
    panel_alt: str = "#113149"
    text: str = "#F8FBFF"
    muted: str = "#AFC1D4"
    grid: str = "#294960"
    border: str = "#294A63"
    observed: str = "#59C8F5"
    baseline: str = "#C1CBD8"
    deficit: str = "#FF704F"
    deficit_fill: str = "#6F3340"
    amber: str = "#F6C85F"
    clean: str = "#52D3A2"
    unknown: str = "#8FA3B8"
    land: str = "#173A52"
    water: str = "#081E2F"
    brand: str = "wall.cloud"
    dpi: int = 300


THEME = Theme()
WINDOW_DAYS = (30, 60, 90, 180)
MANIFEST_VERSION = 2
OUTPUT_FILENAMES = (
    "hero_big_number.png",
    "carousel_01_hook.png",
    "carousel_02_trend.png",
    "carousel_03_map.png",
    "carousel_04_gaps.png",
    "carousel_05_caveats.png",
    "split_expected_vs_reality.png",
    "minimalist_trend.png",
    "original_dashboard_style.png",
)
EXACT_CAVEATS = (
    "Archive records are compared with the 2021–2024 same-date baseline.",
    "NCO status reflects operational-message reporting, not confirmed IGRA archive totals.",
)


@dataclass
class MonitorInputs:
    igra: pd.DataFrame
    nco: pd.DataFrame
    issues: pd.DataFrame
    stations: pd.DataFrame
    input_paths: dict[str, Path]
    states_geojson_path: Path | None = None


@dataclass
class WatchMetrics:
    current_igra: pd.DataFrame
    latest_complete: pd.Series | None
    partial_date: pd.Timestamp | None
    windows: pd.DataFrame
    latest_nco: pd.Series | None
    nco_cycle_text: str
    nco_count: int | None
    station_statuses: pd.DataFrame
    impacted_station_count: int


@dataclass
class SocialPayload:
    """Validated display data reconstructed from the JSON manifest."""

    generated_at: str | None
    latest_date: str | None
    partial_date: str | None
    nco_cycle: str | None
    observed: float
    expected: float
    gap_percent: float
    windows: pd.DataFrame
    series: pd.DataFrame
    stations: pd.DataFrame
    issue_count: int | None
    nco_count: int | None
    caveats: tuple[str, ...]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_monitor_inputs(repo_root: Path) -> MonitorInputs:
    """Load existing monitor products only; this function never refreshes data."""
    repo_root = repo_root.resolve()
    paths = {
        "igra_daily": repo_root / "outputs" / "conus_balloon_launches_by_year_daily.csv",
        "nco_availability": repo_root / "data" / "nco_raob_availability.csv",
        "nco_issues": repo_root / "data" / "nco_raob_station_issues.csv",
        "station_master": repo_root / "data" / "upper_air_station_master.csv",
    }
    return MonitorInputs(
        igra=read_csv(paths["igra_daily"]),
        nco=read_csv(paths["nco_availability"], dtype=str),
        issues=read_csv(paths["nco_issues"], dtype=str),
        stations=read_csv(paths["station_master"], dtype=str),
        input_paths=paths,
        states_geojson_path=repo_root / "comfortwx" / "mapping" / "data" / "us_states.geojson",
    )


def _prepare_igra(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    if "year" not in data:
        data["year"] = data["date"].dt.year
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    for column in ("launches", "launches_7d_avg", "baseline_5yr_avg", "difference_vs_baseline", "percent_vs_baseline"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["date"]).sort_values("date").copy()


def _attach_baseline_range(current: pd.DataFrame, igra: pd.DataFrame) -> pd.DataFrame:
    """Attach the 2021–2024 same-date min/max range used for context."""
    result = current.copy()
    history = _prepare_igra(igra)
    if result.empty or history.empty or "launches_7d_avg" not in history:
        return result
    history = history[history["year"].isin([2021, 2022, 2023, 2024])].dropna(subset=["launches_7d_avg"])
    if history.empty:
        return result
    history["month_day"] = history["date"].dt.strftime("%m-%d")
    band = (
        history.groupby("month_day", as_index=False)["launches_7d_avg"]
        .agg(baseline_low="min", baseline_high="max")
    )
    result["month_day"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%m-%d")
    return result.merge(band, on="month_day", how="left").drop(columns="month_day")


def _prepare_nco(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
    if "conus_count" in data:
        data["conus_count"] = pd.to_numeric(data["conus_count"], errors="coerce")
    if {"cycle_date_utc", "cycle_hour"}.issubset(data.columns):
        data["cycle_dt"] = pd.to_datetime(
            data["cycle_date_utc"].astype(str) + " " + data["cycle_hour"].astype(str).str.zfill(2) + ":00",
            errors="coerce",
            utc=True,
        )
    else:
        data["cycle_dt"] = pd.NaT
    if "message_time_utc" in data:
        data["message_dt"] = pd.to_datetime(data["message_time_utc"], errors="coerce", utc=True)
    return data


def _prepare_stations(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
    for column in ("latitude", "longitude"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    active = data.get("active_expected", pd.Series("true", index=data.index)).astype(str).str.lower().eq("true")
    return data[
        active
        & data["latitude"].between(24.0, 50.0)
        & data["longitude"].between(-126.0, -66.0)
    ].copy()


def _latest_complete_rows(igra: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Exclude a clearly partial last archive day using the monitor's existing rule."""
    data = _prepare_igra(igra)
    required = ["date", "launches", "launches_7d_avg", "baseline_5yr_avg"]
    if data.empty or any(column not in data for column in required):
        return pd.DataFrame(), None
    current = data[data["year"] == data["year"].max()].dropna(subset=required).sort_values("date").copy()
    if len(current) < 15:
        return current, None
    latest = current.iloc[-1]
    prior = current.iloc[-15:-1]
    is_partial = (
        float(latest["launches"]) < float(prior["launches"].median()) * 0.85
        and float(latest["launches_7d_avg"]) < float(current.iloc[-2]["launches_7d_avg"]) - 2.0
    )
    return (current.iloc[:-1].copy(), pd.Timestamp(latest["date"])) if is_partial else (current, None)


def _window_metrics(current: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return pd.DataFrame(columns=["days", "observed", "expected", "deficit", "percent", "latest"])
    latest = pd.Timestamp(current["date"].max())
    records: list[dict[str, object]] = []
    for days in WINDOW_DAYS:
        subset = current[(current["date"] >= latest - pd.Timedelta(days=days - 1)) & (current["date"] <= latest)]
        if subset.empty:
            continue
        observed = float(subset["launches"].sum())
        expected = float(subset["baseline_5yr_avg"].sum())
        deficit = observed - expected
        records.append(
            {
                "days": days,
                "observed": observed,
                "expected": expected,
                "deficit": deficit,
                "percent": deficit / expected * 100.0 if expected else np.nan,
                "latest": latest,
            }
        )
    return pd.DataFrame(records)


def _latest_nco_row(nco: pd.DataFrame) -> pd.Series | None:
    if nco.empty or "cycle_dt" not in nco:
        return None
    sort_columns = [column for column in ("cycle_dt", "message_dt") if column in nco]
    valid = nco.dropna(subset=["cycle_dt"]).sort_values(sort_columns)
    return valid.iloc[-1] if not valid.empty else None


def _status_for_issue(category: str) -> tuple[str, int]:
    if category in {"no_report", "unavailable", "equipment_failure"}:
        return "missing / problem", 2
    return "partial / quality", 1


def _station_statuses(inputs: MonitorInputs, nco: pd.DataFrame, latest_nco: pd.Series | None) -> pd.DataFrame:
    stations = _prepare_stations(inputs.stations)
    if stations.empty:
        return stations
    stations["status"] = "unknown" if latest_nco is None else "available / no issue"
    stations["severity"] = 0
    required = {"cycle_date_utc", "cycle_hour", "station_id"}
    if latest_nco is None or inputs.issues.empty or not required.issubset(inputs.issues.columns):
        return stations
    latest_issues = inputs.issues[
        (inputs.issues["cycle_date_utc"].astype(str) == str(latest_nco["cycle_date_utc"]))
        & (inputs.issues["cycle_hour"].astype(str).str.zfill(2) == str(latest_nco["cycle_hour"]).zfill(2))
    ]
    for _, issue in latest_issues.iterrows():
        station_id = str(issue.get("station_id", "")).upper()
        status, severity = _status_for_issue(str(issue.get("issue_category", "other")))
        mask = stations["station_id"].astype(str).str.upper().eq(station_id)
        if not mask.any() or severity < int(stations.loc[mask, "severity"].max()):
            continue
        stations.loc[mask, "status"] = status
        stations.loc[mask, "severity"] = severity
    return stations


def calculate_metrics(inputs: MonitorInputs) -> WatchMetrics:
    current, partial_date = _latest_complete_rows(inputs.igra)
    latest_complete = current.iloc[-1] if not current.empty else None
    windows = _window_metrics(current)
    nco = _prepare_nco(inputs.nco)
    latest_nco = _latest_nco_row(nco)
    nco_cycle_text = "NCO status unavailable"
    nco_count: int | None = None
    if latest_nco is not None:
        nco_cycle_text = f"{latest_nco['cycle_date_utc']} {str(latest_nco['cycle_hour']).zfill(2)}Z {latest_nco.get('model', 'NCO')}"
        if pd.notna(latest_nco.get("conus_count")):
            nco_count = int(latest_nco["conus_count"])
    statuses = _station_statuses(inputs, nco, latest_nco)
    impacted = 0 if statuses.empty else int(statuses["status"].isin(["missing / problem", "partial / quality"]).sum())
    return WatchMetrics(
        current_igra=current,
        latest_complete=latest_complete,
        partial_date=partial_date,
        windows=windows,
        latest_nco=latest_nco,
        nco_cycle_text=nco_cycle_text,
        nco_count=nco_count,
        station_statuses=statuses,
        impacted_station_count=impacted,
    )


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _format_number(value: float | int | None, *, decimals: int = 0) -> str:
    numeric = _safe_float(value)
    return "N/A" if not math.isfinite(numeric) else f"{numeric:,.{decimals}f}"


def _format_percent(value: float | int | None) -> str:
    numeric = _safe_float(value)
    return "N/A" if not math.isfinite(numeric) else f"{numeric:.1f}%"


def _window_row(payload: SocialPayload, days: int = 90) -> pd.Series | None:
    if payload.windows.empty or "days" not in payload.windows:
        return None
    matches = payload.windows[payload.windows["days"] == days]
    return matches.iloc[0] if not matches.empty else None


def _manifest(metrics: WatchMetrics, output_paths: list[Path], inputs: MonitorInputs) -> dict[str, object]:
    observed = expected = gap_percent = float("nan")
    if metrics.latest_complete is not None:
        observed = _safe_float(metrics.latest_complete.get("launches_7d_avg"))
        expected = _safe_float(metrics.latest_complete.get("baseline_5yr_avg"))
        gap_percent = _safe_float(metrics.latest_complete.get("percent_vs_baseline"))
    windows = [
        {
            "days": int(row["days"]),
            "observed": round(float(row["observed"]), 2),
            "expected": round(float(row["expected"]), 2),
            "deficit": round(float(row["deficit"]), 2),
            "percent_difference": round(float(row["percent"]), 2),
        }
        for _, row in metrics.windows.iterrows()
    ]
    series = _attach_baseline_range(metrics.current_igra, inputs.igra)
    time_series = {
        "date": [pd.Timestamp(value).date().isoformat() for value in series.get("date", pd.Series(dtype="datetime64[ns]"))],
        "observed_7d_average": [round(_safe_float(value), 3) for value in series.get("launches_7d_avg", pd.Series(dtype=float))],
        "expected_baseline": [round(_safe_float(value), 3) for value in series.get("baseline_5yr_avg", pd.Series(dtype=float))],
        "daily_archived_soundings": [round(_safe_float(value), 3) for value in series.get("launches", pd.Series(dtype=float))],
        "baseline_range_low": [
            round(_safe_float(value), 3) if math.isfinite(_safe_float(value)) else None
            for value in series.get("baseline_low", pd.Series(np.nan, index=series.index))
        ],
        "baseline_range_high": [
            round(_safe_float(value), 3) if math.isfinite(_safe_float(value)) else None
            for value in series.get("baseline_high", pd.Series(np.nan, index=series.index))
        ],
    }
    station_columns = [column for column in ("station_id", "latitude", "longitude", "status") if column in metrics.station_statuses]
    station_records = []
    if station_columns:
        for record in metrics.station_statuses[station_columns].to_dict(orient="records"):
            station_records.append(
                {
                    key: (round(float(value), 4) if key in {"latitude", "longitude"} and pd.notna(value) else str(value))
                    for key, value in record.items()
                    if pd.notna(value)
                }
            )
    summary = {
        "available_no_issue": int((metrics.station_statuses.get("status") == "available / no issue").sum()) if not metrics.station_statuses.empty else 0,
        "reported_problem": int((metrics.station_statuses.get("status") == "missing / problem").sum()) if not metrics.station_statuses.empty else 0,
        "partial_or_quality": int((metrics.station_statuses.get("status") == "partial / quality").sum()) if not metrics.station_statuses.empty else 0,
        "unknown": int((metrics.station_statuses.get("status") == "unknown").sum()) if not metrics.station_statuses.empty else 0,
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_files_used": {name: str(path) for name, path in inputs.input_paths.items()},
        "latest_complete_archive_date": metrics.latest_complete["date"].date().isoformat() if metrics.latest_complete is not None else None,
        "excluded_incomplete_archive_date": metrics.partial_date.date().isoformat() if metrics.partial_date is not None else None,
        "nco_cycle": metrics.nco_cycle_text if metrics.latest_nco is not None else None,
        "kpis": {
            "seven_day_archive_observed_per_day": round(observed, 2) if math.isfinite(observed) else None,
            "seven_day_archive_expected_per_day": round(expected, 2) if math.isfinite(expected) else None,
            "seven_day_archive_percent_difference": round(gap_percent, 2) if math.isfinite(gap_percent) else None,
            "recent_windows": windows,
            "reported_operational_issue_statuses": metrics.impacted_station_count if metrics.latest_nco is not None else None,
            "nco_conus_raobs_for_ingest": metrics.nco_count,
        },
        # These values make every chart reproducible from the manifest alone.
        "time_series": time_series,
        "station_status_summary": summary,
        "station_statuses": station_records,
        "output_image_paths": [str(path) for path in output_paths],
        "caveats": list(EXACT_CAVEATS),
    }


def _series_from_manifest(value: object) -> pd.DataFrame:
    if not isinstance(value, Mapping):
        return pd.DataFrame(columns=["date", "observed", "baseline", "daily"])
    dates = value.get("date", [])
    row_count = len(dates) if isinstance(dates, list) else 0

    def values(primary: str, legacy: str) -> list[object]:
        raw = value.get(primary, value.get(legacy, []))
        return raw if isinstance(raw, list) and len(raw) == row_count else [None] * row_count

    data = pd.DataFrame(
        {
            "date": dates if isinstance(dates, list) else [],
            "observed": values("observed_7d_average", "observed"),
            "baseline": values("expected_baseline", "baseline"),
            "daily": values("daily_archived_soundings", "daily"),
            "baseline_low": values("baseline_range_low", "baseline_low"),
            "baseline_high": values("baseline_range_high", "baseline_high"),
        }
    )
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("observed", "baseline", "daily", "baseline_low", "baseline_high"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_social_payload(manifest_path: Path) -> SocialPayload:
    """Read a current or legacy manifest without requiring every optional field."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Metrics manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metrics manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Metrics manifest must contain a JSON object: {manifest_path}")
    kpis = raw.get("kpis", {}) if isinstance(raw.get("kpis"), Mapping) else {}
    windows_raw = kpis.get("recent_windows", [])
    windows = pd.DataFrame(windows_raw if isinstance(windows_raw, list) else [])
    if not windows.empty:
        windows = windows.rename(columns={"percent_difference": "percent"})
        for column in ("days", "observed", "expected", "deficit", "percent"):
            if column in windows:
                windows[column] = pd.to_numeric(windows[column], errors="coerce")
    stations = pd.DataFrame(raw.get("station_statuses", []) if isinstance(raw.get("station_statuses"), list) else [])
    if not stations.empty:
        for column in ("latitude", "longitude"):
            if column in stations:
                stations[column] = pd.to_numeric(stations[column], errors="coerce")
        stations = stations.dropna(subset=[column for column in ("latitude", "longitude") if column in stations])
    caveats_raw = raw.get("caveats", EXACT_CAVEATS)
    caveats = tuple(str(item) for item in caveats_raw) if isinstance(caveats_raw, list) else EXACT_CAVEATS
    # Older manifests may still carry the retired causation disclaimer. Keep
    # the manifest backward-compatible without resurfacing that UI copy.
    caveats = tuple(item for item in caveats if item != "This is a data-availability diagnostic, not a causation claim.")
    return SocialPayload(
        generated_at=str(raw.get("generated_at")) if raw.get("generated_at") else None,
        latest_date=str(raw.get("latest_complete_archive_date")) if raw.get("latest_complete_archive_date") else None,
        partial_date=str(raw.get("excluded_incomplete_archive_date")) if raw.get("excluded_incomplete_archive_date") else None,
        nco_cycle=str(raw.get("nco_cycle")) if raw.get("nco_cycle") else None,
        observed=_safe_float(kpis.get("seven_day_archive_observed_per_day")),
        expected=_safe_float(kpis.get("seven_day_archive_expected_per_day")),
        gap_percent=_safe_float(kpis.get("seven_day_archive_percent_difference")),
        windows=windows,
        series=_series_from_manifest(raw.get("time_series")),
        stations=stations,
        issue_count=(int(_safe_float(kpis.get("reported_operational_issue_statuses"))) if math.isfinite(_safe_float(kpis.get("reported_operational_issue_statuses"))) else None),
        nco_count=(int(_safe_float(kpis.get("nco_conus_raobs_for_ingest"))) if math.isfinite(_safe_float(kpis.get("nco_conus_raobs_for_ingest"))) else None),
        caveats=caveats or EXACT_CAVEATS,
    )


def detect_sharp_drop(series: pd.DataFrame) -> tuple[pd.Timestamp, float] | None:
    """Return the largest day-to-day observed decline in the latest July.

    The callout is evidence-led: no annotation is placed when there are fewer
    than two valid observed points or when the series did not decline. Limiting
    July candidates to the most recent year keeps a current dashboard from
    highlighting a larger but stale July move.
    """
    if series.empty or not {"date", "observed"}.issubset(series.columns):
        return None
    data = series.dropna(subset=["date", "observed"]).copy()
    if len(data) < 2:
        return None
    data["change"] = data["observed"].diff()
    july = data[data["date"].dt.month == 7]
    if not july.empty:
        july = july[july["date"].dt.year.eq(int(july["date"].dt.year.max()))]
    candidates = july.dropna(subset=["change"]) if not july.empty else data.dropna(subset=["change"])
    if candidates.empty:
        return None
    row = candidates.loc[candidates["change"].idxmin()]
    return (pd.Timestamp(row["date"]), float(row["change"])) if float(row["change"]) < 0 else None


def _figure(width_px: int, height_px: int) -> plt.Figure:
    return plt.figure(
        figsize=(width_px / THEME.dpi, height_px / THEME.dpi),
        dpi=THEME.dpi,
        facecolor=THEME.background,
    )


def _save(fig: plt.Figure, path: Path, *, dpi_scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=max(1, round(THEME.dpi * dpi_scale)),
        facecolor=THEME.background,
        metadata={"Software": "CONUS Upper-Air Data Watch / wall.cloud"},
    )
    plt.close(fig)


def _card(
    fig: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    alternate: bool = False,
    accent: str | None = None,
    radius: float = 0.012,
) -> None:
    """Draw a quiet rounded surface that organizes content without boxing it in."""
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            transform=fig.transFigure,
            facecolor=THEME.panel_alt if alternate else THEME.panel,
            edgecolor=THEME.border,
            linewidth=0.65,
            zorder=-1,
        )
    )
    if accent:
        fig.lines.append(
            plt.Line2D(
                [x + 0.014, x + width - 0.014],
                [y + height - 0.004, y + height - 0.004],
                transform=fig.transFigure,
                color=accent,
                linewidth=2.2,
                solid_capstyle="round",
            )
        )


def _date_stamp(payload: SocialPayload) -> str:
    if payload.latest_date:
        try:
            return pd.Timestamp(payload.latest_date).strftime("%b %d, %Y").upper()
        except (TypeError, ValueError):
            return str(payload.latest_date).upper()
    return "LATEST COMPLETE DATE UNAVAILABLE"


def _pill(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    color: str = THEME.observed,
    align: str = "left",
    size: float = 6.2,
) -> None:
    fig.text(
        x,
        y,
        text,
        ha=align,
        va="center",
        fontsize=size,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.7",
            "facecolor": THEME.panel_alt,
            "edgecolor": THEME.border,
            "linewidth": 0.6,
        },
    )


def _header(
    fig: plt.Figure,
    eyebrow: str,
    title: str,
    subtitle: str | None = None,
    *,
    square: bool = False,
    title_size: float | None = None,
    subtitle_width: int | None = None,
) -> None:
    """Set a compact editorial header with a consistent date/context rail."""
    x = 0.07
    fig.lines.append(
        plt.Line2D(
            [x, x + (0.055 if square else 0.035)],
            [0.965, 0.965],
            transform=fig.transFigure,
            color=THEME.observed,
            linewidth=3.0,
            solid_capstyle="round",
        )
    )
    fig.text(x, 0.948, eyebrow.upper(), fontsize=7.1 if square else 7.8, color=THEME.observed, fontweight="bold", va="top")
    size = title_size if title_size is not None else (15.0 if square else 15.8)
    fig.text(x, 0.906, title, fontsize=size, color=THEME.text, fontweight="bold", va="top", linespacing=1.02)
    if subtitle:
        lines = title.count("\n") + 1
        y = 0.838 - (lines - 1) * (0.062 if square else 0.054)
        subtitle_text = textwrap.fill(subtitle, width=subtitle_width or (56 if square else 112))
        fig.text(x, y, subtitle_text, fontsize=8.1 if square else 8.9, color=THEME.muted, va="top", linespacing=1.08)


def _brand(fig: plt.Figure, *, y: float = 0.031) -> None:
    fig.text(0.93, y, THEME.brand, fontsize=8.1, fontweight="bold", color=THEME.observed, ha="right", va="bottom")


def _source(fig: plt.Figure, payload: SocialPayload, *, y: float = 0.031, width: int = 64, x: float = 0.07) -> None:
    source = "Sources: NOAA/NCEI IGRA v2; NWS/NCEP/NCO SDM messages; station master."
    if payload.partial_date:
        source += f" Preliminary archive date excluded: {payload.partial_date}."
    fig.text(x, y, textwrap.fill(source, width=width), fontsize=4.9, color=THEME.muted, va="bottom", linespacing=1.08)


def _footer(fig: plt.Figure, payload: SocialPayload, *, y: float = 0.031, width: int = 64, x: float = 0.07) -> None:
    fig.lines.append(
        plt.Line2D([x, 0.93], [y + 0.039, y + 0.039], transform=fig.transFigure, color=THEME.border, linewidth=0.65)
    )
    _source(fig, payload, y=y, width=width, x=x)
    _brand(fig, y=y)


def _style_axis(ax: plt.Axes, *, grid: str = "y", label_size: float = 7.7) -> None:
    ax.set_facecolor(THEME.panel)
    ax.tick_params(colors=THEME.muted, labelsize=label_size, length=0, width=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME.border)
    ax.spines["bottom"].set_color(THEME.border)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if grid:
        ax.grid(True, axis=grid, color=THEME.grid, linewidth=0.55, alpha=0.58)
    ax.set_axisbelow(True)


def _plot_trend(
    ax: plt.Axes,
    payload: SocialPayload,
    *,
    title: str | None = None,
    annotate_drop: bool = True,
    minimal: bool = False,
) -> None:
    """Render the primary evidence chart with direct labels and recent context."""
    _style_axis(ax, label_size=7.0 if minimal else 7.8)
    if title:
        ax.set_title(title, loc="left", fontsize=9.2 if minimal else 10.2, fontweight="bold", color=THEME.text, pad=10)
    data = payload.series
    if data.empty or data[["observed", "baseline"]].dropna(how="all").empty:
        ax.text(0.5, 0.5, "Time-series data unavailable in the metrics manifest", transform=ax.transAxes, ha="center", va="center", color=THEME.muted, fontsize=8.5, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    valid_dates = data["date"].dropna()
    if not valid_dates.empty:
        recent_start = valid_dates.max() - pd.Timedelta(days=28)
        ax.axvspan(recent_start, valid_dates.max() + pd.Timedelta(days=5), color=THEME.observed, alpha=0.045, zorder=0)
    if {"baseline_low", "baseline_high"}.issubset(data.columns) and not data[["baseline_low", "baseline_high"]].isna().all().all():
        ax.fill_between(
            data["date"],
            data["baseline_low"],
            data["baseline_high"],
            color=THEME.baseline,
            alpha=0.10,
            zorder=0.5,
        )
    ax.plot(data["date"], data["baseline"], color=THEME.baseline, linewidth=1.4 if minimal else 1.7, linestyle=(0, (4, 3)), zorder=3)
    ax.plot(data["date"], data["observed"], color=THEME.observed, linewidth=2.2 if minimal else 2.65, zorder=4)
    below = data["observed"] < data["baseline"]
    ax.fill_between(data["date"], data["observed"], data["baseline"], where=below, color=THEME.deficit_fill, alpha=0.72, interpolate=True, zorder=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Archived soundings / day", color=THEME.muted, fontsize=7.2 if minimal else 7.8, labelpad=7)
    if not valid_dates.empty:
        ax.set_xlim(valid_dates.min() - pd.Timedelta(days=3), valid_dates.max() + pd.Timedelta(days=12))
    ax.text(0.015, 0.965, "\u2501  Observed 7-day average", transform=ax.transAxes, color=THEME.observed, fontsize=6.1 if minimal else 6.6, fontweight="bold", va="top")
    ax.text(0.015, 0.905, "\u254c  Same-date baseline", transform=ax.transAxes, color=THEME.baseline, fontsize=5.9 if minimal else 6.4, va="top")
    if {"baseline_low", "baseline_high"}.issubset(data.columns) and not data[["baseline_low", "baseline_high"]].isna().all().all():
        ax.text(0.015, 0.845, "Band: 2021–2024 range", transform=ax.transAxes, color=THEME.muted, fontsize=5.3 if minimal else 5.8, va="top")

    latest = data.dropna(subset=["date", "observed", "baseline"]).tail(1)
    if not latest.empty:
        row = latest.iloc[0]
        ax.scatter([row["date"]], [row["observed"]], s=20 if minimal else 28, color=THEME.observed, edgecolor=THEME.background, linewidth=0.8, zorder=6)
        ax.scatter([row["date"]], [row["baseline"]], s=14 if minimal else 20, color=THEME.baseline, edgecolor=THEME.background, linewidth=0.7, zorder=5)
        ax.annotate(f"{row['observed']:.1f}", (row["date"], row["observed"]), xytext=(6, -8), textcoords="offset points", color=THEME.observed, fontsize=6.1 if minimal else 6.7, fontweight="bold", va="center")
        ax.annotate(f"{row['baseline']:.1f}", (row["date"], row["baseline"]), xytext=(6, 5), textcoords="offset points", color=THEME.baseline, fontsize=5.8 if minimal else 6.4, va="center")
    if annotate_drop:
        sharp_drop = detect_sharp_drop(data)
        if sharp_drop is not None:
            date, change = sharp_drop
            row = data.loc[data["date"] == date].iloc[-1]
            label = "SHARP JULY DROP" if date.month == 7 else "SHARP RECENT DROP"
            ax.annotate(
                f"{label}\n{change:.1f} / day",
                xy=(date, row["observed"]),
                xytext=(-14, 34),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.0 if minimal else 6.8,
                fontweight="bold",
                color=THEME.deficit,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": THEME.background, "edgecolor": THEME.deficit, "linewidth": 0.8},
                arrowprops={"arrowstyle": "-|>", "color": THEME.deficit, "lw": 1.0, "shrinkA": 3, "shrinkB": 2},
                annotation_clip=True,
                zorder=8,
            )


def _plot_sparkline(ax: plt.Axes, payload: SocialPayload) -> None:
    _style_axis(ax, grid="", label_size=6.5)
    ax.set_facecolor(THEME.panel_alt)
    for spine in ax.spines.values():
        spine.set_visible(False)
    data = payload.series
    if data.empty:
        ax.text(0.5, 0.5, "Time-series unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=THEME.muted)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    dates = data["date"].dropna()
    if not dates.empty:
        ax.axvspan(dates.max() - pd.Timedelta(days=28), dates.max() + pd.Timedelta(days=2), color=THEME.observed, alpha=0.05)
    ax.plot(data["date"], data["baseline"], color=THEME.baseline, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot(data["date"], data["observed"], color=THEME.observed, linewidth=1.75)
    below = data["observed"] < data["baseline"]
    ax.fill_between(data["date"], data["observed"], data["baseline"], where=below, color=THEME.deficit_fill, alpha=0.78, interpolate=True)
    ax.set_xticks([])
    ax.set_yticks([])
    sharp_drop = detect_sharp_drop(data)
    if sharp_drop is not None:
        date, _change = sharp_drop
        row = data.loc[data["date"] == date].iloc[-1]
        ax.scatter([date], [row["observed"]], s=14, color=THEME.deficit, zorder=6)
        ax.annotate(
            "JULY DROP" if date.month == 7 else "RECENT DROP",
            xy=(date, row["observed"]),
            xytext=(-8, 12),
            textcoords="offset points",
            ha="right",
            fontsize=5.4,
            fontweight="bold",
            color=THEME.deficit,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": THEME.background, "edgecolor": THEME.deficit, "linewidth": 0.65},
            arrowprops={"arrowstyle": "-|>", "color": THEME.deficit, "lw": 0.85},
        )


def _plot_window_bars(
    ax: plt.Axes,
    payload: SocialPayload,
    *,
    title: str | None = None,
    label_size: float = 9.0,
    show_xlabel: bool = True,
    highlight_90: bool = True,
) -> None:
    _style_axis(ax, grid="", label_size=label_size)
    if title:
        ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold", color=THEME.text, pad=8)
    windows = payload.windows.copy()
    if windows.empty or "percent" not in windows or "days" not in windows:
        ax.text(0.5, 0.5, "Recent-window data unavailable", transform=ax.transAxes, ha="center", va="center", color=THEME.muted, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    windows = windows.dropna(subset=["days", "percent"]).sort_values("days").reset_index(drop=True)
    values = windows["percent"].to_numpy(dtype=float)
    y = np.arange(len(windows))
    colors = [THEME.amber if highlight_90 and int(days) == 90 else THEME.deficit for days in windows["days"]]
    ax.barh(y, values, height=0.46, color=colors, alpha=0.95)
    ax.axvline(0, color=THEME.baseline, linewidth=0.9, alpha=0.75)
    left = min(-1.0, float(np.nanmin(values)) * 1.36)
    ax.set_xlim(left, 1.25)
    for index, value in enumerate(values):
        ax.text(value + 0.13, index, f"{value:.1f}%", ha="left", va="center", fontsize=label_size, fontweight="bold", color=THEME.background)
        if "deficit" in windows:
            shortfall = abs(_safe_float(windows.loc[index, "deficit"]))
            if not math.isnan(shortfall):
                ax.text(0.98, index, f"{shortfall:,.0f} fewer", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=max(5.4, label_size - 2.1), color=THEME.muted)
    ax.set_yticks(y, [f"{int(days)} days" for days in windows["days"]], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Percent difference from expected archive volume", fontsize=7.2, color=THEME.muted, labelpad=7)


def _iter_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return coordinates
    if geometry_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    return []


def _miller_xy(longitudes: Any, latitudes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude coordinates with a Miller cylindrical projection."""
    lon_radians = np.deg2rad(np.asarray(longitudes, dtype=float))
    lat_radians = np.deg2rad(np.clip(np.asarray(latitudes, dtype=float), -89.5, 89.5))
    y = 1.25 * np.log(np.tan((np.pi / 4.0) + (0.4 * lat_radians)))
    return lon_radians, y


def _draw_conus_base(ax: plt.Axes, states_geojson_path: Path | None) -> None:
    ax.set_facecolor(THEME.water)
    if states_geojson_path is not None and states_geojson_path.exists():
        try:
            geojson = json.loads(states_geojson_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            geojson = {"features": []}
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry", {})
            if not isinstance(geometry, dict):
                continue
            for ring in _iter_rings(geometry):
                if not ring:
                    continue
                longitudes = [float(point[0]) for point in ring]
                latitudes = [float(point[1]) for point in ring]
                if max(longitudes) < -126 or min(longitudes) > -66 or max(latitudes) < 24 or min(latitudes) > 50:
                    continue
                xs, ys = _miller_xy(longitudes, latitudes)
                ax.fill(xs, ys, facecolor=THEME.land, edgecolor="#3C627B", linewidth=0.42, zorder=0)
    west, south = _miller_xy([-126], [24])
    east, north = _miller_xy([-66], [50])
    ax.set_xlim(float(west[0]), float(east[0]))
    ax.set_ylim(float(south[0]), float(north[0]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(THEME.border)


def _plot_station_map(
    ax: plt.Axes,
    payload: SocialPayload,
    states_geojson_path: Path | None,
    *,
    title: str | None = None,
    show_legend: bool = True,
    marker_size: float | None = None,
    legend_font_size: float = 5.0,
    legend_marker_size: float = 5.5,
    legend_columns: int = 2,
    show_empty_legend_entries: bool = False,
) -> None:
    _draw_conus_base(ax, states_geojson_path)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=THEME.text, pad=8)
    stations = payload.stations
    needed = {"latitude", "longitude", "status"}
    if stations.empty or not needed.issubset(stations.columns):
        ax.text(0.5, 0.5, "Station-status points unavailable\nin the metrics manifest", transform=ax.transAxes, ha="center", va="center", color=THEME.muted, fontsize=8.3)
        return
    style = {
        "available / no issue": (THEME.clean, "No issue reported", 13),
        "missing / problem": (THEME.deficit, "NCO-reported problem", 28),
        "partial / quality": (THEME.amber, "Partial / quality issue", 24),
        "unknown": (THEME.unknown, "Status unknown", 13),
    }
    handles = []
    for status, (color, label, size) in style.items():
        subset = stations[stations["status"] == status]
        if subset.empty and not show_empty_legend_entries:
            continue
        size = marker_size if marker_size is not None else size
        is_issue = status in {"missing / problem", "partial / quality"}
        if not subset.empty:
            xs, ys = _miller_xy(subset["longitude"], subset["latitude"])
            ax.scatter(
                xs,
                ys,
                s=size,
                c=color,
                edgecolor=THEME.text if is_issue else THEME.background,
                linewidth=0.55 if is_issue else 0.35,
                alpha=0.96,
                zorder=4 if is_issue else 3,
            )
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=THEME.text if is_issue else THEME.background, markersize=legend_marker_size, label=label))
    if handles and show_legend:
        ax.legend(handles=handles, loc="lower left", frameon=True, facecolor=THEME.background, edgecolor=THEME.border, labelcolor=THEME.muted, fontsize=legend_font_size, ncol=legend_columns, handletextpad=0.6, borderpad=0.75, columnspacing=1.0)


def _station_counts(payload: SocialPayload) -> tuple[int, int, int]:
    if payload.stations.empty or "status" not in payload.stations:
        return 0, 0, 0
    statuses = payload.stations["status"].fillna("unknown")
    clean = int((statuses == "available / no issue").sum())
    issue = int(statuses.isin(["missing / problem", "partial / quality"]).sum())
    return clean, issue, int(len(statuses))


def _stat_card(fig: plt.Figure, x: float, y: float, width: float, height: float, label: str, value: str, detail: str, *, color: str = THEME.amber, value_size: float = 22) -> None:
    _card(fig, x, y, width, height, accent=color)
    pad = min(0.020, width * 0.065)
    fig.text(x + pad, y + height - 0.020, label.upper(), fontsize=5.5, color=THEME.muted, fontweight="bold", va="top")
    fig.text(x + pad, y + 0.006, value, fontsize=value_size, color=color, fontweight="bold", va="bottom")
    if detail:
        fig.text(x + width - pad, y + 0.012, textwrap.shorten(detail, width=32, placeholder="…"), fontsize=4.8, color=THEME.text, ha="right", va="bottom")


def _hero_number(fig: plt.Figure, payload: SocialPayload, *, x: float, y: float, size: float = 64) -> None:
    value = _format_percent(payload.gap_percent)
    fig.text(x, y, value, fontsize=size, color=THEME.deficit, fontweight="bold", va="top")
    arrow = FancyArrowPatch((x + 0.76, y - 0.016), (x + 0.76, y - 0.125), transform=fig.transFigure, arrowstyle="simple", mutation_scale=17, color=THEME.deficit, linewidth=0)
    fig.add_artist(arrow)


def create_hero_big_number(payload: SocialPayload, output_path: Path, *, states_geojson_path: Path | None, dpi_scale: float = 1.0) -> None:
    """Primary vertical post: one metric, one trend, and one location snapshot."""
    fig = _figure(1080, 1920)
    _header(fig, "CONUS UPPER-AIR DATA WATCH", "THE ARCHIVE SIGNAL\nDROPPED SHARPLY", "Latest complete records compared with the 2021–2024 same-date baseline", square=True, title_size=15.2, subtitle_width=48)
    _pill(fig, 0.93, 0.949, _date_stamp(payload), align="right", size=5.3)
    _hero_number(fig, payload, x=0.07, y=0.700, size=58)
    fig.text(0.07, 0.570, "7-day archive gap", fontsize=12.2, color=THEME.text, fontweight="bold", va="top")
    fig.text(0.07, 0.538, "versus the same-date seasonal baseline", fontsize=7.6, color=THEME.muted, va="top")

    _card(fig, 0.07, 0.390, 0.86, 0.120, alternate=True, accent=THEME.deficit)
    fig.text(0.095, 0.492, "RECENT ARCHIVE TREND", fontsize=6.3, fontweight="bold", color=THEME.text, va="top")
    _plot_sparkline(fig.add_axes([0.095, 0.408, 0.81, 0.066]), payload)

    recent_90 = _window_row(payload)
    shortfall = abs(_safe_float(recent_90.get("deficit"))) if recent_90 is not None else float("nan")
    pct_90 = _safe_float(recent_90.get("percent")) if recent_90 is not None else float("nan")
    _stat_card(fig, 0.07, 0.292, 0.41, 0.068, "90-day archive shortfall", _format_number(shortfall), _format_percent(pct_90), color=THEME.amber, value_size=16)
    _stat_card(fig, 0.52, 0.292, 0.41, 0.068, "NCO-reported issue statuses", _format_number(payload.issue_count), "latest message cycle", color=THEME.deficit, value_size=16)

    _card(fig, 0.07, 0.095, 0.86, 0.170)
    fig.text(0.095, 0.248, "WHERE ISSUES WERE REPORTED", fontsize=6.2, fontweight="bold", color=THEME.text, va="top")
    _plot_station_map(fig.add_axes([0.095, 0.125, 0.81, 0.108]), payload, states_geojson_path, show_legend=False)
    fig.text(0.095, 0.111, "● No issue reported", fontsize=5.4, color=THEME.clean, va="top")
    fig.text(0.50, 0.111, "● NCO-reported problem", fontsize=5.4, color=THEME.deficit, va="top")
    fig.text(0.07, 0.078, "DATA-AVAILABILITY DIAGNOSTIC ONLY  ·  NOT A CAUSATION CLAIM", fontsize=5.7, color=THEME.muted, fontweight="bold", va="top")
    _footer(fig, payload, y=0.018, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_carousel_hook(payload: SocialPayload, output_path: Path, *, dpi_scale: float = 1.0) -> None:
    fig = _figure(1080, 1080)
    _header(fig, "01 / THE SIGNAL", "UPPER-AIR ARCHIVE\nAVAILABILITY FELL", "A fast read of the latest complete CONUS archive records", square=True, title_size=14.6)
    _pill(fig, 0.93, 0.950, _date_stamp(payload), align="right", size=5.1)
    _hero_number(fig, payload, x=0.07, y=0.685, size=50)
    fig.text(0.07, 0.475, "below the same-date baseline", fontsize=10.0, color=THEME.text, fontweight="bold", va="top")
    fig.text(0.07, 0.438, "7-day archived-soundings average", fontsize=7.1, color=THEME.muted, va="top")

    _card(fig, 0.07, 0.305, 0.86, 0.105, alternate=True)
    fig.text(0.10, 0.387, "LATEST COMPLETE 7-DAY AVERAGE", fontsize=5.7, color=THEME.muted, fontweight="bold", va="top")
    fig.text(0.10, 0.350, _format_number(payload.observed, decimals=1), fontsize=13.5, color=THEME.observed, fontweight="bold", va="top")
    fig.text(0.29, 0.352, "observed / day", fontsize=5.7, color=THEME.muted, va="top")
    fig.text(0.57, 0.350, _format_number(payload.expected, decimals=1), fontsize=13.5, color=THEME.baseline, fontweight="bold", va="top")
    fig.text(0.77, 0.352, "expected / day", fontsize=5.7, color=THEME.muted, va="top")

    _card(fig, 0.07, 0.160, 0.86, 0.115, accent=THEME.deficit)
    _plot_sparkline(fig.add_axes([0.10, 0.180, 0.80, 0.068]), payload)
    recent_90 = _window_row(payload)
    shortfall = abs(_safe_float(recent_90.get("deficit"))) if recent_90 is not None else float("nan")
    fig.text(0.07, 0.128, f"{_format_number(shortfall)} fewer archive records over 90 days", fontsize=7.0, fontweight="bold", color=THEME.amber, va="top")
    fig.text(0.07, 0.099, "Availability signal only — not a confirmed launch count.", fontsize=5.7, color=THEME.muted, va="top")
    _footer(fig, payload, y=0.028, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_carousel_trend(payload: SocialPayload, output_path: Path, *, dpi_scale: float = 1.0) -> None:
    fig = _figure(1080, 1080)
    _header(fig, "02 / THE TREND", "THE RECENT DIVERGENCE", "Observed 7-day archive average versus the 2021–2024 same-date baseline", square=True, title_size=15.0, subtitle_width=52)
    _pill(fig, 0.93, 0.950, _format_percent(payload.gap_percent), align="right", color=THEME.deficit, size=6.0)
    _card(fig, 0.07, 0.190, 0.86, 0.510, accent=THEME.deficit)
    fig.text(0.105, 0.665, "ARCHIVED SOUNDINGS / DAY", fontsize=8.4, fontweight="bold", color=THEME.text, va="top")
    _plot_trend(fig.add_axes([0.145, 0.245, 0.745, 0.360]), payload, title=None, annotate_drop=True)
    fig.text(0.07, 0.145, "Coral fill marks observed archive availability below the baseline.", fontsize=6.3, color=THEME.muted, va="top")
    _footer(fig, payload, y=0.028, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_carousel_map(payload: SocialPayload, output_path: Path, *, states_geojson_path: Path | None, dpi_scale: float = 1.0) -> None:
    fig = _figure(1080, 1080)
    _header(fig, "03 / THE STATIONS", "WHERE STATUS ISSUES\nWERE REPORTED", "Latest NCO operational-message reporting; state borders shown for context", square=True, title_size=14.5, subtitle_width=52)
    clean_count, mapped_issue_count, total_count = _station_counts(payload)
    _card(fig, 0.05, 0.255, 0.90, 0.445, alternate=True)
    _plot_station_map(fig.add_axes([0.07, 0.295, 0.86, 0.350]), payload, states_geojson_path, show_legend=False)
    issue_value = payload.issue_count if payload.issue_count is not None else mapped_issue_count
    map_cards = (
        (0.07, "MAPPED STATIONS", _format_number(total_count), THEME.observed),
        (0.365, "NO ISSUE REPORTED", _format_number(clean_count), THEME.clean),
        (0.66, "ISSUE STATUSES", _format_number(issue_value), THEME.deficit),
    )
    for x, label, value, color in map_cards:
        _card(fig, x, 0.145, 0.27, 0.080, accent=color)
        fig.text(x + 0.022, 0.213, label, fontsize=4.9, color=THEME.muted, fontweight="bold", va="top")
        fig.text(x + 0.022, 0.150, value, fontsize=10.8, color=color, fontweight="bold", va="bottom")
    fig.text(0.07, 0.118, "Operational-message status is not a confirmed IGRA archive total.", fontsize=5.9, color=THEME.muted, va="top")
    _footer(fig, payload, y=0.028, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_carousel_gaps(payload: SocialPayload, output_path: Path, *, dpi_scale: float = 1.0) -> None:
    fig = _figure(1080, 1080)
    _header(fig, "04 / THE WINDOWS", "THE ARCHIVE GAP\nIS NOT ONE-DAY NOISE", "Recent totals compared with expected archive volume over four windows", square=True, title_size=14.5, subtitle_width=52)
    _card(fig, 0.07, 0.190, 0.86, 0.510, accent=THEME.amber)
    _plot_window_bars(fig.add_axes([0.205, 0.255, 0.67, 0.365]), payload, title="DIFFERENCE FROM EXPECTED", label_size=7.4, show_xlabel=False)
    fig.text(0.095, 0.220, "Amber highlights the 90-day window used in the lead supporting statistic.", fontsize=5.7, color=THEME.muted, va="top")
    fig.text(0.07, 0.142, "Archive availability comparison — not a model-skill metric.", fontsize=6.2, color=THEME.muted, va="top")
    _footer(fig, payload, y=0.028, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_carousel_caveats(payload: SocialPayload, output_path: Path, *, dpi_scale: float = 1.0) -> None:
    fig = _figure(1080, 1080)
    _header(fig, "05 / INTERPRETATION", "WHAT THIS GRAPHIC\nDOES — AND DOES NOT — SAY", "Read this before sharing the archive and station-status signals", square=True, title_size=13.7, subtitle_width=52)
    y_positions = (0.550, 0.380, 0.210)
    accents = (THEME.observed, THEME.amber, THEME.deficit)
    for index, (caveat, y, accent) in enumerate(zip(EXACT_CAVEATS, y_positions, accents), start=1):
        _card(fig, 0.07, y, 0.86, 0.140, alternate=index == 2, accent=accent)
        fig.text(0.10, y + 0.097, f"0{index}", fontsize=6.7, color=accent, fontweight="bold", va="top")
        fig.text(0.18, y + 0.097, textwrap.fill(caveat, width=48), fontsize=7.1, color=THEME.text, va="top", linespacing=1.12)
    _pill(fig, 0.07, 0.155, "DATA-AVAILABILITY DIAGNOSTIC ONLY", color=THEME.amber, size=6.0)
    _footer(fig, payload, y=0.028, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def _single_series_panel(ax: plt.Axes, payload: SocialPayload, column: str, title: str | None, color: str, *, baseline_panel: bool) -> None:
    _style_axis(ax, label_size=7.1)
    data = payload.series
    if title:
        ax.set_title(title, loc="left", fontsize=8.2, color=THEME.text, fontweight="bold", pad=8, linespacing=1.02)
    if data.empty or column not in data:
        ax.text(0.5, 0.5, "Time-series data unavailable", transform=ax.transAxes, ha="center", va="center", color=THEME.muted, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    values = data[column]
    ax.plot(data["date"], values, color=color, linewidth=2.5, zorder=4)
    if not baseline_panel and "baseline" in data:
        ax.plot(data["date"], data["baseline"], color=THEME.baseline, linewidth=1.1, linestyle="--")
        below = data["observed"] < data["baseline"]
        ax.fill_between(data["date"], data["observed"], data["baseline"], where=below, color=THEME.deficit_fill, alpha=0.75, interpolate=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Archived soundings / day", color=THEME.muted, fontsize=7.0)
    dates = data["date"].dropna()
    if not dates.empty:
        ax.set_xlim(dates.min() - pd.Timedelta(days=3), dates.max() + pd.Timedelta(days=10))
    latest = data.dropna(subset=["date", column]).tail(1)
    if not latest.empty:
        row = latest.iloc[0]
        ax.scatter([row["date"]], [row[column]], s=30, color=color, edgecolor=THEME.background, linewidth=0.8, zorder=6)
        ax.annotate(f"{row[column]:.1f}", (row["date"], row[column]), xytext=(6, 0), textcoords="offset points", color=color, fontsize=7.0, fontweight="bold", va="center")


def create_split_expected_vs_reality(payload: SocialPayload, output_path: Path, *, dpi_scale: float = 1.0) -> None:
    fig = _figure(1920, 1080)
    _header(fig, "CONUS UPPER-AIR DATA WATCH", "EXPECTED BASELINE  /  OBSERVED ARCHIVE", "Same dates, same y-scale, two different availability signals", title_size=15.6)
    _pill(fig, 0.93, 0.950, _date_stamp(payload), align="right", size=5.8)

    _card(fig, 0.05, 0.180, 0.405, 0.535, accent=THEME.baseline)
    _card(fig, 0.545, 0.180, 0.405, 0.535, accent=THEME.observed)
    fig.text(0.075, 0.675, "EXPECTED", fontsize=7.0, fontweight="bold", color=THEME.baseline, va="top")
    fig.text(0.075, 0.625, _format_number(payload.expected, decimals=1), fontsize=24.0, fontweight="bold", color=THEME.baseline, va="top")
    fig.text(0.075, 0.545, "soundings / day  ·  same-date baseline", fontsize=6.1, color=THEME.muted, va="top")
    fig.text(0.57, 0.675, "OBSERVED", fontsize=7.0, fontweight="bold", color=THEME.observed, va="top")
    fig.text(0.57, 0.625, _format_number(payload.observed, decimals=1), fontsize=24.0, fontweight="bold", color=THEME.observed, va="top")
    fig.text(0.57, 0.545, "soundings / day  ·  7-day archive average", fontsize=6.1, color=THEME.muted, va="top")

    left = fig.add_axes([0.075, 0.245, 0.35, 0.255])
    right = fig.add_axes([0.57, 0.245, 0.35, 0.255])
    _single_series_panel(left, payload, "baseline", None, THEME.baseline, baseline_panel=True)
    _single_series_panel(right, payload, "observed", None, THEME.observed, baseline_panel=False)
    right.set_ylabel("")
    if not payload.series.empty:
        all_values = pd.concat([payload.series["observed"], payload.series["baseline"]]).dropna()
        if not all_values.empty:
            padding = max(1.0, (all_values.max() - all_values.min()) * 0.12)
            ymin, ymax = all_values.min() - padding, all_values.max() + padding
            left.set_ylim(ymin, ymax)
            right.set_ylim(ymin, ymax)
    fig.text(0.50, 0.615, _format_percent(payload.gap_percent), fontsize=13.0, fontweight="bold", color=THEME.deficit, ha="center", va="center")
    fig.text(0.50, 0.568, "LATEST GAP", fontsize=5.1, fontweight="bold", color=THEME.muted, ha="center", va="center")
    fig.text(0.05, 0.128, "Coral shading on the observed panel marks the archive deficit relative to baseline.", fontsize=6.2, color=THEME.muted, va="top")
    fig.text(0.95, 0.128, "DATA-AVAILABILITY DIAGNOSTIC ONLY", fontsize=6.2, color=THEME.amber, fontweight="bold", ha="right", va="top")
    _footer(fig, payload, y=0.036, width=106, x=0.05)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_minimalist_trend(payload: SocialPayload, output_path: Path, *, dpi_scale: float = 1.0) -> None:
    fig = _figure(1080, 1080)
    _header(fig, "CONUS UPPER-AIR DATA WATCH", "ARCHIVE TREND", "A chart-first view of the 7-day average and same-date baseline", square=True, title_size=15.0)
    _pill(fig, 0.93, 0.950, _format_percent(payload.gap_percent), align="right", color=THEME.deficit, size=6.2)
    _card(fig, 0.07, 0.205, 0.86, 0.500, accent=THEME.observed)
    _plot_trend(fig.add_axes([0.125, 0.285, 0.765, 0.350]), payload, title=None, annotate_drop=True, minimal=True)
    fig.text(0.07, 0.160, "Observed archive availability remains below the seasonal baseline.", fontsize=6.2, color=THEME.muted, va="top")
    _footer(fig, payload, y=0.028, width=62)
    _save(fig, output_path, dpi_scale=dpi_scale)


def create_original_dashboard_style(payload: SocialPayload, output_path: Path, *, states_geojson_path: Path | None, dpi_scale: float = 1.0) -> None:
    """Comprehensive desktop one-pager, rebuilt with the social visual hierarchy."""
    fig = _figure(2400, 1500)
    _header(fig, "CONUS UPPER-AIR DATA WATCH", "NETWORK AVAILABILITY MONITOR", "Archive comparison, recent-window gaps, and NCO operational-message station status", title_size=15.6)
    _pill(fig, 0.95, 0.950, f"ARCHIVE THROUGH  {_date_stamp(payload)}", align="right", size=6.0)
    recent_90 = _window_row(payload)
    shortfall = abs(_safe_float(recent_90.get("deficit"))) if recent_90 is not None else float("nan")
    pct_90 = _safe_float(recent_90.get("percent")) if recent_90 is not None else float("nan")
    _stat_card(fig, 0.05, 0.705, 0.28, 0.095, "current 7-day archive gap", _format_percent(payload.gap_percent), "vs same-date baseline", color=THEME.deficit, value_size=19)
    _stat_card(fig, 0.36, 0.705, 0.28, 0.095, "90-day archive shortfall", _format_number(shortfall), _format_percent(pct_90), color=THEME.amber, value_size=19)
    _stat_card(fig, 0.67, 0.705, 0.28, 0.095, "NCO-reported issue statuses", _format_number(payload.issue_count), payload.nco_cycle or "latest message cycle", color=THEME.deficit, value_size=19)

    _card(fig, 0.05, 0.385, 0.575, 0.260, accent=THEME.deficit)
    fig.text(0.075, 0.618, "ARCHIVED SOUNDINGS VS SAME-DATE BASELINE", fontsize=7.9, fontweight="bold", color=THEME.text, va="top")
    _plot_trend(fig.add_axes([0.090, 0.425, 0.510, 0.155]), payload, title=None, annotate_drop=True)

    _card(fig, 0.67, 0.385, 0.28, 0.260, accent=THEME.amber)
    fig.text(0.695, 0.618, "RECENT ARCHIVE WINDOWS", fontsize=7.9, fontweight="bold", color=THEME.text, va="top")
    _plot_window_bars(
        fig.add_axes([0.720, 0.430, 0.200, 0.155]),
        payload,
        title=None,
        label_size=6.3,
        show_xlabel=False,
        highlight_90=False,
    )

    _card(fig, 0.05, 0.095, 0.575, 0.235)
    fig.text(0.075, 0.305, "LATEST NCO STATION-STATUS SNAPSHOT", fontsize=7.9, fontweight="bold", color=THEME.text, va="top")
    _plot_station_map(fig.add_axes([0.075, 0.125, 0.385, 0.165]), payload, states_geojson_path, show_legend=False)
    clean_count, mapped_issue_count, total_count = _station_counts(payload)
    fig.text(0.475, 0.255, "STATUS COVERAGE", fontsize=5.2, color=THEME.muted, fontweight="bold", va="top")
    fig.text(0.475, 0.220, f"{mapped_issue_count}", fontsize=13.5, color=THEME.deficit, fontweight="bold", va="top")
    fig.text(
        0.520,
        0.219,
        "reported issue\nstatuses",
        fontsize=5.0,
        color=THEME.text,
        fontweight="bold",
        va="top",
        linespacing=1.05,
    )
    fig.text(
        0.475,
        0.165,
        f"{clean_count} no issue reported  /  {total_count} mapped",
        fontsize=4.8,
        color=THEME.muted,
        va="top",
    )
    fig.text(0.095, 0.112, "● No issue reported", fontsize=5.0, color=THEME.clean, va="top")
    fig.text(0.235, 0.112, "● NCO-reported problem", fontsize=5.0, color=THEME.deficit, va="top")

    _card(fig, 0.67, 0.095, 0.28, 0.235, alternate=True, accent=THEME.observed)
    fig.text(0.695, 0.300, "INTERPRETATION GUARDRAILS", fontsize=7.5, fontweight="bold", color=THEME.observed, va="top")
    for index, (caveat, y) in enumerate(zip(EXACT_CAVEATS, (0.255, 0.195, 0.135)), start=1):
        fig.text(0.695, y, f"0{index}", fontsize=5.5, color=THEME.amber, fontweight="bold", va="top")
        wrapped = textwrap.fill(caveat, width=42)
        fig.text(0.724, y, wrapped, fontsize=4.8, color=THEME.text, va="top", linespacing=1.08)
    _footer(fig, payload, y=0.028, width=130, x=0.05)
    _save(fig, output_path, dpi_scale=dpi_scale)


def render_social_graphics_from_manifest(
    manifest_path: Path,
    output_dir: Path | None = None,
    *,
    states_geojson_path: Path | None = None,
    dpi_scale: float = 1.0,
) -> tuple[SocialPayload, list[Path]]:
    """Render the whole suite from a manifest, including graceful empty states."""
    payload = load_social_payload(manifest_path)
    destination = output_dir or manifest_path.parent
    paths = [destination / filename for filename in OUTPUT_FILENAMES]
    create_hero_big_number(payload, paths[0], states_geojson_path=states_geojson_path, dpi_scale=dpi_scale)
    create_carousel_hook(payload, paths[1], dpi_scale=dpi_scale)
    create_carousel_trend(payload, paths[2], dpi_scale=dpi_scale)
    create_carousel_map(payload, paths[3], states_geojson_path=states_geojson_path, dpi_scale=dpi_scale)
    create_carousel_gaps(payload, paths[4], dpi_scale=dpi_scale)
    create_carousel_caveats(payload, paths[5], dpi_scale=dpi_scale)
    create_split_expected_vs_reality(payload, paths[6], dpi_scale=dpi_scale)
    create_minimalist_trend(payload, paths[7], dpi_scale=dpi_scale)
    create_original_dashboard_style(payload, paths[8], states_geojson_path=states_geojson_path, dpi_scale=dpi_scale)
    return payload, paths


def build_social_package(inputs: MonitorInputs, output_dir: Path, *, dpi_scale: float = 1.0) -> tuple[WatchMetrics, list[Path], Path]:
    """Create the reusable manifest, then render every social asset from it."""
    metrics = calculate_metrics(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / filename for filename in OUTPUT_FILENAMES]
    manifest_path = output_dir / "upper_air_data_watch_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(metrics, paths, inputs), indent=2) + "\n", encoding="utf-8")
    _payload, rendered_paths = render_social_graphics_from_manifest(
        manifest_path,
        output_dir,
        states_geojson_path=inputs.states_geojson_path,
        dpi_scale=dpi_scale,
    )
    return metrics, rendered_paths, manifest_path
