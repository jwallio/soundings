"""Validated, read-only data model for the Upper-Air Streamlit dashboard."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .social_graphics import (
    EXACT_CAVEATS,
    MonitorInputs,
    SocialPayload,
    calculate_metrics,
    load_monitor_inputs,
    load_social_payload,
)


MANIFEST_PATH = Path("outputs/upper_air_network_monitor/social/upper_air_data_watch_manifest.json")
STATION_DEFICITS_PATH = Path("outputs/conus_balloon_launches_station_deficits.csv")
IGRA_METADATA_PATH = Path("outputs/conus_balloon_launches_metadata.json")
SPC_STATUS_PATH = Path("data/spc_sounding_availability.csv")
REFRESH_STATUS_PATH = Path("data/upper_air_refresh_status.json")
BASELINE_YEARS = (2021, 2022, 2023, 2024)
ARCHIVE_DETAIL_START_DATE = pd.Timestamp("2021-01-01")


@dataclass(frozen=True)
class DashboardSnapshot:
    """All durable inputs needed by the app at one point in time."""

    repo_root: Path
    payload: SocialPayload
    nco: pd.DataFrame
    issues: pd.DataFrame
    stations: pd.DataFrame
    station_deficits: pd.DataFrame
    igra_metadata: dict[str, object]
    source_status: pd.DataFrame
    manifest_used: bool
    refresh_status: dict[str, object] = field(default_factory=dict)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def enrich_archive_variability(
    series: pd.DataFrame,
    igra: pd.DataFrame,
    baseline_years: Iterable[int] = BASELINE_YEARS,
) -> pd.DataFrame:
    """Attach the same-date historical min/max range to the current series."""
    result = series.copy()
    required = {"date", "year", "launches_7d_avg"}
    if result.empty or igra.empty or not required.issubset(igra.columns):
        return result
    history = igra.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["year"] = pd.to_numeric(history["year"], errors="coerce")
    history["launches_7d_avg"] = pd.to_numeric(history["launches_7d_avg"], errors="coerce")
    years = {int(value) for value in baseline_years}
    history = history[history["year"].isin(years)].dropna(subset=["date", "launches_7d_avg"])
    if history.empty:
        return result
    history["month_day"] = history["date"].dt.strftime("%m-%d")
    band = (
        history.groupby("month_day", as_index=False)["launches_7d_avg"]
        .agg(baseline_low="min", baseline_high="max", baseline_year_count="count")
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["month_day"] = result["date"].dt.strftime("%m-%d")
    return result.merge(band, on="month_day", how="left").drop(columns="month_day")


def archive_detail_series_from_igra(
    igra: pd.DataFrame,
    latest_complete_date: object,
    start_date: object = ARCHIVE_DETAIL_START_DATE,
) -> pd.DataFrame:
    """Build the dashboard detail series from validated IGRA rows since Jan. 2021."""
    columns = ["date", "observed", "baseline", "daily"]
    required = {"date", "launches", "launches_7d_avg", "baseline_5yr_avg"}
    if igra.empty or not required.issubset(igra.columns):
        return pd.DataFrame(columns=columns)

    data = igra.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("launches", "launches_7d_avg", "baseline_5yr_avg"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(latest_complete_date, errors="coerce")
    # The earliest historical year can precede the computed comparison
    # baseline. Keep those observed rows so event maxima remain explorable;
    # baseline-dependent traces simply remain blank until coverage begins.
    data = data.dropna(subset=["date", "launches", "launches_7d_avg"])
    if pd.notna(start):
        data = data[data["date"] >= start]
    if pd.notna(end):
        data = data[data["date"] <= end]

    return (
        data.rename(
            columns={
                "launches": "daily",
                "launches_7d_avg": "observed",
                "baseline_5yr_avg": "baseline",
            }
        )[columns]
        .sort_values("date")
        .reset_index(drop=True)
    )


def prepare_nco(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
    for column in ("conus_count", "alaskan_count", "canadian_count", "mexican_count", "caribbean_count", "pacific_count"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if {"cycle_date_utc", "cycle_hour"}.issubset(data.columns):
        hour = data["cycle_hour"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        data["cycle_hour"] = hour
        data["cycle_dt"] = pd.to_datetime(
            data["cycle_date_utc"].astype(str) + " " + hour + ":00",
            errors="coerce",
            utc=True,
        )
    else:
        data["cycle_dt"] = pd.NaT
    if "message_time_utc" in data:
        data["message_dt"] = pd.to_datetime(data["message_time_utc"], errors="coerce", utc=True)
    if "model" in data:
        data["model"] = data["model"].fillna("NCO").astype(str).str.upper()
    return data.sort_values([column for column in ("cycle_dt", "message_dt") if column in data]).reset_index(drop=True)


NCO_INGEST_MODELS = ("GFS", "NAM", "NCEP")


def expected_nco_reports_by_date(
    stations: pd.DataFrame,
    dates: Iterable[object],
) -> pd.Series:
    """Return the date-effective expected CONUS report inventory.

    The station master is the canonical expected inventory.  If a future
    station-master export exposes effective start/end dates, those dates are
    honored per day.  The current export has a single ``active_expected``
    flag, so its configured active-CONUS count is used consistently rather
    than substituting a historical maximum or today's observed count.
    """
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).normalize()
    date_index = date_index[~date_index.isna()]
    if len(date_index) == 0:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
    if stations.empty:
        return pd.Series(np.nan, index=date_index, dtype="float64")

    data = stations.copy()
    active_values = data.get("active_expected", pd.Series(True, index=data.index))
    active = active_values.astype(str).str.lower().isin({"true", "1", "yes"})
    data = data.loc[active].copy()
    if data.empty:
        return pd.Series(np.nan, index=date_index, dtype="float64")

    start_column = next(
        (column for column in ("effective_start_date", "active_start_date", "expected_start_date") if column in data),
        None,
    )
    end_column = next(
        (column for column in ("effective_end_date", "active_end_date", "expected_end_date") if column in data),
        None,
    )
    starts = pd.to_datetime(data[start_column], errors="coerce").dt.normalize() if start_column else None
    ends = pd.to_datetime(data[end_column], errors="coerce").dt.normalize() if end_column else None

    counts: list[float] = []
    for date in date_index:
        eligible = pd.Series(True, index=data.index)
        if starts is not None:
            eligible &= starts.isna() | starts.le(date)
        if ends is not None:
            eligible &= ends.isna() | ends.ge(date)
        counts.append(float(eligible.sum()))
    return pd.Series(counts, index=date_index, dtype="float64")


def latest_complete_nco_date(frame: pd.DataFrame, now: object | None = None) -> pd.Timestamp | pd.NaT:
    """Return the latest fully dated NCO day, excluding a current UTC day."""
    data = prepare_nco(frame)
    if data.empty or "cycle_dt" not in data:
        return pd.NaT
    dates = data["cycle_dt"].dropna().dt.tz_convert(None).dt.normalize()
    if dates.empty:
        return pd.NaT
    today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize() if now is None else pd.Timestamp(now).tz_localize(None).normalize()
    prior = dates[dates < today]
    return prior.max() if not prior.empty else dates.max()


def nco_daily_ingest(
    frame: pd.DataFrame,
    stations: pd.DataFrame,
    models: Iterable[str] = NCO_INGEST_MODELS,
    cycle_hours: Iterable[str | int] | None = None,
) -> pd.DataFrame:
    """Combine valid NCO model/cycle records into weighted daily rates.

    GFS, NAM, and NCEP rows are alternative operational-message products for
    a cycle, not three copies of one station inventory.  Each present,
    non-negative record contributes one applicable numerator/denominator
    pair.  A repeated message for the same cycle/product is reduced to the
    latest message so it cannot inflate the daily rate.
    """
    columns = ["date", "received", "expected", "percent", "available_rows", *[f"{m.lower()}_count" for m in NCO_INGEST_MODELS]]
    data = prepare_nco(frame)
    if data.empty or not {"cycle_dt", "conus_count", "model"}.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    data = data[data["model"].astype(str).str.upper().isin({str(m).upper() for m in models})].copy()
    if cycle_hours is not None:
        requested_hours = {str(value).replace("Z", "").zfill(2) for value in cycle_hours}
        data = data[data["cycle_dt"].dt.hour.astype(str).str.zfill(2).isin(requested_hours)].copy()
    data["conus_count"] = pd.to_numeric(data["conus_count"], errors="coerce")
    data["date"] = data["cycle_dt"].dt.tz_convert(None).dt.normalize()
    data = data.dropna(subset=["date", "conus_count"])
    data = data[data["conus_count"].ge(0)]
    if data.empty:
        return pd.DataFrame(columns=columns)

    sort_columns = ["cycle_dt"]
    if "message_dt" in data:
        sort_columns.append("message_dt")
    data = data.sort_values(sort_columns).drop_duplicates(
        subset=["cycle_dt", "model"], keep="last"
    )

    expected = expected_nco_reports_by_date(stations, data["date"].unique())
    rows: list[dict[str, object]] = []
    for date, group in data.sort_values("cycle_dt").groupby("date", sort=True):
        expected_per_row = expected.get(pd.Timestamp(date), np.nan)
        received = float(group["conus_count"].sum())
        expected_total = float(expected_per_row * len(group)) if pd.notna(expected_per_row) else np.nan
        row: dict[str, object] = {
            "date": pd.Timestamp(date),
            "received": received,
            "expected": expected_total,
            "percent": received / expected_total * 100.0 if expected_total and math.isfinite(expected_total) else np.nan,
            "available_rows": int(len(group)),
        }
        latest_by_model = group.groupby(group["model"].astype(str).str.upper(), sort=False).tail(1)
        for model in NCO_INGEST_MODELS:
            matches = latest_by_model[latest_by_model["model"].astype(str).str.upper().eq(model)]
            row[f"{model.lower()}_count"] = float(matches.iloc[-1]["conus_count"]) if not matches.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def nco_lookback_metrics(
    daily: pd.DataFrame,
    windows: Iterable[int] = (7, 14, 30, 90),
    end_date: object | None = None,
) -> pd.DataFrame:
    """Calculate weighted rates and equal-period percentage-point deltas."""
    columns = ["days", "current_percent", "previous_percent", "delta_pp", "current_days", "previous_days"]
    if daily.empty or not {"date", "received", "expected"}.issubset(daily.columns):
        return pd.DataFrame(columns=columns)
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ("received", "expected"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    if data.empty:
        return pd.DataFrame(columns=columns)
    end = pd.Timestamp(end_date).normalize() if end_date is not None else data["date"].max()
    rows: list[dict[str, object]] = []
    for value in windows:
        days = int(value)
        comparison = nco_range_comparison(data, end - pd.Timedelta(days=days - 1), end)
        rows.append({
            "days": days,
            "current_percent": comparison["current_percent"],
            "previous_percent": comparison["previous_percent"],
            "delta_pp": comparison["delta_pp"],
            "current_days": comparison["current_days"],
            "previous_days": comparison["previous_days"],
        })
    return pd.DataFrame(rows, columns=columns)


def nco_range_comparison(daily: pd.DataFrame, start_date: object, end_date: object) -> dict[str, object]:
    """Compare a selected range with the immediately preceding equal range."""
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("NCO range start and end must be valid, ordered dates.")
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    previous_end = start - pd.Timedelta(days=1)
    previous_start = previous_end - (end - start)

    data = daily.copy()
    if not data.empty and {"date", "received", "expected"}.issubset(data.columns):
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        for column in ("received", "expected"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
    else:
        data = pd.DataFrame(columns=["date", "received", "expected"])

    def rate(range_start: pd.Timestamp, range_end: pd.Timestamp) -> tuple[float, int]:
        subset = data[data["date"].between(range_start, range_end) & data["received"].notna() & data["expected"].gt(0)]
        expected_total = float(subset["expected"].sum())
        if subset.empty or not expected_total:
            return np.nan, int(len(subset))
        return float(subset["received"].sum() / expected_total * 100.0), int(len(subset))

    current, current_days = rate(start, end)
    previous, previous_days = rate(previous_start, previous_end)
    return {
        "start_date": start,
        "end_date": end,
        "previous_start_date": previous_start,
        "previous_end_date": previous_end,
        "current_percent": current,
        "previous_percent": previous,
        "delta_pp": current - previous if pd.notna(current) and pd.notna(previous) else np.nan,
        "current_days": current_days,
        "previous_days": previous_days,
    }


def format_pp_delta(value: object) -> str:
    """Format a percentage-point delta without inventing a zero."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    sign = "+" if numeric >= 0 else "−"
    return f"{sign}{abs(numeric):.1f} pp"


def prepare_issues(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
    if {"cycle_date_utc", "cycle_hour"}.issubset(data.columns):
        hour = data["cycle_hour"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        data["cycle_hour"] = hour
        data["cycle_dt"] = pd.to_datetime(
            data["cycle_date_utc"].astype(str) + " " + hour + ":00",
            errors="coerce",
            utc=True,
        )
    else:
        data["cycle_dt"] = pd.NaT
    if "message_time_utc" in data:
        data["message_dt"] = pd.to_datetime(data["message_time_utc"], errors="coerce", utc=True)
    for column in ("station_id", "model", "issue_category"):
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    if "station_id" in data:
        data["station_id"] = data["station_id"].str.upper()
    if "model" in data:
        data["model"] = data["model"].str.upper()
    return data.sort_values([column for column in ("cycle_dt", "message_dt", "station_id") if column in data]).reset_index(drop=True)


def prepare_stations(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
    for column in ("latitude", "longitude"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    active = data.get("active_expected", pd.Series(True, index=data.index)).astype(str).str.lower().isin({"true", "1", "yes"})
    conus = data.get("latitude", pd.Series(np.nan, index=data.index)).between(24.0, 50.0) & data.get(
        "longitude", pd.Series(np.nan, index=data.index)
    ).between(-126.0, -66.0)
    data = data[active & conus].copy()
    if "station_id" in data:
        data["station_id"] = data["station_id"].fillna("").astype(str).str.upper()
    return data.reset_index(drop=True)


def prepare_station_deficits(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data
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
        "ytd_observed",
        "ytd_expected",
        "ytd_deficit",
        "ytd_percent",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "station_id" in data:
        data["station_id"] = data["station_id"].fillna("").astype(str).str.upper()
    return data.reset_index(drop=True)


def _payload_from_inputs(inputs: MonitorInputs) -> SocialPayload:
    metrics = calculate_metrics(inputs)
    latest = metrics.latest_complete
    observed = float(latest.get("launches_7d_avg", np.nan)) if latest is not None else np.nan
    expected = float(latest.get("baseline_5yr_avg", np.nan)) if latest is not None else np.nan
    gap = float(latest.get("percent_vs_baseline", np.nan)) if latest is not None else np.nan
    series = metrics.current_igra.rename(
        columns={"launches_7d_avg": "observed", "baseline_5yr_avg": "baseline", "launches": "daily"}
    )
    columns = [column for column in ("date", "observed", "baseline", "daily") if column in series]
    windows = metrics.windows.copy()
    return SocialPayload(
        generated_at=None,
        latest_date=pd.Timestamp(latest["date"]).date().isoformat() if latest is not None else None,
        partial_date=metrics.partial_date.date().isoformat() if metrics.partial_date is not None else None,
        nco_cycle=metrics.nco_cycle_text if metrics.latest_nco is not None else None,
        observed=observed,
        expected=expected,
        gap_percent=gap,
        windows=windows,
        series=series[columns].copy() if columns else pd.DataFrame(columns=["date", "observed", "baseline", "daily"]),
        stations=metrics.station_statuses.copy(),
        issue_count=metrics.impacted_station_count if metrics.latest_nco is not None else None,
        nco_count=metrics.nco_count,
        caveats=EXACT_CAVEATS,
    )


def _source_rows(repo_root: Path, paths: dict[str, Path], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    optional_sources = {"SPC sounding page"}
    date_columns = {
        "IGRA daily archive": "date",
        "NCO availability": "cycle_date_utc",
        "NCO station issues": "cycle_date_utc",
        "SPC sounding page": "date",
    }
    key_columns = {
        "IGRA daily archive": ["date", "year"],
        "NCO availability": ["cycle_date_utc", "cycle_hour", "model"],
        "NCO station issues": ["cycle_date_utc", "cycle_hour", "model", "wmo_id", "station_id", "issue_text"],
    }
    rows: list[dict[str, object]] = []
    for name, path in paths.items():
        frame = frames.get(name, pd.DataFrame())
        resolved = path if path.is_absolute() else repo_root / path
        modified = pd.Timestamp(resolved.stat().st_mtime, unit="s", tz="UTC") if resolved.exists() else pd.NaT
        date_column = date_columns.get(name)
        coverage = pd.to_datetime(frame[date_column], errors="coerce", utc=True).dropna() if date_column and date_column in frame else pd.Series(dtype="datetime64[ns, UTC]")
        keys = key_columns.get(name, [])
        duplicate_rows = int(frame.duplicated(keys).sum()) if keys and set(keys).issubset(frame.columns) else 0
        status = "ready" if resolved.exists() and not frame.empty else ("empty" if resolved.exists() else "missing")
        if status == "ready" and duplicate_rows:
            status = "check"
        rows.append(
            {
                "source": name,
                "required": name not in optional_sources,
                "status": status,
                "rows": int(len(frame)),
                "modified_utc": modified,
                "coverage_start_utc": coverage.min() if not coverage.empty else pd.NaT,
                "coverage_end_utc": coverage.max() if not coverage.empty else pd.NaT,
                "duplicate_rows": duplicate_rows,
                "path": str(resolved),
            }
        )
    return pd.DataFrame(rows)


def source_health_summary(source_status: pd.DataFrame) -> dict[str, object]:
    """Return compact source-health facts for dashboard headers and methods."""
    if source_status.empty:
        return {"total": 0, "ready": 0, "problems": 0, "required_total": 0, "required_ready": 0, "optional_problems": 0, "duplicate_rows": 0, "latest_modified_utc": pd.NaT}
    statuses = source_status.get("status", pd.Series(dtype=str)).astype(str)
    required = source_status.get("required", pd.Series(True, index=source_status.index)).astype(bool)
    modified = pd.to_datetime(source_status.get("modified_utc", pd.Series(dtype=str)), errors="coerce", utc=True).dropna()
    duplicates = pd.to_numeric(source_status.get("duplicate_rows", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {
        "total": int(len(source_status)),
        "ready": int(statuses.eq("ready").sum()),
        "problems": int((required & ~statuses.eq("ready")).sum()),
        "required_total": int(required.sum()),
        "required_ready": int((required & statuses.eq("ready")).sum()),
        "optional_problems": int((~required & ~statuses.eq("ready")).sum()),
        "duplicate_rows": int(duplicates.sum()),
        "latest_modified_utc": modified.max() if not modified.empty else pd.NaT,
    }


def dashboard_file_signature(repo_root: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a hashable signature so Streamlit cache refreshes when files change."""
    paths = (
        repo_root / MANIFEST_PATH,
        repo_root / "outputs/conus_balloon_launches_by_year_daily.csv",
        repo_root / "data/nco_raob_availability.csv",
        repo_root / "data/nco_raob_station_issues.csv",
        repo_root / "data/upper_air_station_master.csv",
        repo_root / STATION_DEFICITS_PATH,
        repo_root / IGRA_METADATA_PATH,
        repo_root / SPC_STATUS_PATH,
    )
    return tuple(
        (str(path), path.stat().st_mtime_ns if path.exists() else 0, path.stat().st_size if path.exists() else 0)
        for path in paths
    )


def load_dashboard_snapshot(repo_root: Path) -> DashboardSnapshot:
    """Load existing products only; this function never invokes the refresh pipeline."""
    repo_root = repo_root.resolve()
    inputs = load_monitor_inputs(repo_root)
    manifest_path = repo_root / MANIFEST_PATH
    manifest_used = manifest_path.exists()
    payload = load_social_payload(manifest_path) if manifest_used else _payload_from_inputs(inputs)
    archive_detail_series = archive_detail_series_from_igra(inputs.igra, payload.latest_date)
    if not archive_detail_series.empty:
        payload.series = archive_detail_series
    payload.series = enrich_archive_variability(payload.series, inputs.igra)
    nco = prepare_nco(inputs.nco)
    issues = prepare_issues(inputs.issues)
    stations = prepare_stations(inputs.stations)
    station_deficits_path = repo_root / STATION_DEFICITS_PATH
    station_deficits = prepare_station_deficits(_read_csv(station_deficits_path))
    igra_metadata_path = repo_root / IGRA_METADATA_PATH
    igra_metadata = _read_json(igra_metadata_path)
    spc_status_path = repo_root / SPC_STATUS_PATH
    spc_status = _read_csv(spc_status_path, dtype=str)
    refresh_status = _read_json(repo_root / REFRESH_STATUS_PATH)
    source_paths = {
        "metrics manifest": manifest_path,
        "IGRA daily archive": inputs.input_paths["igra_daily"],
        "NCO availability": inputs.input_paths["nco_availability"],
        "NCO station issues": inputs.input_paths["nco_issues"],
        "station master": inputs.input_paths["station_master"],
        "station archive deficits": station_deficits_path,
        "IGRA build metadata": igra_metadata_path,
        "SPC sounding page": spc_status_path,
    }
    frames = {
        "metrics manifest": pd.DataFrame([{"manifest": 1}]) if manifest_used else pd.DataFrame(),
        "IGRA daily archive": inputs.igra,
        "NCO availability": nco,
        "NCO station issues": issues,
        "station master": stations,
        "station archive deficits": station_deficits,
        "IGRA build metadata": pd.DataFrame([igra_metadata]) if igra_metadata else pd.DataFrame(),
        "SPC sounding page": spc_status,
    }
    return DashboardSnapshot(
        repo_root=repo_root,
        payload=payload,
        nco=nco,
        issues=issues,
        stations=stations,
        station_deficits=station_deficits,
        igra_metadata=igra_metadata,
        source_status=_source_rows(repo_root, source_paths, frames),
        manifest_used=manifest_used,
        refresh_status=refresh_status,
    )


def filter_series(series: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    if series.empty or "date" not in series:
        return series.copy()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = pd.to_datetime(series["date"], errors="coerce")
    return series[(dates >= start_ts) & (dates <= end_ts)].copy().reset_index(drop=True)


def archive_window_metrics(series: pd.DataFrame, days: Iterable[int] = (30, 60, 90, 180)) -> pd.DataFrame:
    required = {"date", "daily", "baseline"}
    if series.empty or not required.issubset(series.columns):
        return pd.DataFrame(columns=["days", "observed", "expected", "deficit", "percent"])
    data = series.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("daily", "baseline"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "daily", "baseline"]).sort_values("date")
    if data.empty:
        return pd.DataFrame(columns=["days", "observed", "expected", "deficit", "percent"])
    latest = data["date"].max()
    rows: list[dict[str, float | int]] = []
    for window in days:
        subset = data[data["date"] >= latest - pd.Timedelta(days=int(window) - 1)]
        if subset.empty:
            continue
        observed = float(subset["daily"].sum())
        expected = float(subset["baseline"].sum())
        deficit = observed - expected
        rows.append(
            {
                "days": int(window),
                "observed": observed,
                "expected": expected,
                "deficit": deficit,
                "percent": deficit / expected * 100.0 if expected else np.nan,
            }
        )
    return pd.DataFrame(rows)


def filter_nco(frame: pd.DataFrame, models: Iterable[str], cycle_hours: Iterable[str], lookback_days: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    model_values = {str(value).upper() for value in models}
    hour_values = {str(value).zfill(2) for value in cycle_hours}
    if model_values and "model" in data:
        data = data[data["model"].isin(model_values)]
    if hour_values and "cycle_hour" in data:
        data = data[data["cycle_hour"].isin(hour_values)]
    if not data.empty and "cycle_dt" in data:
        latest = data["cycle_dt"].max()
        if pd.notna(latest):
            data = data[data["cycle_dt"] >= latest - pd.Timedelta(days=int(lookback_days))]
    return data.reset_index(drop=True)


def latest_and_previous_comparable_nco(frame: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    """Return the latest NCO row and the preceding row for the same model/cycle hour."""
    if frame.empty or "cycle_dt" not in frame:
        return None, None
    valid = frame.dropna(subset=["cycle_dt"]).sort_values(
        [column for column in ("cycle_dt", "message_dt") if column in frame]
    )
    if valid.empty:
        return None, None
    latest = valid.iloc[-1]
    previous = valid[valid["cycle_dt"] < latest["cycle_dt"]]
    for column in ("model", "cycle_hour"):
        if column in previous and column in latest.index and pd.notna(latest[column]):
            matching = previous[previous[column].astype(str).eq(str(latest[column]))]
            if not matching.empty:
                previous = matching
    if previous.empty:
        return latest, None
    return latest, previous.iloc[-1]


def _issues_for_nco_row(issues: pd.DataFrame, row: pd.Series | None) -> pd.DataFrame:
    if row is None or issues.empty or "cycle_dt" not in issues:
        return pd.DataFrame(columns=issues.columns)
    selected = issues[issues["cycle_dt"].eq(row.get("cycle_dt"))]
    if "model" in selected and pd.notna(row.get("model")):
        model_rows = selected[selected["model"].astype(str).eq(str(row.get("model")))]
        if not model_rows.empty:
            selected = model_rows
    return selected.drop_duplicates("station_id", keep="last") if "station_id" in selected else selected


def station_issue_changes(issues: pd.DataFrame, nco: pd.DataFrame) -> pd.DataFrame:
    """Compare parsed station issues with the preceding comparable NCO cycle."""
    latest, previous = latest_and_previous_comparable_nco(nco)
    columns = ["station_id", "transition", "previous_category", "latest_category", "issue_text"]
    if latest is None or previous is None or "station_id" not in issues:
        return pd.DataFrame(columns=columns)
    current = _issues_for_nco_row(issues, latest).set_index("station_id", drop=False)
    prior = _issues_for_nco_row(issues, previous).set_index("station_id", drop=False)
    station_ids = sorted(set(current.index.astype(str)) | set(prior.index.astype(str)))
    rows: list[dict[str, str]] = []
    for station_id in station_ids:
        current_row = current.loc[station_id] if station_id in current.index else None
        prior_row = prior.loc[station_id] if station_id in prior.index else None
        if isinstance(current_row, pd.DataFrame):
            current_row = current_row.iloc[-1]
        if isinstance(prior_row, pd.DataFrame):
            prior_row = prior_row.iloc[-1]
        current_category = str(current_row.get("issue_category", "")) if current_row is not None else ""
        prior_category = str(prior_row.get("issue_category", "")) if prior_row is not None else ""
        if not prior_category:
            transition = "New issue"
        elif not current_category:
            transition = "Resolved"
        elif current_category != prior_category:
            transition = "Category changed"
        else:
            transition = "Persistent"
        source_row = current_row if current_row is not None else prior_row
        rows.append(
            {
                "station_id": station_id,
                "transition": transition,
                "previous_category": prior_category,
                "latest_category": current_category,
                "issue_text": str(source_row.get("issue_text", "")) if source_row is not None else "",
            }
        )
    order = {"New issue": 0, "Category changed": 1, "Persistent": 2, "Resolved": 3}
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(
        ["transition", "station_id"], key=lambda values: values.map(order) if values.name == "transition" else values
    ).reset_index(drop=True)


def latest_issue_rows(issues: pd.DataFrame, nco: pd.DataFrame) -> pd.DataFrame:
    if issues.empty or nco.empty or "cycle_dt" not in nco or "cycle_dt" not in issues:
        return pd.DataFrame(columns=issues.columns)
    valid_nco = nco.dropna(subset=["cycle_dt"]).sort_values([column for column in ("cycle_dt", "message_dt") if column in nco])
    if valid_nco.empty:
        return pd.DataFrame(columns=issues.columns)
    latest = valid_nco.iloc[-1]
    mask = issues["cycle_dt"].eq(latest["cycle_dt"])
    if "model" in issues and "model" in latest:
        model_mask = issues["model"].eq(str(latest["model"]).upper())
        if (mask & model_mask).any():
            mask &= model_mask
    return issues[mask].copy().sort_values([column for column in ("issue_category", "station_id") if column in issues]).reset_index(drop=True)


def station_status_frame(stations: pd.DataFrame, latest_issues: pd.DataFrame) -> pd.DataFrame:
    data = stations.copy()
    if data.empty:
        return data
    data["status"] = "No issue reported"
    data["issue_category"] = ""
    data["issue_text"] = ""
    if latest_issues.empty or "station_id" not in latest_issues:
        return data
    issue_columns = [column for column in ("station_id", "issue_category", "issue_text", "model", "cycle_dt") if column in latest_issues]
    issue_lookup = latest_issues[issue_columns].drop_duplicates("station_id", keep="last")
    data = data.merge(issue_lookup, on="station_id", how="left", suffixes=("", "_latest"))
    category = data.get("issue_category_latest", data.get("issue_category", pd.Series("", index=data.index))).fillna("").astype(str)
    data["issue_category"] = category
    data["status"] = np.where(category.ne(""), "NCO-reported issue", "No issue reported")
    if "issue_text_latest" in data:
        data["issue_text"] = data["issue_text_latest"].fillna("")
    for column in ("model_latest", "cycle_dt_latest"):
        if column in data:
            data[column.removesuffix("_latest")] = data[column]
    drop_columns = [column for column in data if column.endswith("_latest")]
    return data.drop(columns=drop_columns).reset_index(drop=True)


def issue_counts_by_cycle(issues: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    if issues.empty or not {"cycle_dt", "issue_category"}.issubset(issues.columns):
        return pd.DataFrame(columns=["cycle_dt", "issue_category", "count"])
    data = issues.dropna(subset=["cycle_dt"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["cycle_dt", "issue_category", "count"])
    latest = data["cycle_dt"].max()
    data = data[data["cycle_dt"] >= latest - pd.Timedelta(days=int(lookback_days))]
    return data.groupby(["cycle_dt", "issue_category"], as_index=False).size().rename(columns={"size": "count"})


def format_metric(value: object, *, decimals: int = 1, suffix: str = "") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:,.{decimals}f}{suffix}"

