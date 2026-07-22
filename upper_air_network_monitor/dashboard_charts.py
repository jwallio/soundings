"""Plotly chart builders for the Upper-Air Streamlit dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "background": "#061521",
    "panel": "#0D2538",
    "grid": "#294960",
    "text": "#F8FBFF",
    "muted": "#AFC1D4",
    "observed": "#59C8F5",
    "baseline": "#C1CBD8",
    "deficit": "#FF704F",
    "orange": "#FF8A3D",
    "magenta": "#F15BB5",
    "amber": "#F6C85F",
    "clean": "#52D3A2",
    "unknown": "#8FA3B8",
}

ARCHIVE_EVENT_TAGS = (
    {
        "label": "Hurricane Henri",
        "start": pd.Timestamp("2021-08-19"),
        "end": pd.Timestamp("2021-08-27"),
        "peak": pd.Timestamp("2021-08-23"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Henri prompted enhanced upper-air sampling in the eastern U.S.",
    },
    {
        "label": "Hurricane Ida",
        "start": pd.Timestamp("2021-08-30"),
        "end": pd.Timestamp("2021-09-02"),
        "peak": pd.Timestamp("2021-09-01"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Ida and its inland/remnant impacts prompted enhanced upper-air sampling.",
    },
    {
        "label": "Hurricane Ian",
        "start": pd.Timestamp("2022-09-25"),
        "end": pd.Timestamp("2022-10-04"),
        "peak": pd.Timestamp("2022-09-30"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Ian produced a major supplemental upper-air sampling period.",
    },
    {
        "label": "Hurricane Nicole",
        "start": pd.Timestamp("2022-11-08"),
        "end": pd.Timestamp("2022-11-14"),
        "peak": pd.Timestamp("2022-11-10"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Nicole and downstream impacts coincided with elevated sampling.",
    },
    {
        "label": "Feb. 2023 severe weather",
        "start": pd.Timestamp("2023-02-16"),
        "end": pd.Timestamp("2023-02-20"),
        "peak": pd.Timestamp("2023-02-17"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: a multi-state severe-weather outbreak coincided with elevated sampling.",
    },
    {
        "label": "Hurricane Lee",
        "start": pd.Timestamp("2023-09-13"),
        "end": pd.Timestamp("2023-09-19"),
        "peak": pd.Timestamp("2023-09-16"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Lee's western Atlantic/New England threat coincided with elevated sampling.",
    },
    {
        "label": "Hurricane Debby",
        "start": pd.Timestamp("2024-08-06"),
        "end": pd.Timestamp("2024-08-11"),
        "peak": pd.Timestamp("2024-08-07"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Debby coincided with elevated upper-air sampling.",
    },
    {
        "label": "Hurricane Helene",
        "start": pd.Timestamp("2024-09-26"),
        "end": pd.Timestamp("2024-09-30"),
        "peak": pd.Timestamp("2024-09-26"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Helene coincided with elevated upper-air sampling.",
    },
    {
        "label": "Hurricane Milton",
        "start": pd.Timestamp("2024-10-08"),
        "end": pd.Timestamp("2024-10-13"),
        "peak": pd.Timestamp("2024-10-09"),
        "threshold": 140.0,
        "explainer": "Likely temporal context: Hurricane Milton coincided with elevated upper-air sampling.",
    },
)

WORKFORCE_EVENT_TAGS = (
    {
        "label": "OMB/OPM RIF directive",
        "short_label": "OMB/OPM RIF",
        "date": pd.Timestamp("2025-02-13"),
        "explainer": "OMB and OPM issued a joint directive setting a March deadline for agency RIF lists. Reported retirement and staffing concerns followed; timeline context only.",
    },
    {
        "label": "DOGE contract list",
        "short_label": "DOGE contracts",
        "date": pd.Timestamp("2025-02-17"),
        "explainer": "DOGE released a list of federal contracts targeted for immediate cancellation. The list included NOAA-linked technology and maintenance support; timeline context only.",
    },
    {
        "label": "NOAA position directives",
        "short_label": "NOAA positions",
        "date": pd.Timestamp("2025-03-13"),
        "explainer": "Internal directives called for 650 positions to be cleared and 1,029 more identified across NOAA. Timeline context only; this chart does not attribute sounding changes to the directive.",
    },
    {
        "label": "NWS RAOB reductions",
        "short_label": "NWS RAOB cuts",
        "date": pd.Timestamp("2025-03-20"),
        "explainer": "NWS announced reductions at 11 locations amid staffing constraints and a federal hiring freeze. Timeline context only; this chart does not establish a cause.",
    },
)


def _base_layout(fig: go.Figure, *, height: int, margin: dict[str, int] | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=margin or {"l": 52, "r": 24, "t": 22, "b": 45},
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font={"family": "Arial, sans-serif", "color": COLORS["text"], "size": 13},
        hoverlabel={
            "bgcolor": COLORS["background"],
            "font_color": COLORS["text"],
            "bordercolor": COLORS["grid"],
            "align": "left",
            "namelength": -1,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0, "font": {"color": COLORS["muted"]}, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(showgrid=False, linecolor=COLORS["grid"], tickfont={"color": COLORS["muted"]})
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["baseline"], tickfont={"color": COLORS["muted"]})
    return fig


def empty_figure(message: str, *, height: int = 390) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": COLORS["muted"], "size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height=height)


def _deficit_segments(data: pd.DataFrame) -> list[pd.DataFrame]:
    """Return contiguous below-baseline runs for clean Plotly area fills.

    A single ``where(below)`` trace inserts NaNs whenever the observed series
    crosses the baseline. Plotly can then close the fill polygon across those
    NaNs, producing the detached triangular patches seen in the dashboard.
    Segmenting on both deficit state and date continuity keeps each filled
    polygon bounded by the actual observations that support it.
    """
    eligible = data.loc[
        data["observed"].notna()
        & data["baseline"].notna()
        & data["observed"].lt(data["baseline"]),
        ["date", "observed", "baseline"],
    ].sort_values("date")
    if eligible.empty:
        return []
    breaks = eligible["date"].diff().gt(pd.Timedelta(days=1)).cumsum()
    return [segment for _, segment in eligible.groupby(breaks) if len(segment) >= 2]


def archive_trend_figure(
    series: pd.DataFrame,
    *,
    show_daily: bool = False,
    show_percent_axis: bool = False,
    show_event_tags: bool = False,
    show_workforce_events: bool = False,
    height: int = 430,
) -> go.Figure:
    required = {"date", "observed", "baseline"}
    if series.empty or not required.issubset(series.columns):
        return empty_figure("Archive time-series data are unavailable.", height=height)
    data = series.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("observed", "baseline", "daily"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    if data.empty:
        return empty_figure("Archive time-series data are unavailable.", height=height)

    # Percentage is calculated against the same baseline trace shown in the
    # chart, so the hover value always reconciles with the plotted data.
    data["percent_from_baseline"] = np.where(
        data["baseline"].ne(0),
        (data["observed"] - data["baseline"]) / data["baseline"] * 100.0,
        np.nan,
    ).round(1)

    below = data["observed"] < data["baseline"]
    y_reference_columns = [column for column in ("observed", "baseline", "baseline_low", "baseline_high") if column in data]
    y_reference = pd.concat([data[column] for column in y_reference_columns], ignore_index=True).dropna()
    guide_y_min = math.floor(float(y_reference.min()) - 2) if not y_reference.empty else 0.0
    guide_y_max = math.ceil(float(y_reference.max()) + 2) if not y_reference.empty else 1.0
    workforce_dates = {event["date"] for event in WORKFORCE_EVENT_TAGS if event["short_label"] == "NWS RAOB cuts"}
    if show_workforce_events and data["date"].isin(workforce_dates).any():
        # Keep the short RAOB-cut label in a dedicated upper band,
        # even when a custom window contains only the 2025 workforce dates.
        guide_y_max = max(guide_y_max, 150.0)
    fig = go.Figure()
    if {"baseline_low", "baseline_high"}.issubset(data.columns):
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["baseline_low"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["baseline_high"],
                name="2021–2024 historical range",
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(193,203,216,0.12)",
                hovertemplate="%{x|%b %d, %Y}<br>Historical range high: %{y:.1f}/day<extra></extra>",
            )
        )
    for segment in _deficit_segments(data):
        fig.add_trace(
            go.Scatter(
                x=segment["date"],
                y=segment["baseline"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=segment["date"],
                y=segment["observed"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(255,112,79,0.28)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if show_daily and "daily" in data:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["daily"],
                name="Daily archive records",
                mode="lines",
                line={"color": "rgba(89,200,245,0.34)", "width": 1},
                hovertemplate="%{x|%b %d, %Y}<br>Daily records: %{y:.0f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["baseline"],
            name="2021–2024 same-date baseline",
            mode="lines",
            line={"color": COLORS["baseline"], "width": 2, "dash": "dash"},
            hovertemplate="%{x|%b %d, %Y}<br>Baseline: %{y:.1f}/day<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["observed"],
            name="Observed 7-day average",
            mode="lines",
            line={"color": COLORS["observed"], "width": 4},
            customdata=np.column_stack([data["baseline"], data["percent_from_baseline"]]),
            hovertemplate=(
                "%{x|%b %d, %Y}<br>Observed: %{y:.1f}/day"
                "<br>Baseline: %{customdata[0]:.1f}/day"
                "<br>Difference: %{customdata[1]:+.1f}%<extra></extra>"
            ),
        )
    )
    if show_event_tags:
        event_x: list[pd.Timestamp] = []
        event_y: list[float] = []
        event_labels: list[str] = []
        event_ranges: list[str] = []
        for event in ARCHIVE_EVENT_TAGS:
            peak_month_day = pd.Timestamp(event["peak"]).strftime("%m-%d")
            data_days = data["date"].dt.strftime("%m-%d")
            y_column = "baseline_high" if "baseline_high" in data else "observed"
            event_rows = data[data_days.eq(peak_month_day) & data[y_column].ge(float(event["threshold"]))]
            if event_rows.empty:
                continue
            latest = data["date"].max()
            default_start = latest - pd.Timedelta(days=364) if pd.notna(latest) else pd.NaT
            visible_rows = event_rows[event_rows["date"].between(default_start, latest)] if pd.notna(default_start) else event_rows
            if not visible_rows.empty:
                event_rows = visible_rows
            peak = event_rows.loc[event_rows[y_column].idxmax()]
            event_x.append(pd.Timestamp(peak["date"]))
            event_y.append(float(peak[y_column]))
            event_labels.append(str(event["label"]))
            event_ranges.append(f"{event['start']:%b %d}-{event['end']:%b %d, %Y}")
        if event_x:
            fig.add_trace(
                go.Scatter(
                    x=event_x,
                    y=event_y,
                    name="Event maximums",
                    mode="markers+text",
                    marker={
                        "color": COLORS["magenta"],
                        "size": 11,
                        "symbol": "diamond",
                        "line": {"color": COLORS["text"], "width": 1.1},
                    },
                    text=event_labels,
                    textposition=[
                        "top center",
                        "bottom center",
                        "top center",
                        "bottom center",
                        "top center",
                        "bottom center",
                        "top center",
                        "bottom center",
                        "top center",
                    ][: len(event_labels)],
                    textfont={"color": COLORS["magenta"], "size": 10},
                    customdata=np.column_stack([event_labels, event_ranges]),
                    hovertemplate=(
                        "%{x|%b %d, %Y}<br><b>%{customdata[0]}</b>"
                        "<br>Historical max: %{y:.1f}/day"
                        "<br>Event window: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
    if show_workforce_events:
        # Keep only the operationally relevant RAOB-reduction marker visible.
        # The other timeline context remains in the source constants for
        # reproducibility, but is intentionally not rendered in the dashboard.
        for event in WORKFORCE_EVENT_TAGS:
            if event["short_label"] != "NWS RAOB cuts":
                continue
            matches = data[data["date"].eq(event["date"])]
            if matches.empty:
                continue
            row = matches.iloc[-1]
            if pd.isna(row["observed"]):
                continue
            observed_y = float(row["observed"])
            # Keep the label above the 145 gridline so the guide does not run
            # through the text while preserving the capped chart scale.
            label_y = 146.5
            label_y = min(label_y, guide_y_max - 2.0)
            label_y = max(label_y, guide_y_min + 2.0)
            fig.add_trace(
                go.Scatter(
                    x=[event["date"], event["date"], event["date"]],
                    y=[guide_y_min, label_y, guide_y_max],
                    name="NWS RAOB cuts",
                    mode="lines+text",
                    line={"color": COLORS["orange"], "width": 1.5, "dash": "dot"},
                    opacity=0.9,
                    showlegend=False,
                    text=["", event["short_label"], ""],
                    textposition=["middle right", "middle right", "middle right"],
                    textfont={"color": COLORS["orange"], "size": 9},
                    customdata=np.column_stack(
                        [
                            [event["label"]] * 3,
                            [event["explainer"]] * 3,
                            [observed_y] * 3,
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{x|%b %d, %Y}<br>"
                        "%{customdata[1]}<br>"
                        "7-day avg: %{customdata[2]:.1f}/day"
                        "<extra></extra>"
                    ),
                )
            )
    _base_layout(fig, height=height, margin={"l": 62, "r": 34, "t": 34, "b": 48})
    fig.update_yaxes(title="Archived soundings / day", rangemode="normal")
    fig.update_xaxes(hoverformat="%b %d, %Y")
    return fig


def archive_windows_figure(windows: pd.DataFrame, *, height: int = 350, vertical: bool = False) -> go.Figure:
    if windows.empty or not {"days", "percent", "deficit"}.issubset(windows.columns):
        return empty_figure("Recent-window metrics are unavailable.", height=height)
    data = windows.dropna(subset=["days", "percent"]).sort_values("days", ascending=False).copy()
    colors = [COLORS["deficit"] for _value in data["days"]]
    labels = [f"{int(value)} days" for value in data["days"]]
    custom = np.column_stack([data["deficit"].abs(), data["observed"], data["expected"]])
    if vertical:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=data["percent"],
                marker={"color": colors},
                text=[f"{value:+.1f}%" for value in data["percent"]],
                textposition="outside",
                textfont={"color": COLORS["text"], "size": 13},
                cliponaxis=False,
                customdata=custom,
                hovertemplate="%{x}<br>Difference: %{y:+.1f}%<br>Shortfall: %{customdata[0]:,.0f}<br>Observed: %{customdata[1]:,.0f}<br>Expected: %{customdata[2]:,.0f}<extra></extra>",
            )
        )
        _base_layout(fig, height=height, margin={"l": 68, "r": 28, "t": 24, "b": 58})
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Difference from expected archive volume", ticksuffix="%", zeroline=True, zerolinewidth=1.2)
        return fig
    fig = go.Figure(
        go.Bar(
            x=data["percent"],
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{value:+.1f}%" for value in data["percent"]],
            textposition="inside",
            insidetextanchor="start",
            textfont={"color": COLORS["background"], "size": 13},
            customdata=custom,
            hovertemplate="%{y}<br>Difference: %{x:+.1f}%<br>Shortfall: %{customdata[0]:,.0f}<br>Observed: %{customdata[1]:,.0f}<br>Expected: %{customdata[2]:,.0f}<extra></extra>",
        )
    )
    _base_layout(fig, height=height, margin={"l": 82, "r": 28, "t": 18, "b": 52})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Difference from expected archive volume", ticksuffix="%", zeroline=True, zerolinewidth=1.2)
    return fig


def station_archive_shortfall_figure(
    station_deficits: pd.DataFrame,
    *,
    height: int = 360,
    top_n: int = 8,
    days: int = 90,
) -> go.Figure:
    """Rank stations by fewer IGRA archive records than the same-date baseline."""
    observed_column = f"observed_{days}"
    expected_column = f"expected_{days}"
    deficit_column = f"deficit_{days}"
    shortfall_column = f"missed_{days}"
    has_deficit = deficit_column in station_deficits.columns
    has_shortfall = shortfall_column in station_deficits.columns
    required = {"display_label", observed_column, expected_column}
    if not (has_deficit or has_shortfall):
        required.add(deficit_column)
    if station_deficits.empty or not required.issubset(station_deficits.columns):
        return empty_figure(f"Station-level {days}-day archive data are unavailable.", height=height)
    data = station_deficits.copy()
    for column in (observed_column, expected_column, deficit_column, shortfall_column):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if has_shortfall:
        data["shortfall"] = data[shortfall_column]
    else:
        data["shortfall"] = -data[deficit_column]
    data = data.dropna(subset=["display_label", "shortfall"]).query("shortfall > 0")
    if data.empty:
        return empty_figure(f"No station-level {days}-day archive shortfalls were found.", height=height)
    data = data.nlargest(top_n, "shortfall").sort_values("shortfall")
    fig = go.Figure(
        go.Bar(
            x=data["shortfall"],
            y=data["display_label"],
            orientation="h",
            marker={"color": COLORS["deficit"]},
            text=[f"{value:,.0f}" for value in data["shortfall"]],
            textposition="outside",
            textfont={"color": COLORS["text"], "size": 12},
            cliponaxis=False,
            customdata=np.column_stack([data[observed_column], data[expected_column]]),
            hovertemplate=f"%{{y}}<br>Archive shortfall ({days}D): %{{x:,.1f}}<br>Observed: %{{customdata[0]:,.1f}}<br>Expected: %{{customdata[1]:,.1f}}<extra></extra>",
        )
    )
    _base_layout(fig, height=height, margin={"l": 150, "r": 48, "t": 24, "b": 58})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=f"Fewer archived soundings over {days} days", rangemode="tozero")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def station_archive_surplus_figure(
    station_deficits: pd.DataFrame,
    *,
    height: int = 360,
    top_n: int = 8,
    days: int = 90,
) -> go.Figure:
    """Rank stations with more archived soundings than the 90-day baseline."""
    observed_column = f"observed_{days}"
    expected_column = f"expected_{days}"
    required = {"display_label", observed_column, expected_column}
    if station_deficits.empty or not required.issubset(station_deficits.columns):
        return empty_figure(f"Station-level {days}-day archive data are unavailable.", height=height)
    data = station_deficits.copy()
    for column in (observed_column, expected_column):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["surplus"] = data[observed_column] - data[expected_column]
    data = data.dropna(subset=["display_label", "surplus"]).query("surplus > 0")
    if data.empty:
        return empty_figure(f"No station-level {days}-day archive surpluses were found.", height=height)
    data = data.nlargest(top_n, "surplus").sort_values("surplus")
    fig = go.Figure(
        go.Bar(
            x=data["surplus"],
            y=data["display_label"],
            orientation="h",
            marker={"color": COLORS["clean"]},
            text=[f"+{value:,.0f}" for value in data["surplus"]],
            textposition="outside",
            textfont={"color": COLORS["text"], "size": 12},
            cliponaxis=False,
            customdata=np.column_stack([data[observed_column], data[expected_column]]),
            hovertemplate=f"%{{y}}<br>Archive surplus ({days}D): +%{{x:,.1f}}<br>Observed: %{{customdata[0]:,.1f}}<br>Expected: %{{customdata[1]:,.1f}}<extra></extra>",
        )
    )
    _base_layout(fig, height=height, margin={"l": 150, "r": 48, "t": 24, "b": 58})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=f"More archived soundings over {days} days", rangemode="tozero")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def station_status_map_figure(stations: pd.DataFrame, *, height: int = 430) -> go.Figure:
    required = {"latitude", "longitude", "status"}
    if stations.empty or not required.issubset(stations.columns):
        return empty_figure("Station-status geography is unavailable.", height=height)
    data = stations.dropna(subset=["latitude", "longitude"]).copy()
    if data.empty:
        return empty_figure("Station-status geography is unavailable.", height=height)
    fig = go.Figure()
    styles = {
        "No issue reported": (COLORS["clean"], "circle", 8),
        "NCO-reported issue": (COLORS["deficit"], "diamond", 12),
        "available / no issue": (COLORS["clean"], "circle", 8),
        "missing / problem": (COLORS["deficit"], "diamond", 12),
        "partial / quality": (COLORS["amber"], "diamond", 11),
        "unknown": (COLORS["unknown"], "circle-open", 8),
    }
    for status in data["status"].fillna("unknown").astype(str).unique():
        subset = data[data["status"].fillna("unknown").astype(str).eq(status)]
        color, symbol, size = styles.get(status, (COLORS["unknown"], "circle-open", 8))
        name = status.replace("available / no issue", "No issue reported").replace("missing / problem", "NCO-reported problem").replace("partial / quality", "Partial / quality issue")
        hover = subset.get("station_name", pd.Series("", index=subset.index)).fillna("").astype(str)
        station_id = subset.get("station_id", pd.Series("", index=subset.index)).fillna("").astype(str)
        state = subset.get("state", pd.Series("", index=subset.index)).fillna("").astype(str)
        issue_category = subset.get("issue_category", pd.Series("", index=subset.index)).fillna("").astype(str)
        custom = np.column_stack([station_id, hover, state, issue_category])
        fig.add_trace(
            go.Scattergeo(
                lon=subset["longitude"],
                lat=subset["latitude"],
                mode="markers",
                name=name,
                marker={"size": size, "color": color, "symbol": symbol, "line": {"color": COLORS["text"], "width": 0.7}},
                customdata=custom,
                hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>%{customdata[2]}<br>" + name + "<br>Category: %{customdata[3]}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor=COLORS["panel"],
        font={"family": "Arial, sans-serif", "color": COLORS["text"], "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "xanchor": "left", "x": 0.02, "bgcolor": "rgba(6,21,33,0.78)"},
        geo={
            "scope": "usa",
            "projection_type": "albers usa",
            "fitbounds": "locations",
            "showland": True,
            "landcolor": "#173A52",
            "showlakes": True,
            "lakecolor": COLORS["background"],
            "showocean": True,
            "oceancolor": COLORS["background"],
            "showsubunits": True,
            "subunitcolor": "#4A7087",
            "subunitwidth": 0.8,
            "showcoastlines": True,
            "coastlinecolor": "#4A7087",
            "bgcolor": COLORS["panel"],
        },
    )
    return fig


def nco_ingest_figure(
    nco: pd.DataFrame,
    *,
    show_smoothed_trend: bool = True,
    smooth_window_days: int = 7,
    height: int = 350,
) -> go.Figure:
    if nco.empty or not {"cycle_dt", "conus_count"}.issubset(nco.columns):
        return empty_figure("NCO ingest-count data are unavailable.", height=height)
    data = nco.dropna(subset=["cycle_dt", "conus_count"]).sort_values("cycle_dt")
    if data.empty:
        return empty_figure("NCO ingest-count data are unavailable.", height=height)
    fig = go.Figure()
    palette = {"NAM": COLORS["observed"], "GFS": COLORS["amber"], "NCEP": COLORS["clean"]}
    groups = data.groupby("model") if "model" in data else [("NCO", data)]
    for model, group in groups:
        group = group.sort_values("cycle_dt").copy()
        if not show_smoothed_trend:
            continue
        smoothed = (
            group.set_index("cycle_dt")["conus_count"]
            .rolling(f"{max(2, int(smooth_window_days))}D", min_periods=2)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=group["cycle_dt"],
                y=smoothed.to_numpy(),
                name=f"{model} smoothed ({int(smooth_window_days)}D)",
                mode="lines",
                line={"color": palette.get(str(model), COLORS["unknown"]), "width": 3, "dash": "solid"},
                opacity=0.9,
                hovertemplate=f"%{{x|%b %d, %Y %HZ}}<br>Smoothed {model} ingest ({int(smooth_window_days)}D): %{{y:.1f}}<extra></extra>",
            )
        )
    _base_layout(fig, height=height)
    fig.update_yaxes(title="CONUS RAOBs reported available for ingest", rangemode="tozero")
    fig.update_xaxes(hoverformat="%b %d, %Y %HZ")
    return fig


def issue_category_figure(
    counts: pd.DataFrame,
    *,
    show_smoothed_trend: bool = False,
    smooth_window_days: int = 14,
    height: int = 350,
) -> go.Figure:
    if counts.empty or not {"cycle_dt", "issue_category", "count"}.issubset(counts.columns):
        return empty_figure("NCO issue-category history is unavailable.", height=height)
    palette = {
        "no_report": COLORS["deficit"],
        "unavailable": "#D75555",
        "equipment_failure": COLORS["amber"],
        "missing_parts": "#C89B4B",
        "purged_data": COLORS["observed"],
        "other": COLORS["unknown"],
    }
    fig = go.Figure()
    for category in sorted(counts["issue_category"].dropna().astype(str).unique()):
        subset = counts[counts["issue_category"].astype(str).eq(category)]
        fig.add_trace(
            go.Bar(
                x=subset["cycle_dt"],
                y=subset["count"],
                name=category.replace("_", " ").title(),
                marker={"color": palette.get(category, COLORS["unknown"])},
                hovertemplate="%{x|%b %d, %Y %HZ}<br>Reported statuses: %{y:.0f}<extra>" + category.replace("_", " ").title() + "</extra>",
            )
        )
    if show_smoothed_trend:
        totals = counts.groupby("cycle_dt", as_index=False)["count"].sum().sort_values("cycle_dt")
        smoothed = (
            totals.set_index("cycle_dt")["count"]
            .rolling(f"{max(2, int(smooth_window_days))}D", min_periods=2)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=totals["cycle_dt"],
                y=smoothed.to_numpy(),
                name=f"Smoothed total ({int(smooth_window_days)}D)",
                mode="lines",
                line={"color": COLORS["text"], "width": 3, "dash": "solid"},
                hovertemplate=f"%{{x|%b %d, %Y %HZ}}<br>Smoothed reported statuses ({int(smooth_window_days)}D): %{{y:.1f}}<extra></extra>",
            )
        )
    _base_layout(fig, height=height)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Reported station statuses", rangemode="tozero")
    return fig


def station_archive_deficit_map_figure(frame: pd.DataFrame, metric: str, *, height: int = 430) -> go.Figure:
    if frame.empty or metric not in frame or not {"latitude", "longitude"}.issubset(frame.columns):
        return empty_figure("Station-level archive-deficit data are not available in the current run.", height=height)
    data = frame.dropna(subset=["latitude", "longitude", metric]).copy()
    if data.empty:
        return empty_figure("Station-level archive-deficit data are not available in the current run.", height=height)
    station_id = data.get("station_id", pd.Series("", index=data.index)).fillna("").astype(str)
    station_name = data.get("station_name", pd.Series("", index=data.index)).fillna("").astype(str)
    bound = float(max(10.0, math.ceil(data[metric].abs().max())))
    fig = go.Figure(
        go.Scattergeo(
            lon=data["longitude"],
            lat=data["latitude"],
            mode="markers",
            marker={
                "size": 11,
                "color": data[metric],
                "colorscale": [[0, COLORS["deficit"]], [0.5, "#F4D4CB"], [1, COLORS["observed"]]],
                "cmin": -bound,
                "cmax": bound,
                "cmid": 0,
                "colorbar": {"title": "% vs baseline", "tickfont": {"color": COLORS["muted"]}, "titlefont": {"color": COLORS["muted"]}},
                "line": {"color": COLORS["text"], "width": 0.6},
            },
            customdata=np.column_stack([station_id, station_name, data[metric]]),
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>%{customdata[2]:.1f}% vs baseline<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor=COLORS["panel"],
        font={"family": "Arial, sans-serif", "color": COLORS["text"]},
        geo={
            "scope": "usa",
            "projection_type": "albers usa",
            "fitbounds": "locations",
            "showland": True,
            "landcolor": "#173A52",
            "showsubunits": True,
            "subunitcolor": "#4A7087",
            "showcoastlines": True,
            "coastlinecolor": "#4A7087",
            "bgcolor": COLORS["panel"],
        },
    )
    return fig
