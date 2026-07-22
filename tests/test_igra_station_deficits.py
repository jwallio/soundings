from __future__ import annotations

import datetime as dt
from collections import Counter

from scripts.build_conus_igra_launch_counts_by_year import (
    Station,
    build_station_deficit_rows,
    latest_complete_date,
)


def test_station_deficits_compare_90_days_with_same_date_baseline() -> None:
    end = dt.date(2026, 7, 12)
    counts: Counter[dt.date] = Counter()
    for offset in range(90):
        current = end - dt.timedelta(days=offset)
        counts[current] = 1
        for year in (2021, 2022, 2023, 2024):
            counts[current.replace(year=year)] = 2
    station = Station("USM00000001", 35.0, -97.0, "Test Station", 2000, 2026)
    rows = build_station_deficit_rows(
        [station],
        {station.station_id: counts},
        {station.station_id},
        end,
        [2021, 2022, 2023, 2024],
    )
    assert len(rows) == 1
    assert rows[0]["observed_90"] == 90.0
    assert rows[0]["expected_90"] == 180.0
    assert rows[0]["deficit_90"] == -90.0
    assert rows[0]["missed_90"] == 90.0
    for days in (7, 30, 90, 180, 365):
        assert f"observed_{days}" in rows[0]
        assert f"expected_{days}" in rows[0]
        assert f"deficit_{days}" in rows[0]


def test_partial_latest_date_uses_monitor_threshold() -> None:
    latest = dt.date(2026, 7, 13)
    counts = Counter({latest - dt.timedelta(days=offset): 130 for offset in range(1, 15)})
    counts[latest] = 10
    assert latest_complete_date(counts, latest) == dt.date(2026, 7, 12)
