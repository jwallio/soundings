from __future__ import annotations

import json

from scripts.parse_nco_sdm_raob_messages import parse_messages
from scripts.parse_nco_sdm_raob_messages import download_messages


def test_parse_messages_accepts_legacy_iem_availability_wording() -> None:
    text = """
    SENIOR DUTY METEOROLOGIST NWS ADMINISTRATIVE MESSAGE
    NWS NCEP CENTRAL OPERATIONS COLLEGE PARK MD
    0138Z SAT JAN 25 2025
    The 00Z NAM began and is running on time with 13 Alaskan...25 Canadian...
    68 CONUS...14 Mexican...and 4 Caribbean raobs available for ingest.
    00Z NAM RAOB RECAP...
    70414/SYA - No report
    """

    availability, issues = parse_messages(text)

    assert len(availability) == 1
    assert availability[0]["model"] == "NAM"
    assert availability[0]["cycle_date_utc"] == "2025-01-25"
    assert availability[0]["conus_count"] == "68"
    assert availability[0]["pacific_count"] == ""
    assert issues[0]["station_id"] == "SYA"


def test_parse_messages_keeps_generic_ncep_production_as_own_series() -> None:
    text = """
    SENIOR DUTY METEOROLOGIST NWS ADMINISTRATIVE MESSAGE
    NWS NCEP CENTRAL OPERATIONS COLLEGE PARK MD
    0329Z FRI MAY 22 2026
    The 00Z NCEP model production suite has started and is running on time
    with 7 Alaskan, 25 Canadian, 58 CONUS, 15 Mexican, 6 Caribbean, and 10
    Pacific stations available for ingest.
    """

    availability, issues = parse_messages(text)

    assert len(availability) == 1
    assert not issues
    assert availability[0]["model"] == "NCEP"
    assert availability[0]["cycle_hour"] == "00"
    assert availability[0]["conus_count"] == "58"


def test_iem_download_writes_raw_snapshot_and_receipt(monkeypatch, tmp_path) -> None:
    class Response:
        text = "ADMSDM test"
        content = text.encode("utf-8")
        url = "https://example.test/admsdm"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("scripts.parse_nco_sdm_raob_messages.requests.get", lambda *args, **kwargs: Response())
    result = download_messages(
        source="iem",
        start="2025-01-01",
        end="2025-01-31",
        archive_dir=tmp_path,
    )

    assert result == "ADMSDM test"
    raw = tmp_path / "admsdm_2025-01-01_2025-01-31.txt"
    receipt = tmp_path / "admsdm_2025-01-01_2025-01-31.json"
    assert raw.read_text(encoding="utf-8") == "ADMSDM test"
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    assert metadata["product"] == "ADMSDM"
    assert metadata["raw_file"] == raw.name
