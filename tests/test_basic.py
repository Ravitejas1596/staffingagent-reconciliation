"""Basic smoke tests for reconciliation pipeline."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import build_graph
from src.compliance import compliance_node
from src.normalizer import CLIENT_ALIASES, normalize_node
from src.scorer import score_node
from src.matcher import match_node
from src.state import ReconciliationState


def _blank_state(**overrides) -> ReconciliationState:
    """Return a minimal valid state dict, with optional field overrides."""
    base: ReconciliationState = {
        "vms_path": None,
        "ats_path": None,
        "vms_normalized": None,
        "ats_normalized": None,
        "matched_records": [],
        "scored_records": [],
        "flagged_ids": [],
        "compliance_results": [],
        "final_results": [],
        "errors": [],
        "run_id": "test",
        "processing_log": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_csv_files_exist() -> None:
    assert (ROOT / "data" / "vms_records.csv").is_file()
    assert (ROOT / "data" / "ats_bullhorn_records.csv").is_file()


def test_normalization() -> None:
    out = normalize_node(_blank_state())
    vms = out["vms_normalized"]
    ats = out["ats_normalized"]
    assert isinstance(vms, list) and len(vms) > 0
    assert isinstance(ats, list) and len(ats) > 0
    vms_clients = {r["client_name"] for r in vms}
    ats_clients = {r["client_name"] for r in ats}
    for raw, canonical in CLIENT_ALIASES.items():
        assert raw not in vms_clients, f"Alias '{raw}' should have been resolved"
        assert canonical in vms_clients, f"Canonical '{canonical}' missing from VMS"
        assert canonical in ats_clients, f"Canonical '{canonical}' missing from ATS"


def test_normalization_custom_paths() -> None:
    """normalize_node respects vms_path / ats_path from state."""
    out = normalize_node(_blank_state(
        vms_path=str(ROOT / "data" / "vms_records.csv"),
        ats_path=str(ROOT / "data" / "ats_bullhorn_records.csv"),
    ))
    assert isinstance(out["vms_normalized"], list) and len(out["vms_normalized"]) == 20
    assert isinstance(out["ats_normalized"], list) and len(out["ats_normalized"]) == 20


def _run_through_score() -> list:
    st = _blank_state()
    n = normalize_node(st)
    st.update(n)
    m = match_node(st)  # type: ignore[arg-type]
    st.update(m)
    s = score_node(st)  # type: ignore[arg-type]
    return list(s["scored_records"])


def test_scoring_missing_timesheet_below_half() -> None:
    scored = _run_through_score()
    pl1008 = next(r for r in scored if r["placement_id"] == "PL-1008")
    assert pl1008["confidence_score"] < 0.50


def test_clean_records() -> None:
    scored = _run_through_score()
    for pid in ("PL-1001", "PL-1015"):
        r = next(x for x in scored if x["placement_id"] == pid)
        assert r["confidence_score"] >= 0.85


def test_excessive_hours_boundary() -> None:
    """PL-1011 has exactly 50 total hours — must trigger the compliance flag."""
    scored = _run_through_score()
    pl1011 = next(r for r in scored if r["placement_id"] == "PL-1011")
    # 45 reg + 5 OT = 50h; >= 50 threshold must fire
    assert pl1011["vms_hours"] + pl1011["vms_ot_hours"] == 50.0


def test_po_cross_comparison() -> None:
    """PO numbers are compared across systems, not just checked for presence."""
    scored = _run_through_score()
    # PL-1010 has a blank VMS PO — po_missing should be True
    pl1010 = next(r for r in scored if r["placement_id"] == "PL-1010")
    assert pl1010["po_missing"] is True


def test_outer_join_no_silent_drops() -> None:
    """All 20 placement IDs from both CSVs are present in matched output."""
    st = _blank_state()
    n = normalize_node(st)
    st.update(n)
    m = match_node(st)  # type: ignore[arg-type]
    ids = {r["placement_id"] for r in m["matched_records"]}
    assert len(ids) == 20


def test_compliance_escalates_to_hitl() -> None:
    """Records that score 'clean' must still be flagged for HITL when compliance fires.

    PL-1006 (David Park, CA), PL-1009 (Lisa Wang, CA), PL-1020 (Ryan Jackson, CA)
    all have confidence=1.0 from the scorer but are in California, so compliance_node
    must add the CA DAILY OT flag and upgrade requires_human_review to True.
    """
    st = _blank_state()
    n = normalize_node(st)
    st.update(n)
    m = match_node(st)  # type: ignore[arg-type]
    st.update(m)
    s = score_node(st)  # type: ignore[arg-type]
    st.update(s)
    c = compliance_node(st)  # type: ignore[arg-type]
    after = {r["placement_id"]: r for r in c["scored_records"]}

    for pid in ("PL-1006", "PL-1009", "PL-1020"):
        r = after[pid]
        assert r["confidence_score"] == 1.0, f"{pid} should still have confidence 1.0"
        assert r["requires_human_review"] is True, f"{pid} should be HITL after compliance"
        assert any("CA DAILY OT" in f for f in r["compliance_flags"]), (
            f"{pid} should have CA DAILY OT flag"
        )
        assert r["severity"] != "clean", f"{pid} severity should be upgraded from clean"


def test_normalizer_rejects_missing_columns() -> None:
    """normalize_node must raise ValueError when a required CSV column is absent."""
    import tempfile, os

    # Write a VMS CSV that is missing vms_hours
    bad_csv = "placement_id,candidate_name,vms_bill_rate\nPL-1,Test User,50.0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(bad_csv)
        bad_path = f.name

    good_ats = str(ROOT / "data" / "ats_bullhorn_records.csv")
    try:
        normalize_node(_blank_state(vms_path=bad_path, ats_path=good_ats))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "vms_hours" in str(e).lower() or "missing" in str(e).lower()
    finally:
        os.unlink(bad_path)


def test_output_schema() -> None:
    run_id = str(uuid.uuid4())
    initial_state = _blank_state(run_id=run_id)
    result = build_graph().invoke(initial_state)
    data = result.get("final_results") or []
    required = {
        "placement_id",
        "candidate_name",
        "vms_platform",
        "client_name",
        "ats_state",
        "ats_job_title",
        "week_ending",
        "match_status",
        "confidence_score",
        "requires_human_review",
        "severity",
        "discrepancies",
        "vms_snapshot",
        "ats_snapshot",
        "financial_impact",
        "name_similarity_score",
        "compliance_flags",
        "llm_explanation",
        "recommended_action",
        "risk_level",
        "escalation_note",
    }
    assert len(data) == 20
    for row in data:
        assert required <= set(row.keys())

    out_path = ROOT / "output" / "reconciliation_results.json"
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(doc.keys()) == {"run_metadata", "records"}
    meta = doc["run_metadata"]
    assert meta["run_id"] == run_id
    assert "generated_at" in meta and meta["generated_at"]
    assert "summary_stats" in meta
    ss = meta["summary_stats"]
    assert ss["record_count"] == 20
    assert "models_used" in meta
    assert "llm_reasoning" in meta["models_used"]
    assert len(doc["records"]) == 20
    for row in doc["records"]:
        assert required <= set(row.keys())
