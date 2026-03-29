"""Join VMS/ATS and detect per-placement discrepancies."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from rapidfuzz import fuzz

from src.state import ReconciliationState

logger = logging.getLogger(__name__)


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return str(val) if val is not None else default


def _build_missing_record(placement_id: str, row: pd.Series, missing_side: str) -> Dict[str, Any]:
    """Build a minimal record for a placement that exists in only one system."""
    if missing_side == "ats":
        # VMS record with no ATS counterpart
        name = _safe_str(row.get("candidate_name_vms") or row.get("candidate_name"))
        discrepancy = "VMS record with no matching ATS placement — orphaned timesheet"
        client = _safe_str(row.get("client_name_vms") or row.get("client_name"))
        vms_status = _safe_str(row.get("vms_status"))
    else:
        # ATS record with no VMS counterpart
        name = _safe_str(row.get("candidate_name_ats") or row.get("candidate_name"))
        discrepancy = "ATS record with no matching VMS timesheet — possible missing submission"
        client = _safe_str(row.get("client_name_ats") or row.get("client_name"))
        vms_status = "Missing From VMS"

    return {
        "placement_id": placement_id,
        "candidate_name": name,
        "ats_candidate_name": name,
        "vms_platform": _safe_str(row.get("vms_platform")),
        "client_name": client,
        "ats_client_name": client,
        "vms_period_end": _safe_str(row.get("vms_period_end")),
        "ats_job_title": _safe_str(row.get("ats_job_title")),
        "ats_state": _safe_str(row.get("ats_state")),
        "vms_hours": _safe_float(row.get("vms_hours")),
        "vms_ot_hours": _safe_float(row.get("vms_ot_hours")),
        "vms_bill_rate": _safe_float(row.get("vms_bill_rate")),
        "vms_ot_rate": _safe_float(row.get("vms_ot_rate")),
        "vms_per_diem": _safe_float(row.get("vms_per_diem")),
        "vms_total": _safe_float(row.get("vms_total")),
        "vms_po_number": _safe_str(row.get("vms_po_number"), "MISSING"),
        "vms_status": vms_status,
        "ats_hours": _safe_float(row.get("ats_hours")),
        "ats_ot_hours": _safe_float(row.get("ats_ot_hours")),
        "ats_bill_rate": _safe_float(row.get("ats_bill_rate")),
        "ats_ot_rate": _safe_float(row.get("ats_ot_rate")),
        "ats_per_diem": _safe_float(row.get("ats_per_diem")),
        "ats_total": _safe_float(row.get("ats_total")),
        "ats_po_number": _safe_str(row.get("ats_po_number"), "MISSING"),
        "ats_pay_rate": _safe_float(row.get("ats_pay_rate")),
        "name_similarity_score": 0,
        "hours_mismatch": False,
        "ot_hours_mismatch": False,
        "bill_rate_mismatch": False,
        "ot_rate_mismatch": False,
        "per_diem_mismatch": False,
        "total_mismatch": False,
        "po_missing": True,
        "po_mismatch": False,
        "name_mismatch": False,
        "status_issue": True,
        "client_mismatch": False,
        "discrepancy_list": [discrepancy],
        "record_type": f"{'vms' if missing_side == 'ats' else 'ats'}_only",
    }


def match_node(state: ReconciliationState) -> dict:
    """Outer-join on placement_id; fuzzy names; build discrepancy_list per row.

    Uses an outer join so records present in only one system are surfaced as
    'missing' rather than silently dropped (was an inner join).

    Fuzzy name matching uses the pre-stripped compare fields (middle initials
    removed) produced by normalize_node, so 'Angela M. Davis' == 'Angela Davis'.
    Raw display names are preserved in all output fields.

    PO numbers are cross-compared: VMS-PO missing AND VMS-PO≠ATS-PO are flagged
    as distinct discrepancies.
    """
    vms_records = state.get("vms_normalized")
    ats_records = state.get("ats_normalized")
    if not vms_records or not ats_records:
        logger.error("match_node | missing normalized records")
        return {"matched_records": [], "errors": ["match_node: missing normalized data"]}

    vms = pd.DataFrame(vms_records)
    ats = pd.DataFrame(ats_records)

    merged = pd.merge(
        vms,
        ats,
        on="placement_id",
        how="outer",       # outer join: captures records missing from either side
        suffixes=("_vms", "_ats"),
    )
    matched: List[dict] = []

    for _, row in merged.iterrows():
        placement_id = str(row["placement_id"])

        # Detect records that exist in only one system
        vms_absent = pd.isna(row.get("vms_hours"))
        ats_absent = pd.isna(row.get("ats_hours"))

        if vms_absent:
            rec = _build_missing_record(placement_id, row, "vms")
            matched.append(rec)
            logger.info("Match decision %s | ATS-only record (no VMS timesheet)", placement_id)
            continue

        if ats_absent:
            rec = _build_missing_record(placement_id, row, "ats")
            matched.append(rec)
            logger.info("Match decision %s | VMS-only record (no ATS placement)", placement_id)
            continue

        # Both sides present — normal comparison path
        name_vms = _safe_str(row.get("candidate_name_vms"))
        name_ats = _safe_str(row.get("candidate_name_ats"))

        # Use stripped-compare fields (middle initials removed) for similarity scoring
        compare_vms = _safe_str(row.get("vms_candidate_compare"), name_vms)
        compare_ats = _safe_str(row.get("ats_candidate_compare"), name_ats)
        name_similarity_score = int(fuzz.token_sort_ratio(compare_vms, compare_ats))

        vh = _safe_float(row.get("vms_hours"))
        vo = _safe_float(row.get("vms_ot_hours"))
        ah = _safe_float(row.get("ats_hours"))
        ao = _safe_float(row.get("ats_ot_hours"))
        vbr = _safe_float(row.get("vms_bill_rate"))
        abr = _safe_float(row.get("ats_bill_rate"))
        vor = _safe_float(row.get("vms_ot_rate"))
        aor = _safe_float(row.get("ats_ot_rate"))
        vpd = _safe_float(row.get("vms_per_diem"))
        apd = _safe_float(row.get("ats_per_diem"))
        vt = _safe_float(row.get("vms_total"))
        atot = _safe_float(row.get("ats_total"))
        vpo = _safe_str(row.get("vms_po_number"), "MISSING")
        apo = _safe_str(row.get("ats_po_number"), "MISSING")
        vstatus = _safe_str(row.get("vms_status"))
        client_v = _safe_str(row.get("client_name_vms")).strip()
        client_a = _safe_str(row.get("client_name_ats")).strip()

        hours_mismatch = abs(vh - ah) > 0.01
        ot_hours_mismatch = abs(vo - ao) > 0.01
        bill_rate_mismatch = abs(vbr - abr) > 0.01
        ot_rate_mismatch = abs(vor - aor) > 0.01
        per_diem_mismatch = abs(vpd - apd) > 0.01
        total_mismatch = abs(vt - atot) > 0.01
        po_missing = vpo == "MISSING" or apo == "MISSING"
        # Cross-compare POs when both sides have values (catches PO-ABC vs PO-XYZ mismatches)
        po_mismatch = (not po_missing) and (vpo != apo)
        name_mismatch = name_similarity_score < 85
        status_issue = vstatus not in ("Approved",)
        client_mismatch = client_v != client_a

        logger.info(
            "Processing %s | %s | name_similarity=%s (compare: '%s' vs '%s')",
            placement_id,
            name_vms,
            name_similarity_score,
            compare_vms,
            compare_ats,
        )

        discrepancy_list: List[str] = []
        if hours_mismatch:
            discrepancy_list.append(
                f"Hours mismatch: VMS={vh:.1f}h vs ATS={ah:.1f}h"
            )
        if ot_hours_mismatch:
            discrepancy_list.append(
                f"OT hours mismatch: VMS={vo:.1f}h vs ATS={ao:.1f}h"
            )
        if bill_rate_mismatch:
            discrepancy_list.append(
                f"Bill rate mismatch: VMS={_fmt_money(vbr)}/hr vs ATS={_fmt_money(abr)}/hr "
                f"(diff={_fmt_money(abs(vbr - abr))})"
            )
        if ot_rate_mismatch:
            discrepancy_list.append(
                f"OT rate mismatch: VMS={_fmt_money(vor)}/hr vs ATS={_fmt_money(aor)}/hr"
            )
        if per_diem_mismatch:
            discrepancy_list.append(
                f"Per diem mismatch: VMS={_fmt_money(vpd)} vs ATS={_fmt_money(apd)}"
            )
        if total_mismatch:
            discrepancy_list.append(
                f"Total amount mismatch: VMS={_fmt_money(vt)} vs ATS={_fmt_money(atot)} "
                f"(diff={_fmt_money(abs(vt - atot))})"
            )
        if po_missing:
            if vpo == "MISSING" and apo == "MISSING":
                discrepancy_list.append("PO number missing from both VMS and ATS records")
            elif vpo == "MISSING":
                discrepancy_list.append("Missing PO number on VMS record")
            else:
                discrepancy_list.append("Missing PO number on ATS record")
        if po_mismatch:
            discrepancy_list.append(
                f"PO number mismatch: VMS='{vpo}' vs ATS='{apo}'"
            )
        if name_mismatch:
            discrepancy_list.append(
                f"Name variation: VMS='{name_vms}' vs ATS='{name_ats}' "
                f"(similarity={name_similarity_score})"
            )
        if status_issue:
            if vstatus == "Pending Approval":
                discrepancy_list.append(
                    "VMS status: Pending Approval — not yet approved for invoicing"
                )
            elif vstatus == "Missing Timesheet":
                discrepancy_list.append(
                    "VMS status: Missing Timesheet — no approved timesheet in VMS"
                )
            else:
                discrepancy_list.append(f"VMS status not Approved: {vstatus}")
        if client_mismatch:
            discrepancy_list.append(
                f"Client name mismatch after alias resolution: VMS='{client_v}' vs ATS='{client_a}'"
            )

        rec: Dict[str, Any] = {
            "placement_id": placement_id,
            "candidate_name": name_vms,
            "ats_candidate_name": name_ats,
            "vms_platform": _safe_str(row.get("vms_platform")),
            "client_name": client_v,
            "ats_client_name": client_a,
            "vms_period_end": _safe_str(row.get("vms_period_end")),
            "ats_job_title": _safe_str(row.get("ats_job_title")),
            "ats_state": _safe_str(row.get("ats_state")),
            "vms_hours": vh,
            "vms_ot_hours": vo,
            "vms_bill_rate": vbr,
            "vms_ot_rate": vor,
            "vms_per_diem": vpd,
            "vms_total": vt,
            "vms_po_number": vpo,
            "vms_status": vstatus,
            "ats_hours": ah,
            "ats_ot_hours": ao,
            "ats_bill_rate": abr,
            "ats_ot_rate": aor,
            "ats_per_diem": apd,
            "ats_total": atot,
            "ats_po_number": apo,
            "ats_pay_rate": _safe_float(row.get("ats_pay_rate")),
            "name_similarity_score": name_similarity_score,
            "hours_mismatch": hours_mismatch,
            "ot_hours_mismatch": ot_hours_mismatch,
            "bill_rate_mismatch": bill_rate_mismatch,
            "ot_rate_mismatch": ot_rate_mismatch,
            "per_diem_mismatch": per_diem_mismatch,
            "total_mismatch": total_mismatch,
            "po_missing": po_missing,
            "po_mismatch": po_mismatch,
            "name_mismatch": name_mismatch,
            "status_issue": status_issue,
            "client_mismatch": client_mismatch,
            "discrepancy_list": discrepancy_list,
            "record_type": "full_match",
        }
        matched.append(rec)
        logger.info(
            "Match decision %s | %s | discrepancies=%s",
            placement_id,
            name_vms,
            len(discrepancy_list),
        )

    return {"matched_records": matched}
