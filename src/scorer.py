"""Deterministic confidence scoring — no LLM."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.state import ReconciliationState

logger = logging.getLogger(__name__)


def score_node(state: ReconciliationState) -> dict:
    # RULE: Confidence scoring is deterministic — LLM adds no value here
    matched = state.get("matched_records") or []
    scored: List[Dict[str, Any]] = []
    flagged_ids: List[str] = []

    for rec in matched:
        placement_id = rec["placement_id"]
        candidate_name = rec["candidate_name"]
        logger.info("Processing %s | %s", placement_id, candidate_name)

        vms_status = str(rec.get("vms_status", ""))
        missing_timesheet = vms_status == "Missing Timesheet"
        record_type = rec.get("record_type", "full_match")
        is_orphaned = record_type in ("vms_only", "ats_only")

        vt = float(rec["vms_total"])
        atot = float(rec["ats_total"])
        total_diff = abs(vt - atot)

        if is_orphaned:
            confidence = 0.0
        elif missing_timesheet:
            confidence = 0.35
        else:
            confidence = 1.0
            if rec.get("hours_mismatch"):
                confidence -= 0.25
            if rec.get("ot_hours_mismatch"):
                confidence -= 0.10
            if rec.get("bill_rate_mismatch"):
                confidence -= 0.20
            if rec.get("ot_rate_mismatch"):       # was silently unpenalized
                confidence -= 0.10
            if rec.get("per_diem_mismatch"):       # was silently unpenalized
                confidence -= 0.05
            if total_diff >= 500.0:
                confidence -= 0.20
            elif total_diff >= 100.0:
                confidence -= 0.10
            if rec.get("po_missing"):
                confidence -= 0.15
            if rec.get("po_mismatch"):             # VMS PO ≠ ATS PO (both present)
                confidence -= 0.10
            if rec.get("name_mismatch"):
                confidence -= 0.10
            if rec.get("status_issue"):
                confidence -= 0.15
            if rec.get("client_mismatch"):
                confidence -= 0.05

        confidence = max(0.0, min(1.0, confidence))

        # Hard business rules: these conditions block invoicing regardless of confidence score.
        # - Missing/mismatched PO: cannot legally invoice without a valid PO on both systems.
        # - Status issue: Pending Approval / Disputed records cannot be invoiced yet.
        # - Large financial risk: variances > $500 require manual reconciliation before payment.
        po_issue = bool(rec.get("po_missing") or rec.get("po_mismatch"))
        status_issue = bool(rec.get("status_issue"))
        large_financial_risk = total_diff >= 500.0
        requires_human_review = confidence <= 0.85 or po_issue or status_issue or large_financial_risk
        discrepancies = list(rec.get("discrepancy_list") or [])
        has_discrepancies = len(discrepancies) > 0

        if is_orphaned or missing_timesheet:
            match_status = "missing"
        elif confidence > 0.85 and not has_discrepancies:
            match_status = "match"
        else:
            match_status = "mismatch"

        if requires_human_review:
            flagged_ids.append(placement_id)

        only_minor = False
        if requires_human_review and not missing_timesheet and not is_orphaned:
            non_minor = any(
                rec.get(k)
                for k in (
                    "hours_mismatch",
                    "ot_hours_mismatch",
                    "bill_rate_mismatch",
                    "ot_rate_mismatch",
                    "per_diem_mismatch",
                    "total_mismatch",
                    "po_missing",
                    "po_mismatch",
                    "status_issue",
                )
            )
            only_minor = not non_minor and (
                rec.get("name_mismatch") or rec.get("client_mismatch")
            )

        if is_orphaned or missing_timesheet or confidence < 0.50 or total_diff >= 500.0:
            severity = "critical"
        elif confidence > 0.85 and not has_discrepancies and not requires_human_review:
            severity = "clean"
        elif only_minor:
            severity = "info"
        elif requires_human_review:
            severity = "warning"
        else:
            severity = "warning"

        hitl = "True" if requires_human_review else "False"
        logger.info(
            "%s | score=%.2f | status=%s | severity=%s | HITL=%s",
            placement_id,
            confidence,
            match_status,
            severity,
            hitl,
        )

        out = dict(rec)
        out["confidence_score"] = confidence
        out["match_status"] = match_status
        out["requires_human_review"] = requires_human_review
        out["severity"] = severity
        out["discrepancies"] = discrepancies
        out["financial_impact_abs"] = total_diff
        scored.append(out)

    return {"scored_records": scored, "flagged_ids": flagged_ids}
