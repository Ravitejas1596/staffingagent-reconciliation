"""Deterministic compliance checks — no LLM."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

from src.state import ReconciliationState

logger = logging.getLogger(__name__)

# RULE: Compliance checks are deterministic rules — no LLM needed


def compliance_node(state: ReconciliationState) -> dict:
    scored = state.get("scored_records") or []
    updated: List[Dict[str, Any]] = []

    for rec in scored:
        r = copy.deepcopy(rec)
        flags: List[str] = []
        placement_id = r.get("placement_id", "")
        candidate = r.get("candidate_name", "")
        logger.info("Processing %s | %s | compliance check", placement_id, candidate)

        vh = float(r.get("vms_hours", 0.0))
        vo = float(r.get("vms_ot_hours", 0.0))
        total_hours = vh + vo
        ats_state = str(r.get("ats_state", ""))
        vms_status = str(r.get("vms_status", ""))

        if total_hours >= 50:
            msg = (
                f"EXCESSIVE HOURS: {total_hours} total hours in one week meets or exceeds 50h threshold — "
                "potential labor law violation. Requires supervisor approval before invoicing."
            )
            flags.append(msg)
            logger.info("Compliance %s | EXCESSIVE HOURS | total_hours=%s", placement_id, total_hours)

        if ats_state == "CA":
            msg = (
                "CA DAILY OT RULE: California Labor Code requires overtime after 8 hours/day "
                "(not just 40h/week). VMS platform reports weekly totals only — cannot confirm "
                "daily OT was applied correctly. Manual verification required."
            )
            flags.append(msg)
            logger.info("Compliance %s | CA DAILY OT", placement_id)

        if vms_status == "Missing Timesheet":
            msg = (
                "MISSING TIMESHEET: Worker has no approved timesheet in VMS. Cannot process payroll "
                "or invoice hospital until timesheet is submitted and approved."
            )
            flags.append(msg)
            logger.info("Compliance %s | MISSING TIMESHEET", placement_id)

        # FLSA OT rate check: federal law requires OT at >= 1.5x the regular rate.
        # If OT hours are present but the OT rate is below 1.5x, the invoice amount
        # is wrong and could expose the firm to a Department of Labor audit.
        vms_bill_rate = float(r.get("vms_bill_rate", 0.0))
        vms_ot_rate = float(r.get("vms_ot_rate", 0.0))
        if vo > 0 and vms_bill_rate > 0:
            expected_ot = vms_bill_rate * 1.5
            if vms_ot_rate < expected_ot - 0.01:
                msg = (
                    f"FLSA OT RATE: Overtime rate ${vms_ot_rate:.2f}/hr is below the "
                    f"required 1.5x regular rate (${expected_ot:.2f}/hr). Federal law "
                    "requires overtime at minimum 1.5x. Correct before invoicing."
                )
                flags.append(msg)
                logger.info(
                    "Compliance %s | FLSA OT RATE | ot_rate=%.2f expected=%.2f",
                    placement_id, vms_ot_rate, expected_ot,
                )

        if vms_status == "Pending Approval":
            msg = (
                "PENDING APPROVAL: VMS record not yet approved by hospital. Cannot invoice until "
                "hospital approves the timesheet."
            )
            flags.append(msg)
            logger.info("Compliance %s | PENDING APPROVAL", placement_id)

        r["compliance_flags"] = flags

        # Hard rule: any compliance flag that blocks invoicing must escalate to human review,
        # even if the record scored clean. The scorer runs before compliance, so it cannot
        # account for these checks — compliance_node must correct that here.
        if flags and not r.get("requires_human_review"):
            r["requires_human_review"] = True
            # Only upgrade severity if it was clean — keep critical/warning as-is
            if r.get("severity") == "clean":
                r["severity"] = "warning"
            # Populate fallback LLM fields — this record never went through llm_reason_node
            # so the HITL reviewer needs guidance from the compliance flags themselves.
            if not r.get("llm_explanation"):
                r["llm_explanation"] = "Compliance rule triggered: " + "; ".join(flags)
            if not r.get("recommended_action"):
                r["recommended_action"] = (
                    "Review the compliance flags for this record and resolve before invoicing."
                )
            if not r.get("risk_level"):
                r["risk_level"] = "MEDIUM"
            if not r.get("escalation_note"):
                r["escalation_note"] = "Middle-office lead should review compliance flags."
            logger.info(
                "Compliance %s | upgraded to HITL — %d flag(s) detected", placement_id, len(flags)
            )

        updated.append(r)

    newly_flagged = [
        r["placement_id"]
        for r in updated
        if r.get("requires_human_review")
    ]
    return {"scored_records": updated, "compliance_results": updated, "flagged_ids": newly_flagged}
