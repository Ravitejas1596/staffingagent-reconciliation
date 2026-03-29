"""Write reconciliation_results.json and print terminal summary."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.llm_reasoner import MODEL_ID
from src.state import ReconciliationState

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
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
)


def _build_summary_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    weeks = sorted(set(r["week_ending"] for r in rows)) if rows else []
    week = ", ".join(weeks) if weeks else None
    clean_ids = [r["placement_id"] for r in rows if r["severity"] == "clean"]
    hitl_n = sum(1 for r in rows if r["requires_human_review"])
    at_risk = sum(r["financial_impact"] for r in rows if r["requires_human_review"])
    return {
        "record_count": n,
        "week_ending": week,
        "clean_matches_count": len(clean_ids),
        "clean_placement_ids": clean_ids,
        "critical_count": sum(1 for r in rows if r["severity"] == "critical"),
        "warning_count": sum(1 for r in rows if r["severity"] == "warning"),
        "info_count": sum(1 for r in rows if r["severity"] == "info"),
        "missing_timesheets_count": sum(1 for r in rows if r["match_status"] == "missing"),
        "requires_human_review_count": hitl_n,
        "total_financial_impact_at_risk": round(at_risk, 2),
    }


def _default_action_for_discrepancies(discrepancies: List[str]) -> str:
    """Generate a plain-English action string for non-HITL records that still have discrepancies."""
    text = " ".join(discrepancies).lower()
    if "po" in text and ("missing" in text or "blank" in text):
        return (
            "Contact the hospital to obtain the Purchase Order (PO) number for this placement. "
            "Invoicing cannot proceed without a valid PO number recorded on both systems."
        )
    if "po" in text and "mismatch" in text:
        return (
            "The PO number in the hospital system does not match our records. "
            "Confirm the correct PO number with the client and update both systems before invoicing."
        )
    if "hours" in text:
        return (
            "Review the hours reported in the hospital system against our Bullhorn records "
            "and correct any discrepancy before submitting for payroll."
        )
    if "bill rate" in text or "rate" in text:
        return (
            "The bill rate does not match the contracted amount. "
            "Verify the correct rate with the client and update as needed before invoicing."
        )
    if "name" in text:
        return (
            "Verify the candidate name matches exactly across both systems to prevent payroll issues."
        )
    if "per diem" in text:
        return (
            "The per diem allowance does not match between systems. "
            "Confirm the correct per diem amount and update both records before invoicing."
        )
    if "total" in text and ("amount" in text or "mismatch" in text):
        return (
            "The total invoice amount differs between systems. "
            "Verify hours and rates on both sides and correct the discrepancy before submitting for payment."
        )
    if "status" in text:
        return (
            "The VMS record status is not yet Approved. "
            "Follow up with the hospital to get the timesheet approved before invoicing."
        )
    # Generic fallback for any other discrepancy
    parts = discrepancies[:2]
    issues = "; ".join(parts)
    return f"Review and resolve the following before processing: {issues}."


def output_node(state: ReconciliationState) -> dict:
    scored = state.get("scored_records") or []
    run_id = str(state.get("run_id", ""))
    final_results: List[Dict[str, Any]] = []

    for rec in scored:
        pid = rec.get("placement_id", "")
        logger.info("output_node | building final record for %s", pid)

        vt = float(rec.get("vms_total", 0.0))
        atot = float(rec.get("ats_total", 0.0))
        fin = abs(vt - atot)

        llm_exp = str(rec.get("llm_explanation", "") or "")
        rec_action = str(rec.get("recommended_action", "") or "")
        risk = str(rec.get("risk_level", "") or "")
        esc = str(rec.get("escalation_note", "") or "")

        discrepancies = list(rec.get("discrepancies") or [])
        if not rec.get("requires_human_review"):
            llm_exp = llm_exp or ""
            risk = risk or "LOW"
            esc = esc or ""
            # Only say "no action" when the record is genuinely clean — no discrepancies at all.
            # A record can have discrepancies (e.g. missing PO) but still have been routed through
            # the non-LLM path; in that case generate a plain-English action instead.
            if not discrepancies:
                rec_action = rec_action or "No action required — record cleared for processing."
            else:
                rec_action = rec_action or _default_action_for_discrepancies(discrepancies)

        row: Dict[str, Any] = {
            "placement_id": str(rec.get("placement_id", "")),
            "candidate_name": str(rec.get("candidate_name", "")),
            "vms_platform": str(rec.get("vms_platform", "")),
            "client_name": str(rec.get("client_name", "")),
            "ats_state": str(rec.get("ats_state", "")),
            "ats_job_title": str(rec.get("ats_job_title", "")),
            "week_ending": str(rec.get("vms_period_end", "")),
            "match_status": rec.get("match_status", "mismatch"),
            "confidence_score": float(rec.get("confidence_score", 0.0)),
            "requires_human_review": bool(rec.get("requires_human_review", False)),
            "severity": str(rec.get("severity", "warning")),
            "discrepancies": list(rec.get("discrepancies", [])),
            "vms_snapshot": {
                "hours": float(rec.get("vms_hours", 0.0)),
                "ot_hours": float(rec.get("vms_ot_hours", 0.0)),
                "bill_rate": float(rec.get("vms_bill_rate", 0.0)),
                "ot_rate": float(rec.get("vms_ot_rate", 0.0)),
                "per_diem": float(rec.get("vms_per_diem", 0.0)),
                "total": vt,
                "po_number": str(rec.get("vms_po_number", "")),
                "status": str(rec.get("vms_status", "")),
            },
            "ats_snapshot": {
                "hours": float(rec.get("ats_hours", 0.0)),
                "ot_hours": float(rec.get("ats_ot_hours", 0.0)),
                "bill_rate": float(rec.get("ats_bill_rate", 0.0)),
                "ot_rate": float(rec.get("ats_ot_rate", 0.0)),
                "per_diem": float(rec.get("ats_per_diem", 0.0)),
                "total": atot,
                "po_number": str(rec.get("ats_po_number", "")),
                "pay_rate": float(rec.get("ats_pay_rate", 0.0)),
            },
            "financial_impact": fin,
            "name_similarity_score": int(rec.get("name_similarity_score", 0)),
            "compliance_flags": list(rec.get("compliance_flags", [])),
            "llm_explanation": llm_exp,
            "recommended_action": rec_action,
            "risk_level": risk,
            "escalation_note": esc,
        }
        final_results.append(row)

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reconciliation_results.json"
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary_stats = _build_summary_stats(final_results)
    payload: Dict[str, Any] = {
        "run_metadata": {
            "run_id": run_id,
            "generated_at": generated_at,
            "summary_stats": summary_stats,
            "models_used": {
                "llm_reasoning": MODEL_ID,
            },
        },
        "records": final_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", out_path)

    _write_dashboard(final_results, out_dir)
    _print_summary(final_results)

    return {"final_results": final_results}


def _write_dashboard(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Regenerate dashboard/hitl_dashboard.html with the latest embedded data.

    The dashboard HTML file contains a /*__RECON_DATA_START__*/.../*__RECON_DATA_END__*/
    marker block that is replaced with the fresh JSON after each pipeline run.
    If the file doesn't exist (first-ever run) it is skipped and a warning is logged.
    """
    import re as _re

    dashboard_path = out_dir.parent / "dashboard" / "hitl_dashboard.html"
    if not dashboard_path.exists():
        logger.warning("Dashboard template not found at %s — skipping dashboard update", dashboard_path)
        return
    try:
        html = dashboard_path.read_text(encoding="utf-8")
        new_data = json.dumps(records, separators=(",", ":"))
        replacement = f"/*__RECON_DATA_START__*/{new_data}/*__RECON_DATA_END__*/"
        # Use a lambda so regex engine never interprets the replacement string
        updated, n = _re.subn(
            r"/\*__RECON_DATA_START__\*/\[[\s\S]*?\]/\*__RECON_DATA_END__\*/",
            lambda _: replacement,
            html,
        )
        if n == 0:
            logger.warning("Dashboard data marker not found — dashboard NOT updated")
            return
        dashboard_path.write_text(updated, encoding="utf-8")
        logger.info("Dashboard updated: %s (%s records)", dashboard_path, len(records))
    except Exception as e:
        logger.warning("Dashboard update failed: %s", e)


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    weeks = sorted(set(r["week_ending"] for r in rows)) if rows else []
    week = ", ".join(weeks) if weeks else "unknown"

    clean_ids = [r["placement_id"] for r in rows if r["severity"] == "clean"]
    critical_n = sum(1 for r in rows if r["severity"] == "critical")
    warning_n = sum(1 for r in rows if r["severity"] == "warning")
    missing_n = sum(1 for r in rows if r["match_status"] == "missing")
    hitl_n = sum(1 for r in rows if r["requires_human_review"])

    at_risk = sum(r["financial_impact"] for r in rows if r["requires_human_review"])

    print()
    print("============================================")
    print("  STAFFINGAGENT.AI — RECONCILIATION REPORT")
    print(f"  Week Ending: {week} | Records: {n}")
    print("============================================")
    print(f"  ✅ Clean matches:     {len(clean_ids)}  ({', '.join(clean_ids)})")
    print(f"  🔴 Critical issues:   {critical_n}")
    print(f"  🟡 Warnings:          {warning_n}")
    miss_list = [r["placement_id"] for r in rows if r["match_status"] == "missing"]
    print(f"  🔵 Missing timesheets:{missing_n}  ({', '.join(miss_list)})")
    print(f"  👤 Requires HITL:     {hitl_n}")
    print(f"  💰 Total $ at risk:   ${at_risk:,.2f}")
    print("============================================")
    print("  FLAGGED FOR HUMAN REVIEW:")
    flagged = [r for r in rows if r["requires_human_review"]]
    sev_rank = {"critical": 0, "warning": 1, "info": 2, "clean": 3}
    flagged.sort(key=lambda x: (sev_rank.get(x["severity"], 9), x["placement_id"]))
    for r in flagged:
        sev = r["severity"].upper()
        print(
            f"  {r['placement_id']} | {r['candidate_name']} | "
            f"score={r['confidence_score']:.2f} | {sev} | "
            f"${r['financial_impact']:,.2f} at risk"
        )
    print("============================================")
    print()
