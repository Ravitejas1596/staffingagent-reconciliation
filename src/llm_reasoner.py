"""LLM reasoning for HITL-flagged placements."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List

import anthropic

from src.state import ReconciliationState

logger = logging.getLogger(__name__)

# LLM: Claude adds value here — explaining WHY discrepancies matter
# and recommending specific actions the middle-office team should take.
# Rules can detect the mismatch; Claude explains the business impact.

# Model is configurable via LLM_MODEL env var so cost tier can be swapped
# without touching code (e.g. claude-haiku-4-5-20251001 for low-complexity batches).
MODEL_ID = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Minimum seconds between consecutive API calls — prevents burst rate-limit errors
# at scale (2000+ placements/week means potentially hundreds of flagged records).
_INTER_CALL_DELAY_S = 0.25

# Retry config: up to 3 attempts with exponential backoff (1s, 2s, 4s)
_MAX_RETRIES = 3


def _fallback_response() -> Dict[str, Any]:
    return {
        "explanation": "LLM reasoning unavailable — manual review required",
        "recommended_action": "Manually verify VMS vs ATS and resolve discrepancies before invoicing.",
        "risk_level": "MEDIUM",
        "can_auto_resolve": False,
        "escalation_note": "Middle-office lead should review source documents.",
    }


def _call_claude_with_retry(client: anthropic.Anthropic, prompt: str, pid: str) -> Dict[str, Any]:
    """Call Claude with exponential-backoff retry on transient errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in msg.content:
                if hasattr(block, "text"):
                    text += block.text
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)
            return json.loads(text)
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
            logger.warning(
                "Claude API error for %s (attempt %s/%s): %s — retrying in %ss",
                pid, attempt, _MAX_RETRIES, e, wait,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Claude call failed for %s: %s", pid, e)
            break
    return _fallback_response()


def llm_reason_node(state: ReconciliationState) -> dict:
    flagged = list(state.get("flagged_ids") or [])
    scored: List[Dict[str, Any]] = list(state.get("scored_records") or [])
    if not flagged:
        logger.info("llm_reason_node | no flagged records; skipping Claude")
        return {}

    by_id = {r["placement_id"]: r for r in scored}
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_key:
        logger.warning("ANTHROPIC_API_KEY not set; using LLM fallback for flagged records")
    client = anthropic.Anthropic() if has_key else None

    for i, pid in enumerate(flagged):
        rec = by_id.get(pid)
        if rec is None:
            continue
        candidate_name = rec.get("candidate_name", "")
        logger.info("Calling Claude for %s — %s", pid, candidate_name)
        t0 = time.perf_counter()

        # Throttle: pause between calls to avoid burst rate-limit errors at scale
        if i > 0:
            time.sleep(_INTER_CALL_DELAY_S)

        discrepancy_list = "\n".join(f"- {d}" for d in rec.get("discrepancies", []))
        prompt = f"""You are a healthcare staffing middle-office reconciliation expert.

A reconciliation agent has flagged this placement record for human review.

Placement ID: {rec.get("placement_id")}
Candidate: {rec.get("candidate_name")}
Job Title: {rec.get("ats_job_title")}
State: {rec.get("ats_state")}
VMS Platform: {rec.get("vms_platform")}
Client: {rec.get("client_name")}
Week Ending: {rec.get("vms_period_end")}

VMS Record:
- Regular Hours: {rec.get("vms_hours", 0.0):.1f}
- OT Hours: {rec.get("vms_ot_hours", 0.0):.1f}
- Bill Rate: ${rec.get("vms_bill_rate", 0.0):.2f}/hr
- OT Rate: ${rec.get("vms_ot_rate", 0.0):.2f}/hr
- Per Diem: ${rec.get("vms_per_diem", 0.0):.2f}
- Total: ${rec.get("vms_total", 0.0):.2f}
- PO Number: {rec.get("vms_po_number", "N/A")}
- Status: {rec.get("vms_status", "Unknown")}

ATS/Bullhorn Record:
- Regular Hours: {rec.get("ats_hours", 0.0):.1f}
- OT Hours: {rec.get("ats_ot_hours", 0.0):.1f}
- Bill Rate: ${rec.get("ats_bill_rate", 0.0):.2f}/hr
- OT Rate: ${rec.get("ats_ot_rate", 0.0):.2f}/hr
- Per Diem: ${rec.get("ats_per_diem", 0.0):.2f}
- Total: ${rec.get("ats_total", 0.0):.2f}
- PO Number: {rec.get("ats_po_number", "N/A")}

Discrepancies detected:
{discrepancy_list}

Confidence score: {rec.get("confidence_score", 0.0):.2f}

Provide a JSON response with exactly these fields:
{{
  "explanation": "2-3 sentence plain English explanation of what went wrong and why it matters financially",
  "recommended_action": "specific action the middle-office team should take before this can be invoiced",
  "risk_level": "HIGH | MEDIUM | LOW",
  "can_auto_resolve": false,
  "escalation_note": "who specifically needs to review this and why"
}}

Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""

        parsed = _fallback_response()
        if client is not None:
            parsed = _call_claude_with_retry(client, prompt, pid)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("Claude response received for %s in %sms", pid, elapsed_ms)

        rec["llm_explanation"] = str(parsed.get("explanation", ""))
        rec["recommended_action"] = str(parsed.get("recommended_action", ""))
        rec["risk_level"] = str(parsed.get("risk_level", "MEDIUM"))
        rec["escalation_note"] = str(parsed.get("escalation_note", ""))

    return {"scored_records": scored}
