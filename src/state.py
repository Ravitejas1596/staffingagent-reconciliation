"""LangGraph shared state for VMS vs ATS reconciliation."""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict


class ReconciliationState(TypedDict):
    """Shared state for the reconciliation StateGraph.

    DataFrames are NOT stored here — they are not JSON-serializable and will
    break LangSmith tracing and LangGraph checkpointing.  Normalized records
    are stored as plain list[dict] so the state can be safely serialized.
    """

    # Configurable input paths (set from CLI; None falls back to ./data/ defaults)
    vms_path: Optional[str]
    ats_path: Optional[str]

    # Normalized records as plain dicts (replaces pd.DataFrame fields)
    vms_normalized: Optional[List[dict]]
    ats_normalized: Optional[List[dict]]

    matched_records: List[dict]
    scored_records: List[dict]
    flagged_ids: List[str]
    compliance_results: List[dict]
    final_results: List[dict]
    errors: Annotated[List[str], operator.add]
    run_id: str
    processing_log: Annotated[List[str], operator.add]
