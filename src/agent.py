"""LangGraph reconciliation pipeline — entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.compliance import compliance_node
from src.llm_reasoner import llm_reason_node
from src.matcher import match_node
from src.normalizer import normalize_node
from src.output_writer import output_node
from src.scorer import score_node
from src.state import ReconciliationState

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def route_after_scoring(state: ReconciliationState) -> str:
    if state.get("flagged_ids"):
        return "llm_reason"
    return "compliance"


def build_graph():
    graph = StateGraph(ReconciliationState)

    graph.add_node("ingest_normalize", normalize_node)
    graph.add_node("match", match_node)
    graph.add_node("score", score_node)
    graph.add_node("llm_reason", llm_reason_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("ingest_normalize")
    graph.add_edge("ingest_normalize", "match")
    graph.add_edge("match", "score")

    graph.add_conditional_edges(
        "score",
        route_after_scoring,
        {
            "llm_reason": "llm_reason",
            "compliance": "compliance",
        },
    )

    graph.add_edge("llm_reason", "compliance")
    graph.add_edge("compliance", "output")
    graph.add_edge("output", END)

    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="StaffingAgent VMS vs ATS reconciliation pipeline")
    parser.add_argument("--vms", metavar="PATH", help="Path to VMS records CSV (default: data/vms_records.csv)")
    parser.add_argument("--ats", metavar="PATH", help="Path to ATS/Bullhorn records CSV (default: data/ats_bullhorn_records.csv)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    # Prefer project .env; fall back to parent workspace (e.g. monorepo root) if missing
    load_dotenv(root / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        load_dotenv(root.parent / ".env")

    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "staffingagent-reconciliation")
        logger.info("LangSmith tracing enabled (LANGCHAIN_API_KEY present)")

    initial_state = cast(
        ReconciliationState,
        {
            "vms_path": str(args.vms) if args.vms else None,
            "ats_path": str(args.ats) if args.ats else None,
            "vms_normalized": None,
            "ats_normalized": None,
            "matched_records": [],
            "scored_records": [],
            "flagged_ids": [],
            "compliance_results": [],
            "final_results": [],
            "errors": [],
            "run_id": str(uuid.uuid4()),
            "processing_log": [],
        },
    )

    app = build_graph()
    result = app.invoke(initial_state)
    n = len(result.get("final_results") or [])
    logger.info("Reconciliation complete | run_id=%s | records=%s", initial_state["run_id"], n)
    print(f"Done. Wrote output/reconciliation_results.json ({n} records).")


if __name__ == "__main__":
    main()
