# StaffingAgent.ai -- VMS vs ATS Reconciliation Agent

A LangGraph-based reconciliation pipeline that compares hospital VMS timesheet records against Bullhorn ATS placement records to catch financial mismatches, compliance violations, and missing data **before** payroll and invoicing.


---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run the pipeline
python src/agent.py

# 4. Run tests
python -m pytest tests/ -v
```

Output is written to `output/reconciliation_results.json` (20 records).
Open `dashboard/hitl_dashboard.html` in a browser for the HITL review interface.

### Custom data paths

```bash
python src/agent.py --vms path/to/vms.csv --ats path/to/ats.csv
```

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes (for LLM) | Claude API key for HITL reasoning |
| `LLM_MODEL` | No | Override model (default: `claude-sonnet-4-6`). Use `claude-haiku-4-5` for lower cost |
| `LANGCHAIN_API_KEY` | No | Enables LangSmith tracing when set |

If `ANTHROPIC_API_KEY` is not set, the pipeline still runs -- flagged records get a fallback response instead of Claude-generated explanations.

---

## Architecture

### Pipeline overview

```
                          +---------------------+
                          |  vms_records.csv     |
                          |  ats_bullhorn_records |
                          +----------+----------+
                                     |
                                     v
                    +----------------+----------------+
                    |      1. ingest_normalize        |
                    |  Load CSVs, normalize columns,  |
                    |  resolve client aliases,         |
                    |  fill POs, coerce numerics       |  RULES
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |          2. match                |
                    |  Outer join on placement_id,     |
                    |  fuzzy name matching (rapidfuzz), |
                    |  build discrepancy list per row  |  RULES
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |          3. score                |
                    |  Penalty-based confidence 0-1,   |
                    |  hard business rules (PO, status,|
                    |  financial risk), flag for HITL   |  RULES
                    +----------------+----------------+
                                     |
                          +----------+----------+
                          |                     |
                    flagged_ids            no flagged
                    non-empty?              records
                          |                     |
                          v                     |
                    +-----+------+              |
                    | 4. llm_reason|             |
                    | Claude explains            |
                    | business impact,           |
                    | recommends actions |  LLM  |
                    +-----+------+              |
                          |                     |
                          +----------+----------+
                                     |
                                     v
                    +----------------+----------------+
                    |        5. compliance             |
                    |  Excessive hours (>=50h),        |
                    |  CA daily OT, FLSA OT rate,      |
                    |  missing timesheet, pending      |
                    |  approval. Escalates to HITL.    |  RULES
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |          6. output               |
                    |  Build final JSON, update        |
                    |  dashboard, print summary        |  RULES
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |  reconciliation_results.json     |
                    |  hitl_dashboard.html             |
                    +----------------------------------+
```

### Node details

| # | Node | File | What it does | LLM? |
|---|------|------|-------------|------|
| 1 | **ingest_normalize** | `normalizer.py` | Loads CSVs, normalizes column names (case-insensitive), resolves client aliases ("GHR - General Healthcare" -> "General Healthcare Resources"), fills blank PO numbers with "MISSING", coerces numerics, strips middle initials for name comparison, drops duplicate placement_ids and blank IDs with warnings | No |
| 2 | **match** | `matcher.py` | Outer join on `placement_id` (catches records missing from either system), rapidfuzz name similarity on stripped compare fields, detects 11 mismatch types (hours, OT hours, bill rate, OT rate, per diem, total, PO missing, PO mismatch, name, status, client), generates human-readable discrepancy strings | No |
| 3 | **score** | `scorer.py` | Penalty-based confidence (starts at 1.0, deductions per issue). Hard business rules override confidence: missing/mismatched PO, status issue, or financial risk >= $500 force HITL regardless. Sets `match_status`, `severity`, `requires_human_review` | No |
| 4 | **llm_reason** | `llm_reasoner.py` | **Conditional** -- only runs when flagged records exist. Calls Claude for each flagged placement with full context (both snapshots, discrepancy list, confidence). Claude returns explanation, recommended action, risk level, and escalation note. Retry with exponential backoff on rate limits. Falls back to a generic response if API is unavailable | **Yes** |
| 5 | **compliance** | `compliance.py` | 5 compliance checks: excessive hours (>=50h/week), CA daily OT (Labor Code daily overtime), FLSA OT rate (must be >= 1.5x regular), missing timesheet, pending approval. Any flag upgrades the record to `requires_human_review = true` with fallback action text | No |
| 6 | **output** | `output_writer.py` | Assembles final JSON with `run_metadata` + `records` array, re-embeds data into the HITL dashboard HTML, prints terminal summary | No |

### State management

All nodes share a `ReconciliationState` TypedDict (see `state.py`). Key design decision: **no DataFrames in state** -- normalizer converts to `list[dict]` so state is JSON-serializable, which keeps LangGraph checkpointing and LangSmith tracing working correctly.

---

## Where LLM adds value vs. where rules are sufficient

This is the core design decision of the system.

### Rules (5 of 6 nodes)

Confidence scoring, mismatch detection, and compliance checks are **fully deterministic**:

- **Reproducible**: same input always produces the same score and flags
- **Auditable**: every penalty and threshold is traceable in the code
- **Cheap**: no API calls, runs in milliseconds per record
- **Scalable**: 5000+ records cost nothing beyond compute

Rules handle: "Is there a mismatch?" and "How bad is it?"

### LLM (1 of 6 nodes, conditional)

Claude runs **only for flagged records** and **only for narrative**:

- **Explains** why a discrepancy matters in business terms (not just "hours differ by 4" but "the hospital reported 4 fewer hours, which means $368 less revenue if we invoice as-is")
- **Recommends** a specific action ("Contact the Fieldglass admin at Cross Country to verify the timesheet correction before resubmitting")
- **Prioritizes** with risk levels so the middle-office team tackles critical issues first

The LLM does NOT: calculate confidence scores, detect mismatches, check compliance rules, or make invoicing decisions. Those are all deterministic.

### Why this split matters

Finance and compliance teams need **traceable, stable logic** for audit trails. Natural language is reserved for **operator guidance on exceptions**, which keeps cost and variance contained while supporting HITL workflows. At 2000 placements/week with a 15% flag rate, only ~300 records hit the LLM -- the other 1700 are processed instantly by rules.

---

## Confidence scoring

Starts at **1.0** and subtracts penalties:

| Issue | Penalty |
|-------|---------|
| Hours mismatch (regular) | -0.25 |
| Bill rate mismatch | -0.20 |
| Total diff >= $500 | -0.20 |
| PO number missing | -0.15 |
| Status not Approved | -0.15 |
| OT hours mismatch | -0.10 |
| OT rate mismatch | -0.10 |
| PO number mismatch (both present, different) | -0.10 |
| Name similarity < 85 | -0.10 |
| Per diem mismatch | -0.05 |
| Client name mismatch | -0.05 |

Special cases: orphaned records (one-sided) get **0.0**, missing timesheets get **0.35**.

**Hard business rules** (force HITL regardless of confidence):
- Missing or mismatched PO number -- cannot invoice without a valid PO
- Status issue (Pending Approval, Disputed) -- cannot invoice unapproved records
- Financial variance >= $500 -- requires manual reconciliation

Threshold: `confidence <= 0.85` triggers human review.

---

## Compliance checks (beyond field matching)

| Check | Rule | Why it matters |
|-------|------|---------------|
| **Excessive hours** | Total hours >= 50/week | Potential labor law violation; requires supervisor approval |
| **CA daily OT** | Worker in California | CA Labor Code requires OT after 8h/day, not just 40h/week. VMS reports weekly only -- can't confirm daily OT automatically |
| **FLSA OT rate** | OT rate < 1.5x regular rate | Federal law requires OT at minimum 1.5x. Wrong rate = compliance violation |
| **Missing timesheet** | VMS status = "Missing Timesheet" | Can't process payroll or invoice without an approved timesheet |
| **Pending approval** | VMS status = "Pending Approval" | Hospital hasn't approved yet -- can't invoice |

Any compliance flag **automatically escalates** the record to `requires_human_review = true`, even if the confidence score is 1.0 and all data matches.

---

## Output schema

`output/reconciliation_results.json`:

```json
{
  "run_metadata": {
    "run_id": "uuid",
    "generated_at": "2026-03-27T...",
    "summary_stats": {
      "record_count": 20,
      "clean_matches_count": 6,
      "critical_count": 4,
      "warning_count": 10,
      "requires_human_review_count": 14,
      "total_financial_impact_at_risk": 8314.50
    },
    "models_used": { "llm_reasoning": "claude-sonnet-4-6" }
  },
  "records": [
    {
      "placement_id": "PL-1001",
      "candidate_name": "Maria Santos",
      "match_status": "match | mismatch | missing",
      "confidence_score": 1.0,
      "requires_human_review": false,
      "severity": "clean | warning | critical | info",
      "discrepancies": [],
      "recommended_action": "No action required...",
      "compliance_flags": [],
      "llm_explanation": "",
      "risk_level": "LOW | MEDIUM | HIGH",
      "financial_impact": 0.0,
      "vms_snapshot": { "hours": 40.0, "bill_rate": 85.0, "..." : "..." },
      "ats_snapshot": { "hours": 40.0, "bill_rate": 85.0, "..." : "..." }
    }
  ]
}
```

---

## HITL dashboard

Open `dashboard/hitl_dashboard.html` in any browser. No server required -- it's a single self-contained HTML file.

Features:
- Left panel: all 20 records sorted by severity, with one-line summaries
- Right panel: full detail view with plain-English explanations
- Translates technical terms: "VMS" becomes "Hospital System (Fieldglass)", "ATS" becomes "Our Records (Bullhorn)"
- Compliance notices section with actionable guidance
- Decision tracking: mark records as approved/escalated/corrected
- Auto-updated with fresh data after each pipeline run

---

## Data resilience

The normalizer handles common real-world data issues:

| Issue | Handling |
|-------|----------|
| Column name case (`VMS_Hours` vs `vms_hours`) | Auto-normalized to lowercase |
| Blank placement_id | Dropped with logged warning |
| Duplicate placement_id (correction rows) | Keeps last row per ID, logs duplicates |
| Non-numeric values ("N/A" in hours column) | Coerced to 0.0 via `pd.to_numeric(errors="coerce")` |
| Missing PO numbers (NaN) | Filled with "MISSING" sentinel |
| Client name aliases | Resolved via alias map |
| Name variations (middle initials, apostrophes) | Stripped for comparison, preserved for display |
| Unicode characters in names | Preserved in UTF-8 JSON output |

---

## Test suite

11 tests covering the critical paths:

```
test_csv_files_exist              -- data files present
test_normalization                -- client aliases resolved
test_normalization_custom_paths   -- CLI path override works
test_scoring_missing_timesheet    -- PL-1008 scores < 0.50
test_clean_records                -- PL-1001, PL-1015 score >= 0.85
test_excessive_hours_boundary     -- PL-1011 (exactly 50h) triggers flag
test_po_cross_comparison          -- PL-1010 missing PO detected
test_outer_join_no_silent_drops   -- all 20 IDs present after join
test_compliance_escalates_to_hitl -- CA records upgraded to HITL
test_normalizer_rejects_missing_columns -- bad CSV raises ValueError
test_output_schema                -- full pipeline, all fields present
```

### What would be added in production

- **Golden-file tests**: snapshot `reconciliation_results.json` against fixed fixtures to catch regressions
- **Property tests**: scoring monotonicity (more penalties = lower score), penalty bounds (never below 0.0 or above 1.0)
- **Contract tests**: Anthropic JSON response shape validation, fallback behavior when API returns errors
- **Load tests**: batch CSV sizes (5000+ rows), parallel LLM call throughput with rate limits
- **LangSmith trace assertions**: verify each node ran, correct routing taken, state mutations valid

---

## Project structure

```
staffingagent-reconciliation/
  src/
    agent.py           # LangGraph entrypoint + CLI
    state.py           # ReconciliationState TypedDict
    normalizer.py      # CSV load, schema validation, data cleaning
    matcher.py         # Outer join, fuzzy matching, discrepancy detection
    scorer.py          # Confidence scoring + hard business rules
    llm_reasoner.py    # Claude reasoning for flagged records
    compliance.py      # Labor law + invoicing compliance checks
    output_writer.py   # JSON output + dashboard update
  tests/
    test_basic.py      # 11 tests
  data/
    vms_records.csv    # 20 hospital VMS timesheet records
    ats_bullhorn_records.csv  # 20 Bullhorn ATS placement records
  dashboard/
    hitl_dashboard.html  # Self-contained HITL review interface
  output/
    reconciliation_results.json  # Generated output
  .env.example
  .gitignore
  requirements.txt
  README.md
```
