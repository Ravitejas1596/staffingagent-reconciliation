"""CSV load and normalization node."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

from src.state import ReconciliationState

logger = logging.getLogger(__name__)

CLIENT_ALIASES = {
    "GHR - General Healthcare": "General Healthcare Resources",
    "TRIO Workforce": "TRIO Workforce Solutions",
}

VMS_NUMERIC_COLS = [
    "vms_hours",
    "vms_bill_rate",
    "vms_ot_hours",
    "vms_ot_rate",
    "vms_per_diem",
    "vms_total",
]
ATS_NUMERIC_COLS = [
    "ats_hours",
    "ats_bill_rate",
    "ats_ot_hours",
    "ats_ot_rate",
    "ats_per_diem",
    "ats_total",
    "ats_pay_rate",
]


def _strip_middle_initials(name: str) -> str:
    """Strip middle initials for comparison; display name unchanged elsewhere."""
    s = name.strip()
    s = re.sub(r"\s+[A-Z]\.\s+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _validate_schema(df: "pd.DataFrame", label: str, required: List[str]) -> None:
    """Raise ValueError with a clear message if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"{label} CSV has no rows.")
    logger.info("normalize_node | %s schema OK (%d rows, %d cols)", label, len(df), len(df.columns))


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def normalize_node(state: ReconciliationState) -> dict:
    """Load CSVs, normalize client names, POs, numerics; log actions.

    Paths are read from state['vms_path'] / state['ats_path'] when provided
    (set via CLI --vms / --ats flags), otherwise fall back to ./data/ defaults.

    Returns list[dict] for vms_normalized and ats_normalized so that LangGraph
    state remains JSON-serializable (no pd.DataFrame in state).
    """
    data_dir = _default_data_dir()
    vms_path = Path(state.get("vms_path") or data_dir / "vms_records.csv")
    ats_path = Path(state.get("ats_path") or data_dir / "ats_bullhorn_records.csv")
    log_lines: List[str] = []

    logger.info("normalize_node | loading VMS from %s", vms_path)
    logger.info("normalize_node | loading ATS from %s", ats_path)
    vms = pd.read_csv(vms_path)
    ats = pd.read_csv(ats_path)

    # Normalize column names so "VMS_Hours" / "vms_hours" / "Vms_Hours" all work
    vms.columns = vms.columns.str.lower().str.strip()
    ats.columns = ats.columns.str.lower().str.strip()

    _validate_schema(vms, "VMS", ["placement_id", "candidate_name"] + VMS_NUMERIC_COLS)
    _validate_schema(ats, "ATS", ["placement_id", "candidate_name"] + ATS_NUMERIC_COLS)

    # Reject rows with blank placement_id — they cannot be matched and would produce
    # orphan records with placement_id "nan" downstream.
    for df, label in ((vms, "VMS"), (ats, "ATS")):
        blank_ids = df["placement_id"].isna()
        if blank_ids.any():
            n_blank = int(blank_ids.sum())
            log_lines.append(f"Dropped {n_blank} {label} row(s) with missing placement_id")
            logger.warning("normalize_node | dropped %d %s rows with blank placement_id", n_blank, label)
    vms = vms.dropna(subset=["placement_id"])
    ats = ats.dropna(subset=["placement_id"])

    # Guard against duplicate placement_ids (e.g. correction rows). Keep last row
    # per placement_id and log the duplicates so they are visible in the audit trail.
    for df, label in ((vms, "VMS"), (ats, "ATS")):
        dupes = df[df["placement_id"].duplicated(keep=False)]
        if not dupes.empty:
            dupe_ids = dupes["placement_id"].unique().tolist()
            log_lines.append(
                f"{label}: duplicate placement_id(s) {dupe_ids} — keeping last row per ID"
            )
            logger.warning("normalize_node | %s duplicate placement_ids: %s", label, dupe_ids)
    vms = vms.drop_duplicates(subset=["placement_id"], keep="last")
    ats = ats.drop_duplicates(subset=["placement_id"], keep="last")

    if "candidate_name" in vms.columns:
        vms["candidate_name"] = vms["candidate_name"].map(
            lambda x: str(x).strip() if pd.notna(x) else x
        )
        logger.info("Normalization: stripped whitespace on VMS candidate_name")
    if "candidate_name" in ats.columns:
        ats["candidate_name"] = ats["candidate_name"].map(
            lambda x: str(x).strip() if pd.notna(x) else x
        )
        logger.info("Normalization: stripped whitespace on ATS candidate_name")

    if "vms_po_number" in vms.columns:
        before = vms["vms_po_number"].isna().sum()
        vms["vms_po_number"] = vms["vms_po_number"].fillna("MISSING")
        if before:
            log_lines.append(f"Filled {int(before)} VMS PO number(s) with MISSING")
            logger.info("Normalization: filled %s NaN vms_po_number with MISSING", before)
    if "ats_po_number" in ats.columns:
        before = ats["ats_po_number"].isna().sum()
        ats["ats_po_number"] = ats["ats_po_number"].fillna("MISSING")
        if before:
            log_lines.append(f"Filled {int(before)} ATS PO number(s) with MISSING")
            logger.info("Normalization: filled %s NaN ats_po_number with MISSING", before)

    for df, label in ((vms, "VMS"), (ats, "ATS")):
        if "client_name" in df.columns:
            for old, new in CLIENT_ALIASES.items():
                mask = df["client_name"] == old
                n = int(mask.sum())
                if n:
                    df.loc[mask, "client_name"] = new
                    msg = f"{label}: resolved client alias '{old}' -> '{new}' ({n} row(s))"
                    log_lines.append(msg)
                    logger.info("Normalization: %s", msg)

    # Build compare columns (middle initials stripped) — used in match_node fuzzy scoring
    for df, prefix in ((vms, "vms"), (ats, "ats")):
        ccol = "candidate_name"
        if ccol in df.columns:
            df[f"{prefix}_candidate_compare"] = df[ccol].map(
                lambda x: _strip_middle_initials(str(x)) if pd.notna(x) else ""
            )
            logger.info(
                "Normalization: added %s_candidate_compare (middle initials stripped for fuzzy match)",
                prefix,
            )

    for col in VMS_NUMERIC_COLS:
        if col in vms.columns:
            vms[col] = pd.to_numeric(vms[col], errors="coerce").fillna(0.0).astype(float)
    for col in ATS_NUMERIC_COLS:
        if col in ats.columns:
            ats[col] = pd.to_numeric(ats[col], errors="coerce").fillna(0.0).astype(float)

    log_lines.append("Converted numeric columns to float; NaN filled with 0.0")
    logger.info("Normalization: numeric columns coerced to float, NaN -> 0.0")

    # Convert to list[dict] — keeps LangGraph state JSON-serializable
    return {
        "vms_normalized": vms.to_dict("records"),
        "ats_normalized": ats.to_dict("records"),
        "processing_log": log_lines,
    }
