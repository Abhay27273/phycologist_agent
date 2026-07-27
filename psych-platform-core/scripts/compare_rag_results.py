"""
compare_rag_results.py
======================
Computes before/after metric deltas for overall and per-topic RAG performance.

Usage:
    python scripts/compare_rag_results.py

Expects (relative to project root):
    logs/rag_evaluation_results_before_reindex.csv  (baseline snapshot)
    logs/rag_evaluation_results.csv                 (latest run)
    logs/rag_evaluation_topic_summary.csv           (latest per-topic, written by evaluate_rag.py)

Outputs:
    logs/rag_delta_report.csv       — per-row metric deltas
    logs/rag_delta_topic_report.csv — per-topic aggregate deltas
    Console: colour-coded summary table
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LOGS = Path(__file__).resolve().parent.parent / "logs"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

BEFORE_RAW  = LOGS / "rag_evaluation_results_before_reindex.csv"
AFTER_RAW   = LOGS / "rag_evaluation_results.csv"
AFTER_TOPIC = LOGS / "rag_evaluation_topic_summary.csv"

# Output files
DELTA_RAW   = LOGS / "rag_delta_report.csv"
DELTA_TOPIC = LOGS / "rag_delta_topic_report.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Required file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {len(df)} rows  ({path.name})")
    return df


def _coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def _overall_summary(df: pd.DataFrame, label: str) -> dict:
    row: dict = {"run": label}
    for m in METRICS:
        if m in df.columns:
            vals = df[m].dropna()
            row[f"{m}_avg"] = float(vals.mean()) if not vals.empty else None
        else:
            row[f"{m}_avg"] = None
    return row


def _question_key(df: pd.DataFrame) -> pd.Series:
    for col in ("user_input", "question"):
        if col in df.columns:
            return df[col].str.strip().str.lower().str[:80]
    raise KeyError("Neither 'user_input' nor 'question' column found in results CSV")


# ──────────────────────────────────────────────────────────────────────────────
# Overall delta
# ──────────────────────────────────────────────────────────────────────────────

def build_raw_delta(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    bk = _question_key(before).rename("_key")
    ak = _question_key(after).rename("_key")

    b = before.assign(_key=bk)
    a = after.assign(_key=ak)

    merged = b.merge(a, on="_key", suffixes=("_before", "_after"), how="outer")

    delta_rows = []
    for _, row in merged.iterrows():
        entry: dict = {"question": row.get("_key", "")}
        # preserve topic/mood from after side if present
        for meta in ("topic", "mood"):
            for sfx in ("_after", "_before", ""):
                k = meta + sfx
                if k in row.index and pd.notna(row.get(k)):
                    entry[meta] = row[k]
                    break
        for m in METRICS:
            b_val = pd.to_numeric(row.get(f"{m}_before"), errors="coerce")
            a_val = pd.to_numeric(row.get(f"{m}_after"), errors="coerce")
            entry[f"{m}_before"] = b_val
            entry[f"{m}_after"]  = a_val
            if pd.notna(b_val) and pd.notna(a_val):
                entry[f"{m}_delta"] = round(a_val - b_val, 4)
            else:
                entry[f"{m}_delta"] = None
        delta_rows.append(entry)

    return pd.DataFrame(delta_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-topic delta
# ──────────────────────────────────────────────────────────────────────────────

def build_topic_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw deltas by topic."""
    if "topic" not in delta_df.columns:
        print("[WARN] No 'topic' column in delta report; skipping per-topic breakdown.")
        return pd.DataFrame()

    rows = []
    for topic, grp in delta_df.groupby("topic", dropna=False):
        row: dict = {"topic": topic, "n": int(len(grp))}
        for m in METRICS:
            for label in ("before", "after", "delta"):
                col = f"{m}_{label}"
                if col in grp.columns:
                    vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                    row[col + "_avg"] = round(float(vals.mean()), 4) if not vals.empty else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values("topic").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Console printer
# ──────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _fmt(val, is_delta=False, width=8):
    if val is None or (isinstance(val, float) and val != val):
        return "  n/a  ".rjust(width)
    s = f"{val:+.4f}" if is_delta else f"{val:.4f}"
    if is_delta:
        color = GREEN if val > 0.005 else (RED if val < -0.005 else YELLOW)
        return (color + s.rjust(width) + RESET)
    return s.rjust(width)


def print_overall_comparison(before_row: dict, after_row: dict):
    print(f"\n{'='*70}")
    print(f"{BOLD}Overall RAG Metrics — Before vs After Reindex{RESET}")
    print(f"{'='*70}")
    header = f"{'Metric':<22} {'Before':>8} {'After':>8} {'Delta':>10}"
    print(header)
    print("-" * 52)
    for m in METRICS:
        b = before_row.get(f"{m}_avg")
        a = after_row.get(f"{m}_avg")
        delta = (a - b) if (b is not None and a is not None) else None
        print(f"  {m:<20} {_fmt(b)} {_fmt(a)} {_fmt(delta, is_delta=True)}")
    print()


def print_topic_comparison(topic_df: pd.DataFrame):
    if topic_df.empty:
        return

    print(f"\n{'='*70}")
    print(f"{BOLD}Per-Topic RAG Deltas{RESET}")
    print(f"{'='*70}")

    for _, row in topic_df.iterrows():
        topic = row.get("topic", "unknown")
        n = int(row.get("n", 0))
        print(f"\n  {BOLD}{topic.upper()}{RESET} (n={n})")
        for m in METRICS:
            b   = row.get(f"{m}_before_avg")
            a   = row.get(f"{m}_after_avg")
            d   = row.get(f"{m}_delta_avg")
            print(f"    {m:<22} {_fmt(b)} → {_fmt(a)}   delta {_fmt(d, is_delta=True)}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading evaluation results…")
    before = _coerce_metrics(_load(BEFORE_RAW, "BEFORE"))
    after  = _coerce_metrics(_load(AFTER_RAW,  "AFTER"))

    # Overall summaries
    before_summary = _overall_summary(before, "before")
    after_summary  = _overall_summary(after,  "after")
    print_overall_comparison(before_summary, after_summary)

    # Row-level delta
    delta_df = build_raw_delta(before, after)
    delta_df.to_csv(DELTA_RAW, index=False)
    print(f"  Raw delta report saved to: {DELTA_RAW.name}")

    # Per-topic delta
    # Try to get topic info from after CSV first (it has topic column from new evaluate_rag.py)
    if "topic" not in after.columns and AFTER_TOPIC.exists():
        # Fall back to per-topic summary CSV if after CSV lacks topic column
        topic_summary = pd.read_csv(AFTER_TOPIC)
        # Rebuild delta using topic column from delta_df if available
        topic_delta = build_topic_delta(delta_df)
    else:
        topic_delta = build_topic_delta(delta_df)

    if not topic_delta.empty:
        topic_delta.to_csv(DELTA_TOPIC, index=False)
        print_topic_comparison(topic_delta)
        print(f"  Per-topic delta report saved to: {DELTA_TOPIC.name}")
    else:
        # Use the after topic summary for standalone display if no before topic data
        if AFTER_TOPIC.exists():
            print(f"\n[INFO] No per-topic BEFORE baseline available.")
            print(f"       Displaying current (AFTER) per-topic summary only:\n")
            ts = pd.read_csv(AFTER_TOPIC)
            print(ts.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
