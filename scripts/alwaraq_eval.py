"""
Golden-set evaluation for /alwaraq/answer (implementionPlan.md §10).

Golden file: JSON Lines, one question per line:
    {"query": "...", "document_id": "123"}                       # book scope
    {"query": "...", "expected_books": ["123", "456"]}             # library scope
    {"query": "...", "document_id": "123", "expect_no_evidence": true}

Usage:
    venv/bin/python -m scripts.alwaraq_eval scripts/alwaraq_golden.sample.jsonl \
        --base-url http://localhost:6000
"""

import argparse
import json
import statistics
import time

import requests


def run(golden_path: str, base_url: str, timeout: int) -> None:
    with open(golden_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    rows = []
    for case in cases:
        params = {"query": case["query"]}
        if case.get("document_id"):
            params["document_id"] = case["document_id"]
        started = time.time()
        try:
            resp = requests.get(f"{base_url}/alwaraq/answer", params=params, timeout=timeout).json()
        except Exception as e:
            resp = {"error": str(e), "status_code": 0}
        latency = time.time() - started

        status = resp.get("status") if resp.get("status_code") == 200 else "error"
        cited_docs = {s.get("document_id") for s in (resp.get("sources") or {}).values()}
        expected = set(case.get("expected_books") or [])
        row = {
            "query": case["query"],
            "status": status,
            "latency": latency,
            "routing_hit": bool(expected & {b["bookId"] for b in resp.get("books_searched") or []}) if expected else None,
            "cited_expected": bool(expected & cited_docs) if expected else None,
            "no_evidence_ok": (status == "no_evidence") if case.get("expect_no_evidence") else None,
            "answered_ok": (status == "ok") if not case.get("expect_no_evidence") else None,
            "error": resp.get("error"),
        }
        rows.append(row)
        print(f"[{status:>11}] {latency:5.1f}s  {case['query'][:70]}" + (f"  ERROR: {row['error']}" if row["error"] else ""))

    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{100 * sum(vals) / len(vals):.0f}% ({sum(vals)}/{len(vals)})" if vals else "n/a"

    latencies = sorted(r["latency"] for r in rows)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0
    print("\n── Summary ──")
    print(f"Questions:                 {len(rows)}")
    print(f"Answered (expected ok):    {rate('answered_ok')}")
    print(f"No-evidence correctness:   {rate('no_evidence_ok')}")
    print(f"Book routing recall:       {rate('routing_hit')}")
    print(f"Cited an expected book:    {rate('cited_expected')}")
    print(f"Errors:                    {sum(1 for r in rows if r['status'] == 'error')}")
    if latencies:
        print(f"Latency median / p95:      {statistics.median(latencies):.1f}s / {p95:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--base-url", default="http://localhost:6000")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    run(args.golden, args.base_url, args.timeout)
