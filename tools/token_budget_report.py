# File: tools/token_budget_report.py
"""Report completion-token distributions per model from slm_results/*.json.

Answers one question: what max_new_tokens does each model actually need?

Records that stopped on "length" are right-censored -- all we learn from them is
that the model wanted at least the cap, never how much more. Percentiles are
therefore computed over EOS-terminated records only, and a recommended cap is
emitted only when the truncation rate is zero. When it isn't, the honest answer
is "raise the cap and rerun the truncated ids", not a percentile of the subset
that happened to finish.

The null-answer split matters for thinking models specifically: parse_model_response
returns model_answer=None whenever <think> has no closing </think>, so for those
models every truncated record is a guaranteed unparseable -- a token-budget bug
wearing the costume of a reasoning failure.
"""
import argparse
import glob
import json
import math
import os
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_REPO_ROOT, "slm_results")


def _percentile(sorted_values, fraction):
    """Nearest-rank percentile; avoids a numpy dependency for a handful of ints."""
    if not sorted_values:
        return None
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def _latest_per_id(records):
    """Keep the last record for each id.

    Reruns append rather than replace, so a question rerun at a higher cap
    leaves its truncated original in the file. Counting both would keep the
    truncation rate permanently above zero and block a recommendation forever.
    evaluate_results.py already scores last-record-wins; match it.
    """
    by_id = {}
    for record in records:
        by_id[record.get("id")] = record
    return list(by_id.values())


def summarize(records):
    records = _latest_per_id(records)
    reasons = Counter()
    null_by_reason = Counter()
    eos_lengths = []
    caps = set()
    seconds = 0.0
    tokens_timed = 0
    untelemetered = 0

    for record in records:
        reason = record.get("stop_reason")
        if reason is None or record.get("completion_tokens") is None:
            untelemetered += 1
            continue
        reasons[reason] += 1
        if record.get("max_new_tokens") is not None:
            caps.add(int(record["max_new_tokens"]))
        if record.get("gen_seconds"):
            seconds += record["gen_seconds"]
            tokens_timed += int(record["completion_tokens"])
        if record.get("model_answer") is None:
            null_by_reason[reason] += 1
        if reason == "eos":
            eos_lengths.append(int(record["completion_tokens"]))

    eos_lengths.sort()
    scored = sum(reasons.values())
    return {
        "n": len(records),
        "untelemetered": untelemetered,
        "scored": scored,
        "reasons": reasons,
        "null_by_reason": null_by_reason,
        "caps": sorted(caps),
        "trunc_rate": (reasons["length"] / scored) if scored else None,
        "p50": _percentile(eos_lengths, 0.50),
        "p90": _percentile(eos_lengths, 0.90),
        "p95": _percentile(eos_lengths, 0.95),
        "p99": _percentile(eos_lengths, 0.99),
        "max": eos_lengths[-1] if eos_lengths else None,
        "tok_per_s": round(tokens_timed / seconds, 1) if seconds else None,
    }


def recommend(summary, margin=1.25, granularity=64):
    """Return (cap, explanation). cap is None when the data can't support one."""
    if summary["max"] is None:
        return None, "no EOS-terminated records to measure"
    if summary["trunc_rate"]:
        largest = max(summary["caps"]) if summary["caps"] else None
        return None, (
            f"CENSORED: {summary['trunc_rate']:.0%} truncated at cap {largest}. "
            f"Rerun the truncated ids at >= {2 * largest if largest else '?'} "
            "before trusting any recommendation."
        )
    cap = int(math.ceil(summary["max"] * margin / granularity) * granularity)
    return cap, f"max EOS length {summary['max']} x {margin} margin, rounded up to {granularity}"


def truncated_ids(records):
    return [
        record.get("id")
        for record in _latest_per_id(records)
        if record.get("stop_reason") == "length" and record.get("id")
    ]


def _format(signature, summary, cap, why, records):
    reasons = summary["reasons"]
    lines = [
        f"{signature}",
        "  n={n}  eos={eos}  length={length}  stop={stop}  error={error}".format(
            n=summary["n"],
            eos=reasons["eos"],
            length=reasons["length"],
            stop=reasons["stop"],
            error=reasons["error"],
        ),
    ]
    if summary["untelemetered"]:
        lines.append(
            f"  {summary['untelemetered']} record(s) predate telemetry and were skipped"
        )
    if summary["trunc_rate"] is not None:
        caps = ", ".join(str(c) for c in summary["caps"]) or "unknown"
        lines.append(f"  truncation rate    {summary['trunc_rate']:.1%}   (cap {caps})")
    nulls = summary["null_by_reason"]
    if sum(nulls.values()):
        detail = "  ".join(
            f"{count}/{reasons[reason]} {reason}" for reason, count in sorted(nulls.items())
        )
        lines.append(f"  null model_answer  {detail}")
    if summary["max"] is not None:
        lines.append(
            "  completion_tokens (EOS only)  "
            f"p50={summary['p50']} p90={summary['p90']} p95={summary['p95']} "
            f"p99={summary['p99']} max={summary['max']}"
        )
    if summary["tok_per_s"]:
        lines.append(f"  throughput         {summary['tok_per_s']} tok/s")
    lines.append(f"  recommended        {cap if cap is not None else why}")
    if cap is not None:
        lines.append(f"                     ({why})")
    else:
        ids = truncated_ids(records)
        if ids:
            preview = ",".join(ids[:8]) + ("..." if len(ids) > 8 else "")
            lines.append(f"                     --ids {preview}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=_RESULTS_DIR)
    parser.add_argument("--margin", type=float, default=1.25,
                        help="Safety multiplier on the longest EOS-terminated response.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    report = {}
    blocks = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            continue
        signature = os.path.splitext(os.path.basename(path))[0]
        summary = summarize(records)
        cap, why = recommend(summary, args.margin)
        report[signature] = {
            **{key: value for key, value in summary.items()
               if key not in ("reasons", "null_by_reason")},
            "stop_reasons": dict(summary["reasons"]),
            "null_by_stop_reason": dict(summary["null_by_reason"]),
            "recommended_max_new_tokens": cap,
            "recommendation_basis": why,
            "truncated_ids": truncated_ids(records),
        }
        blocks.append(_format(signature, summary, cap, why, records))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif blocks:
        print("\n\n".join(blocks))
    else:
        print(f"No result files found in {args.results_dir}")


if __name__ == "__main__":
    main()
