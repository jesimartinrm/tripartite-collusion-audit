"""Split the appended runs inside the shipped transcripts into one file per run.

The original `JsonlLogger` appended to a transcript whose name depends only on the
run index, so every re-run of an audit, calibration or sweep was written onto the
previous one: 1,155 of the 1,737 transcripts under `results/` hold between 2 and 48
runs back to back, with no separator. A new run is recognisable because `round_id`
drops back to 1.

This script never modifies `results/`. For every `<name>.jsonl` it writes
`<name>__run1.jsonl`, `<name>__run2.jsonl`, ... under an output root that mirrors the
original layout. The companion `<name>_interventions.jsonl` was written with truncation,
so it only ever describes the *last* run; it is copied next to that run as
`<name>__runN_interventions.jsonl`.

    python scripts/split_transcript_runs.py --dry-run          # counts only
    python scripts/split_transcript_runs.py                    # results/ -> results_split/
    python scripts/split_transcript_runs.py --dedupe           # skip runs identical to an earlier one
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List


def split_runs(lines: List[str]) -> List[List[str]]:
    """Group transcript lines into runs; a run starts where round_id returns to 1."""
    runs: List[List[str]] = []
    prev_round = None
    for line in lines:
        try:
            round_id = json.loads(line).get("round_id")
        except json.JSONDecodeError:
            round_id = None
        if round_id == 1 and prev_round is not None and prev_round > 1:
            runs.append([])
        if not runs:
            runs.append([])
        runs[-1].append(line)
        prev_round = round_id if isinstance(round_id, int) else prev_round
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("results"), help="Where the transcripts are")
    parser.add_argument("--out", type=Path, default=Path("results_split"), help="Where to write the split runs")
    parser.add_argument("--dedupe", action="store_true", help="Skip a run that is byte-identical to an earlier run of the same file")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    args = parser.parse_args()

    transcripts = sorted(p for p in args.root.rglob("*.jsonl") if not p.name.endswith("_interventions.jsonl"))
    n_files = n_multi = n_runs = n_written = n_skipped = 0
    max_runs = 0
    for path in transcripts:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        runs = split_runs(lines)
        n_files += 1
        n_runs += len(runs)
        max_runs = max(max_runs, len(runs))
        if len(runs) > 1:
            n_multi += 1
        if args.dry_run:
            continue
        rel = path.relative_to(args.root)
        seen: set[str] = set()
        for i, run in enumerate(runs, start=1):
            blob = "\n".join(run) + "\n"
            if args.dedupe and blob in seen:
                n_skipped += 1
                continue
            seen.add(blob)
            dst = args.out / rel.with_name(f"{rel.stem}__run{i}.jsonl")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(blob, encoding="utf-8")
            n_written += 1
        companion = path.with_name(f"{path.stem}_interventions.jsonl")
        if companion.exists():
            shutil.copy2(companion, args.out / rel.with_name(f"{rel.stem}__run{len(runs)}_interventions.jsonl"))

    print(f"transcripts: {n_files} | with more than one run: {n_multi} | runs in total: {n_runs} | most runs in one file: {max_runs}")
    if not args.dry_run:
        print(f"written: {n_written} run files under {args.out}/" + (f" | skipped as identical: {n_skipped}" if args.dedupe else ""))
        print("interventions logs describe only the last run of each file and were copied next to it.")


if __name__ == "__main__":
    main()
