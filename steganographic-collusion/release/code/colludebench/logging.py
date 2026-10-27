"""Logging utilities for ColludeBench."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping


@dataclass
class TranscriptRecord:
    run_id: str
    round_id: int
    agent_id: str
    role: str
    message: str
    action: str
    reward: float
    interventions: Mapping[str, float]


class JsonlLogger:
    """JSONLines transcript logger: one run per file.

    The file is truncated when the logger is created (one logger per run) and appended
    to on each flush(). The original was append-only with no truncation, and run ids are
    stable across invocations, so every re-run of an experiment was appended to the
    previous transcript -- the shipped dataset carries up to 48 runs per file. See
    scripts/split_transcript_runs.py for pulling those apart.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._buffer: list[TranscriptRecord] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log_turn(self, *, run_id: str, round_id: int, agent_id: str, role: str, message: str, action: str, reward: float, interventions: Mapping[str, float]) -> None:
        self._buffer.append(
            TranscriptRecord(
                run_id=run_id,
                round_id=round_id,
                agent_id=agent_id,
                role=role,
                message=message,
                action=action,
                reward=reward,
                interventions=dict(interventions),
            )
        )

    def flush(self) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            for record in self._buffer:
                fh.write(json.dumps(asdict(record)) + "\n")
        self._buffer.clear()


def write_manifest(path: Path, *, scenario: str, runs: Iterable[Mapping[str, str]]) -> None:
    """Persist manifest for reproducibility."""
    payload = {"scenario": scenario, "runs": list(runs)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
