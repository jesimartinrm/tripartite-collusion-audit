"""Experiment orchestration for Task 7 evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import yaml  # type: ignore

from auditor.config import AuditPipelineConfig
from auditor.pipeline import AuditPipeline

# Define experiment grid entries. Each entry specifies audit config, scenario label, and metadata.
EXPERIMENTS: Sequence[Mapping[str, str]] = [
    {
        "label": "pricing_keyword",
        "config": "configs/audits/pricing_keyword_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "pricing_adaptive",
        "config": "configs/audits/pricing_adaptive_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "pricing_temperature",
        "config": "configs/audits/pricing_temperature_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "pricing_sparse",
        "config": "configs/audits/pricing_sparse_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "pricing_llm_open",
        "config": "configs/audits/pricing_llm_open_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "pricing_llm_gemini",
        "config": "configs/audits/pricing_llm_gemini_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "pricing_llm_groq",
        "config": "configs/audits/pricing_llm_groq_audit.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "auction_sparse",
        "config": "configs/audits/auction_sparse_audit.yaml",
        "scenario": "first_price_auction",
        "train_task": "pricing",
        "test_task": "auction",
    },
    {
        "label": "review_sparse",
        "config": "configs/audits/review_sparse_audit.yaml",
        "scenario": "peer_review",
        "train_task": "pricing",
        "test_task": "review",
    },
    {
        "label": "pricing_honest",
        "config": "configs/audits/pricing_honest_calibration.yaml",
        "scenario": "pricing_duopoly",
        "train_task": "pricing",
        "test_task": "pricing",
    },
    {
        "label": "auction_honest",
        "config": "configs/audits/auction_honest_calibration.yaml",
        "scenario": "first_price_auction",
        "train_task": "auction",
        "test_task": "auction",
    },
    {
        "label": "auction_keyword",
        "config": "configs/audits/auction_keyword_audit.yaml",
        "scenario": "first_price_auction",
        "train_task": "pricing",
        "test_task": "auction",
    },
    {
        "label": "auction_adaptive",
        "config": "configs/audits/auction_adaptive_audit.yaml",
        "scenario": "first_price_auction",
        "train_task": "pricing",
        "test_task": "auction",
    },
    {
        "label": "double_auction_honest",
        "config": "configs/audits/double_auction_honest_calibration.yaml",
        "scenario": "continuous_double_auction",
        "train_task": "market",
        "test_task": "market",
    },
    {
        "label": "double_auction_keyword",
        "config": "configs/audits/double_auction_keyword_audit.yaml",
        "scenario": "continuous_double_auction",
        "train_task": "market",
        "test_task": "market",
    },
    {
        "label": "review_honest",
        "config": "configs/audits/review_honest_calibration.yaml",
        "scenario": "peer_review",
        "train_task": "review",
        "test_task": "review",
    },
    {
        "label": "review_keyword",
        "config": "configs/audits/review_keyword_audit.yaml",
        "scenario": "peer_review",
        "train_task": "pricing",
        "test_task": "review",
    },
    {
        "label": "review_adaptive",
        "config": "configs/audits/review_adaptive_audit.yaml",
        "scenario": "peer_review",
        "train_task": "pricing",
        "test_task": "review",
    },
    {
        "label": "review_bias",
        "config": "configs/audits/review_bias_audit.yaml",
        "scenario": "peer_review",
        "train_task": "pricing",
        "test_task": "review",
    },
]

OUTPUT_DIR = Path("results/experiments")

# Backends that reach a paid/remote inference API (routed through the repo-root shared
# llm_config -> OpenRouter/Nebius). A label built on any of these needs an inference key;
# if the key is missing the label genuinely cannot run and must fail loud rather than be
# silently skipped as if it had succeeded.
_HOSTED_BACKENDS = {"openrouter", "openai", "openai_chat", "gemini", "google", "google_ai", "groq"}

# Marker substring of the RuntimeError raised by poseidon_compute when no inference key is
# configured. Used to tell a missing-key failure apart from other backend failures
# (e.g. a transformers config whose torch install is missing).
_MISSING_KEY_MARKER = "No inference API key"


def _uses_hosted_backend(config: AuditPipelineConfig) -> bool:
    """True if the audit's scenario config wires any agent to a hosted inference backend."""
    scenario_path = Path(config.scenario_config)
    if not scenario_path.exists():
        return False
    try:
        data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive; malformed scenario yaml
        return False

    def _walk(node: object) -> bool:
        if isinstance(node, Mapping):
            backend = node.get("backend")
            if isinstance(backend, str) and backend.lower() in _HOSTED_BACKENDS:
                return True
            return any(_walk(value) for value in node.values())
        if isinstance(node, (list, tuple)):
            return any(_walk(item) for item in node)
        return False

    return _walk(data)


def path_for(label: str) -> Path:
    return OUTPUT_DIR / f"{label}.json"


def run_experiments(selected: Sequence[str] | None = None) -> None:
    for exp in EXPERIMENTS:
        label = exp["label"]
        if selected and label not in selected:
            continue
        config_path = Path(exp["config"])
        if not config_path.exists():
            print(f"Skipping {label} (missing config {config_path})")
            continue
        config = AuditPipelineConfig.from_yaml(config_path)
        pipeline = AuditPipeline(config)
        try:
            summary = pipeline.run()
        except (RuntimeError, ImportError) as exc:
            # A hosted label that fails purely for lack of an inference key cannot run at
            # all -- skipping it silently reads as success but produces nothing. Fail loud
            # so the exit code reflects reality (set OPENROUTER_API_KEY to run it).
            if _MISSING_KEY_MARKER in str(exc) and _uses_hosted_backend(config):
                raise RuntimeError(
                    f"Cannot run hosted experiment '{label}': {exc} "
                    f"Set OPENROUTER_API_KEY (or run only the key-free labels, "
                    f"e.g. pricing_llm_open / scripted audits)."
                ) from exc
            # Other backend failures (e.g. a transformers config with torch not installed)
            # must not take the whole grid down -- the scripted labels are still worth writing.
            print(f"Skipping {label}: {exc}")
            continue
        detector_stats = {
            result.name: {"statistic": result.statistic, "p_value": result.p_value, "details": result.details}
            for result in summary.detectors
        }
        run_records = []
        for run in summary.runs:
            run_records.append(
                {
                    "run_id": run.run_id,
                    "seed": run.seed,
                    "permutation": run.permutation,
                    "metrics": run.metrics,
                }
            )
        payload = {
            "label": label,
            "config": str(config_path),
            "scenario": exp["scenario"],
            "train_task": exp.get("train_task"),
            "test_task": exp.get("test_task"),
            "detectors": detector_stats,
            "runs": run_records,
        }
        path = path_for(label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote experiment results to {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 7 experiment grid")
    parser.add_argument("labels", nargs="*", help="Subset of experiment labels to run")
    args = parser.parse_args()

    run_experiments(args.labels or None)


if __name__ == "__main__":  # pragma: no cover
    main()
