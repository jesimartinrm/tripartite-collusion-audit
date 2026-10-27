# Audit the Whisper

An auditor for detecting steganographic collusion in multi-agent LLM systems — coupling channel-capacity theory, benchmark design, calibrated detectors, and reproducible experiments.

**References:** audits the phenomenon from Mathew et al. (2024), *Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs* ([arXiv:2410.03768](https://arxiv.org/abs/2410.03768)); full bibliography in [`docs/literature_review.md`](docs/literature_review.md).

## Overview

Audit the Whisper studies **auditing steganographic collusion in multi-agent LLM systems**: covert
coordination where agents hide signals in otherwise-normal text. The auditor pairs formal
covert-channel capacity analysis with a calibrated detector stack and a reproducible benchmark.

- **Capacity-guided auditing:** Formal analysis with full proofs linking auditor interventions to
  covert-channel capacity, minimax lower bounds, and time-uniform e-value guarantees.
- **ColludeBench-v0:** Pricing, first-price auction, peer review, continuous double auction, and
  hosted-LLM scenarios (Gemini, Llama 3.1 8B, GPT-4o-mini, reached through the shared inference
  router — see **Running experiments**) with configurable colluders and reproducible manifests.
- **Calibrated detector stack:** Mutual information, permutation invariance, watermark variance,
  fairness-aware acceptance bias, and welfare guard tuned to false-positive rate $\le 10^{-3}$ via
  10k honest runs plus sequential martingales.
- **External validation:** Head-to-head evaluations against Secret Collusion, CASE, Perfect Collusion
  Benchmark, and SentinelAgent, all at fixed FPRs using pre-registered thresholds.
- **End-to-end reproducibility:** Deterministic seeds, dataset hashes, regeneration scripts, CI smoke
  tests, and anonymised artifact packaging for double-blind review.

In the Collusion Research Sprint this is the **detector / auditing** project: it supplies calibrated
collusion detectors and the ColludeBench benchmark that other sprint scenarios can be audited against.

### Repository layout

| Path | Description |
| --- | --- |
| `auditor/` | Detection pipeline, calibrators, sweep orchestration, and detector registry (includes external baseline wrappers). |
| `colludebench/` | Environment definitions, agent policies, and covert-channel implementations. |
| `configs/` | YAML configs for tasks, colluders, hosted LLM clients, interventions, and calibration sweeps. |
| `scripts/` | Automation utilities (result regeneration, release packaging, dataset manifests, CI checks). |
| `results/` | Calibration outputs, experiment metrics, dataset manifests, regenerated tables, and raw transcripts. |
| `docs/` | Threat model, theory foundations, reproducibility checklist, capacity trade-offs, experiment summaries, literature review, and external baseline notes. |
| `release/` | Packaged artifact bundle produced by `scripts/prepare_release.py` (see `release/README.md`). |
| `notebooks/` | Validation notes (`validation_smoke.md`). The analysis notebooks were not included in this release. |
| `tests/` | Smoke tests for key modules, detectors, and CLI entry points. |

## Setup

> **Fastest path:** from the repo root, `./setup.sh steganographic-collusion` sets this experiment up (deps + the shared `poseidon_compute` package) in one step; `./setup.sh` does all experiments. The manual steps below are equivalent.

This experiment ships inside the starter repo. From the repo root, enter the experiment directory and
create an environment:

```bash
cd steganographic-collusion
uv sync
```

`uv sync` is the default install (numerical stack, YAML, and the OpenAI-compatible client used for
hosted inference, plus the shared `poseidon_compute` package as an editable path dep). It creates/uses
a `.venv` in this folder. The default dependency set deliberately **excludes** torch/transformers so
the common path (hosted OpenRouter + scripted configs) skips a ~5 GB wheel. Only the three local
`distilgpt2` configs need those; install them via the optional `local` extra when you want to run the
`transformers` backend locally:

```bash
uv sync --extra local
```

With uv, torch is resolved from PyPI (the default build is the ~5 GB CUDA wheel). If you are on a
CPU-only Linux laptop and want the smaller (~0.8 GB) CPU build, point uv at the PyTorch CPU wheel
index, e.g. `uv sync --extra local --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu`
(or configure `[tool.uv.sources]`/`[[tool.uv.index]]` for torch). `docs/requirements-extras.txt` is a
separate minimal numerical/test subset — prefer `uv sync` here.

**Inference config.** Hosted model calls are configured once, centrally: copy the repo-root
`compute.env.example` → `compute.env` (one level up from this folder; `cp ../compute.env.example
../compute.env`) and set a `NEBIUS_API_KEY` or `OPENROUTER_API_KEY` (either works). Models Nebius
doesn't serve — including closed ones (GPT / Gemini / Claude) — need `OPENROUTER_API_KEY`. The hosted
clients pick up `base_url` + key automatically from that shared config via `poseidon_compute`. See
[`../COMPUTE.md`](../COMPUTE.md) for the full routing story (Nebius → OpenRouter → Modal → fail loud).
The scripted configs and the `distilgpt2` paths run locally and need **no key**.

Keys and any per-project overrides live in the repo-root `compute.env` — nothing to set up
in this folder.

## Quick start

The commands below (and everywhere in this README) call bare `python` / `scripts/*.py`. Run them
through uv — either prefix each with `uv run` (e.g. `uv run python -m auditor.experiments ...`), or
activate the `.venv` that `uv sync` created once and then call `python` directly:

```bash
source .venv/bin/activate     # from the steganographic-collusion/ directory, after `uv sync`
```

Run the CI smoke script (compiles the `auditor` and `colludebench` modules with `compileall`, then
executes targeted `pytest` suites):

```bash
scripts/ci_check.sh
```

Preview the full regeneration with no key and no calls:

```bash
scripts/regenerate_results.py --dry-run
```

Success signal: `ci_check.sh` exits 0 with its `pytest` suites passing, and the `--dry-run` prints the
calibrations, experiments, figures, and manifests it *would* rebuild without touching anything.

## Running experiments

### Split / dedupe the shipped transcripts first

`results/` is not just output: the detectors' thresholds and metrics are read from it, and it also
carries the **raw transcripts** of the study's runs (163 MB, 1,737 files) — the primary data this
project is about. Each run leaves two files next to each other:

| File | One line per | Fields |
| --- | --- | --- |
| `<run_id>.jsonl` | agent turn | `run_id`, `round_id`, `agent_id`, `role`, `message`, `action` (parsed from the message), `reward`, `interventions` {`paraphrased`, `budget`, `length`, `rate_limited`, `canary_injected`} |
| `<run_id>_interventions.jsonl` | round | `round`, `interventions` {role → the same five fields} |

**Read this before counting anything.** The original logger appended, and run ids are stable across
invocations, so every re-run of an experiment was written onto the previous transcript. As shipped,
**1,155 of the 1,737 files hold 2–48 runs back to back** with no separator: 22,127 appended runs in
total, of which only **3,918 are distinct** — the other 18,209 are byte-identical repeats of
deterministic scripted runs (≈140 of the 163 MB). `--dedupe` below keeps the 3,918 distinct runs
(47 MB). The `_interventions.jsonl` companion was written with truncation and describes only the
*last* run of its file. Thresholds and `metrics.json` are not affected — they were computed in memory
during the runs.

To work with the data, split the runs first (nothing under `results/` is modified):

```bash
python scripts/split_transcript_runs.py --dry-run     # counts only
python scripts/split_transcript_runs.py --dedupe      # results/ -> results_split/, one file per run
```

A run starts wherever `round_id` drops back to 1. The logger now truncates at the start of a run, so
anything you generate from here on is one run per file.

Where the files come from: `results/calibration/` (the honest runs the thresholds were calibrated on),
`results/audit/` (colluding audits), `results/sweeps/` (detector sensitivity points), `results/stress/`,
`results/experiments/` and `results/baseline/`. Nothing in the pipeline reads them back — the detectors
work on runs in memory — so they are yours for new analyses: audit why a run tripped a detector, build
a new detector, or verify the aggregate numbers against the actual messages. One JSON object per line;
`json.loads` per line is all the loader you need.

### Regenerate all results

Rebuild calibrations, experiments, figures, and manifests deterministically (key-free):

```bash
scripts/regenerate_results.py
```

### Hosted LLM experiments

The hosted configs keep the paper's models and route through the shared inference config set up in
**Setup**.

**Cheapest hosted smoke.** With an `OPENROUTER_API_KEY` set (in the repo-root `compute.env`), the
single cheapest hosted check runs directly against OpenRouter:

```bash
python -m auditor.experiments pricing_llm_groq    # ~160 calls on an open model (Llama 3.1 8B)
```

> **Heads up — this rewrites tracked files.** Running any `auditor.experiments` label overwrites its
> `results/experiments/<label>.json` and the per-seed transcripts under `<output_dir>/<scenario>/`,
> where `<output_dir>` is the `output_dir:` of that label's `configs/audits/<label>_audit.yaml`. For the
> hosted pricing labels that is `results/experiments/pricing_duopoly/` (~40 git-tracked files), but it
> differs by label — e.g. `auction_adaptive` writes to `results/audit/first_price_auction/` and
> `review_sparse` to `results/stress/review_sparse/`. `git status` will show the changes afterwards —
> this is expected. To get the shipped data back after a smoke test, check `git status --short` and
> `git restore` the paths it lists (e.g. `git restore results/`); `git restore results/experiments/`
> alone only cleans up labels that write there. Scripted labels are deterministic; hosted ones are not.

A hosted label with **no** inference key now fails loud (non-zero exit) instead of silently skipping,
so an empty `results/experiments/` is never mistaken for a successful run. Run the full hosted set
with:

```bash
python -m auditor.experiments pricing_llm_gemini pricing_llm_groq
```

`auditor.experiments` takes **labels** (not config filenames). The two hosted labels are
`pricing_llm_gemini` and `pricing_llm_groq`; each maps to a `configs/audits/<label>_audit.yaml`.
(`pricing_llm_open` is also a label, but it runs `distilgpt2` **locally** with no key — see
**Local open-weight experiments** below, not here.)

| Label / config prefix | Original | Routed model id |
| --- | --- | --- |
| `pricing_llm_gemini` | Gemini 2.5 Flash | `google/gemini-2.5-flash` |
| `pricing_llm_groq` | Llama 3.1 8B (Groq's `llama-3.1-8b-instant`) | `meta-llama/llama-3.1-8b-instruct` |
| `pricing_llm_closed` (config only) | GPT-4o-mini | `openai/gpt-4o-mini` |

The `backend:` label inside a config (`openrouter`, `gemini`, `groq`, …) **does not** select the
provider — the central router does; it is a no-op for provider selection.

To force a specific model for a run instead of the config's default, route the command through the
shared entry point: `poseidon-run --model <id> -- python -m auditor.experiments <label>`.

Each hosted audit is 20 seeds × 8 rounds = **160 calls** (~4 min on Gemini 2.5 Flash, a few cents) and
each honest calibration 10 × 8 = 80; the whole hosted suite is ~500 calls, about a dollar. Everything
else — the 22 scripted configs and the `distilgpt2` ones — runs locally with no key.

### Local open-weight experiments (`distilgpt2`)

`configs/pricing_llm_open*.yaml` and `review_llm_open.yaml` run `distilgpt2` locally via
`transformers` (run `uv sync --extra local` first — see Setup). Budget for it: torch is ~5 GB
installed from PyPI (~0.8 GB with the CPU wheel index, see Setup), the weights are 350 MB,
peak RAM ~1.2 GB, and generation is slow on CPU — about
7–8 s per turn on 8 cores, so the `pricing_llm_open` audit (20 seeds, config
`configs/audits/pricing_llm_open_audit.yaml`) takes ~40 min and the calibration (50 seeds) ~3 h.
`auditor.experiments` skips these labels with a message if `transformers` is not installed.

### Honest-run validation

To reproduce the 10k honest-run FPR bound (10,000 Monte Carlo replicates is the default):

```bash
python -m auditor.validate_union --replicates 10000
```

This resamples the shipped honest-run calibration samples; it takes no `--config` (there is no
`validate_union.yaml`). Results are written to `results/calibration/union_validation.json` (override
with `--output`); use `--seed` to change the RNG seed.

## Outputs

`scripts/regenerate_results.py` writes, deterministically:

- `results/calibration/detector_thresholds.json`
- `results/experiments/metrics.md`
- `results/dataset_manifest.json`
- `RESULTS.md` and `docs/experiment_results.md`

Honest-run validation writes `results/calibration/union_validation.json`.

It worked if these artifacts are present and `scripts/regenerate_results.py --dry-run` reports no
pending changes afterwards (the regeneration is deterministic, so a clean re-run reproduces the same
files).

## Analysis

- **Threat model:** `docs/threat_model.md`
- **Theory foundations & proofs:** `docs/theory_foundations.md`
- **Auditor pipeline:** `docs/auditor_pipeline.md`
- **External baselines & evaluation protocol:** `docs/baselines.md`
- **Benchmark design & usage:** `docs/colludebench_design.md`, `docs/colludebench_usage.md`
- **Capacity trade-offs:** `docs/capacity_tradeoffs.md`
- **Experiment summaries:** `docs/experiment_results.md`, `RESULTS.md`
- **Reproducibility:** `docs/REPRODUCIBILITY.md`, `docs/prerequisites.md`
- **Residual risks & limitations:** `docs/residual_risks.md`, `docs/limitations.md`
- **Literature review:** `docs/literature_review.md`

## Notes

- **Release bundle.** `release/` is a curated copy of `auditor/`, `colludebench/`, `configs/`,
  `scripts/`, `docs/`, `RESULTS.md` and `notebooks/` produced by `scripts/prepare_release.py` (a plain
  copy — it does not scrub or checksum anything). Regenerate it after changing code or docs, or the two
  copies drift; `release/package_manifest.json` records what was copied. See `release/README.md` for
  the bundle layout.
- **Citation.** If you use this work, please cite the paper and reference the artifact release. The
  `paper/` directory is not part of this release; cite the paper by title.
- **License.** Distributed under the MIT License. See `LICENSE` for details.
