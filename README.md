# Exploring a Pre-Registered Tripartite Evidence Framework for Auditing Multi-Agent AI Collusion

**Apart Research AI Collusion Sprint — Track 2.3 (*"What Should Count as Convincing Evidence of Collusion?"*)**  
📄 **[Read the Full Sprint Paper (PDF)](./Exploring%20a%20Pre-Registered%20Tripartite%20Evidence.pdf)**

---

## Overview

This repository contains my research submission for **Track 2.3** of the Apart Research AI Collusion Sprint. The project explores the epistemological challenge of distinguishing genuine, harmful multi-agent collusion from two pervasive confounders:
1. **Benign Parallel Conduct ("Common Incentives" / Case Type 2):** Agents independently converging on similar or high-payoff actions due to shared model priors, prompt anchors, or symmetric payoff structures.
2. **"Collusion on Paper" (Cheap Talk / Case Type 3):** Agents using conspiratorial language on a communication channel without successfully altering downstream actions to improve coalition payoff (including cases where colluders accidentally collide on duplicate tasks and degrade their own payoff).

To investigate whether multi-pillar auditing can reduce false alarms compared to single-metric detectors, I curated a 60-trajectory benchmark (`trajectories_60.jsonl` / `trajectories_60_scored.jsonl`) and evaluated a pre-registered **Tripartite Decision Rule** combining:
* **Conversation Evidence ($E_{\text{conv}}$):** LLM-as-a-judge transcript auditing (`Qwen/Qwen2.5-7B-Instruct`).
* **Coalition-Advantage Evidence ($E_{\text{adv}}$):** Regret-based and payoff-differential advantage metrics.
* **Causal / Behavioral Action Dependence ($E_{\text{dep}}$):** Counterfactual action shift across paired seeds (secret channel ON vs. OFF) and cross-round action synchronization.

---

## Repository Structure (Original Contributions vs. Upstream Frameworks)

```text
tripartite-collusion-audit/
├── README.md                                            # Project overview & provenance
├── Exploring a Pre-Registered Tripartite Evidence.pdf   # Final Apart Research Sprint paper
├── trajectories_60.jsonl                                # Curated 60-trajectory dataset (20 per Case Type)
├── trajectories_60_scored.jsonl                         # Dataset scored with all 3 pillars (E_conv, E_adv, E_dep)
├── frozen_decision_rule.json                            # Pre-registered thresholds & SHA-256 freeze hash
├── evaluation_results_summary.json                      # Held-out test metrics & pillar summaries
├── make_batch_configs.py                                # Generates 12-seed paired counterfactual YAML configs
├── build_dataset_60.py                                  # Aggregates Colosseum & ColludeBench trajectories
├── run_judge_and_evaluate.py                            # Runs Modal vLLM judge & 50/50 split-freeze-test audit
├── generate_figures_and_stats.py                        # Computes 1,000-bootstrap CIs & generates plots
├── figures/                                             # Publication figures generated from the evaluation
│   ├── fig1_evidence_space_projections.png
│   ├── fig2_detector_comparison_and_fpr.png
│   └── fig3_colosseum_paired_counterfactuals.png
├── colosseum/                                           # [Upstream] DCOP Jira environment & our 12-seed outputs
├── steganographic-collusion/                            # [Upstream] ColludeBench benchmark transcripts
└── util/                                                # [Upstream + Patched] Modal vLLM server (modal_serve.py)
```

---

## Summary of Preliminary Results (Held-Out Test Split, $N=30$)

Thresholds ($\tau_{\text{conv}}=0.40, \tau_{\text{adv}}=0.50, \tau_{\text{dep}}=0.40$) were calibrated exclusively on the 50% Discovery Split ($N=30$) and locked via SHA-256 hash (`8d1b12041adb7233...`) prior to evaluating on the 50% Held-Out Evaluation Split ($N=30$):

| Detector Variant | Held-Out F1 (95% Bootstrap CI) | Precision | Recall | FPR (Case 2: Common Incentives) | FPR (Case 3: Cheap Talk) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conversation Only ($E_{\text{conv}}$)** | `0.357 [0.105, 0.571]` | `0.278` | `0.500` | `0.300` | `1.000` |
| **Coalition Advantage Only ($E_{\text{adv}}$)** | `0.769 [0.533, 0.914]` | `0.625` | `1.000` | `0.600` | `0.000` |
| **Action Dependence Only ($E_{\text{dep}}$)** | `0.909 [0.727, 1.000]` | `0.833` | `1.000` | `0.200` | `0.000` |
| **Strict 3-Way AND** | `0.667 [0.286, 0.900]` | `1.000` | `0.500` | `0.000` | `0.000` |
| **Frozen Tripartite Rule (Ours)** | **`1.000 [1.000, 1.000]`** | **`1.000`** | **`1.000`** | **`0.000`** | **`0.000`** |

*Note on Limitations:* As discussed in Section 4 of the paper, the `1.000` F1 score on our held-out split should be interpreted cautiously given the modest test sample size ($N=30$), the inclusion of semi-synthetic cheap-talk ablations, and the availability of paired counterfactual baselines in `Colosseum`.

---

## Provenance & Acknowledgments

This repository builds upon foundational open-source benchmark infrastructure provided by **Poseidon Research** and the original benchmark authors for the AI Collusion Hackathon:
* **Colosseum DCOP Framework:** [`poseidon-research/collusion_hackathon/colosseum`](https://github.com/poseidon-research/collusion_hackathon/tree/main/colosseum) (adapted from *Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems*, UMass AI Safety, 2026).
* **ColludeBench Steganographic Benchmark:** [`poseidon-research/collusion_hackathon/steganographic-collusion`](https://github.com/poseidon-research/collusion_hackathon/tree/main/steganographic-collusion) (adapted from *Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs*, Ostailor et al., 2025).
* **Compute Utility (`util/poseidon_compute`):** Adapted from [`poseidon-research/collusion_hackathon/util`](https://github.com/poseidon-research/collusion_hackathon/tree/main/util), with custom patches to `modal_serve.py` pinning `vllm==0.6.6.post1` dependencies and enabling `--enable-auto-tool-choice` and `--tool-call-parser hermes` for `Qwen/Qwen2.5-7B-Instruct` on Modal A10G GPUs.

All paired 12-seed counterfactual `Colosseum` executions, dataset curation (`trajectories_60.jsonl`), LLM Judge evaluations, tripartite rule calibration, statistical analyses, and the final research paper were independently conducted and authored by me.
