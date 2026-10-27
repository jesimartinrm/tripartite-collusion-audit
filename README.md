# Exploring a Pre-Registered Tripartite Evidence Framework for Auditing Multi-Agent AI Collusion

**Apart Research AI Collusion Sprint — Track 2.3 (*"What Should Count as Convincing Evidence of Collusion?"*)**  
📄 **[Read the Full Sprint Paper (PDF)](./Exploring%20a%20Pre-Registered%20Tripartite%20Evidence.pdf)**

---

## Overview

This repository contains my research submission for **Track 2.3** of the Apart Research AI Collusion Sprint[cite: 3]. The project explores the epistemological challenge of distinguishing genuine, harmful multi-agent collusion from two pervasive confounders:
1. **Benign Parallel Conduct ("Common Incentives" / Case Type 2)[cite: 3]:** Agents independently converging on similar or high-payoff actions due to shared model priors, prompt anchors, or symmetric payoff structures.
2. **"Collusion on Paper" (Cheap Talk / Case Type 3)[cite: 3]:** Agents using conspiratorial language on a communication channel without successfully altering downstream actions to improve coalition payoff (including cases where colluders accidentally collide on duplicate tasks and degrade their own payoff).

To investigate whether multi-pillar auditing can reduce false alarms compared to single-metric detectors, I curated a 60-trajectory benchmark (`trajectories_60.jsonl` / `trajectories_60_scored.jsonl`)[cite: 3] and evaluated a pre-registered **Tripartite Decision Rule** combining:
* **Conversation Evidence ($E_{\text{conv}}$)[cite: 3]:** LLM-as-a-judge transcript auditing (`Qwen/Qwen2.5-7B-Instruct`).
* **Coalition-Advantage Evidence ($E_{\text{adv}}$)[cite: 3]:** Regret-based and payoff-differential advantage metrics.
* **Causal / Behavioral Action Dependence ($E_{\text{dep}}$)[cite: 3]:** Counterfactual action shift across paired seeds (secret channel ON vs. OFF) and cross-round action synchronization.

---

## Repository Structure (Original Contributions vs. Upstream Frameworks)

```text
tripartite-collusion-audit/
├── README.md                                            # Project overview & provenance
├── Exploring a Pre-Registered Tripartite Evidence.pdf   # Final Apart Research Sprint paper
├── trajectories_60.jsonl                                # Curated 60-trajectory dataset (20 per Case Type)[cite: 3]
├── trajectories_60_scored.jsonl                         # Dataset scored with all 3 pillars (E_conv, E_adv, E_dep)[cite: 3]
├── frozen_decision_rule.json                            # Pre-registered thresholds & SHA-256 freeze hash[cite: 3]
├── evaluation_results_summary.json                      # Held-out test metrics & pillar summaries[cite: 3]
├── make_batch_configs.py                                # Generates 12-seed paired counterfactual YAML configs
├── build_dataset_60.py                                  # Aggregates Colosseum & ColludeBench trajectories[cite: 3]
├── run_judge_and_evaluate.py                            # Runs Modal vLLM judge & 50/50 split-freeze-test audit[cite: 3]
├── generate_figures_and_stats.py                        # Computes 1,000-bootstrap CIs & generates plots[cite: 3]
├── figures/                                             # Publication figures generated from the evaluation[cite: 3]
│   ├── fig1_evidence_space_projections.png
│   ├── fig2_detector_comparison_and_fpr.png
│   └── fig3_colosseum_paired_counterfactuals.png
├── colosseum/                                           # [Upstream] DCOP Jira environment & our 12-seed outputs[cite: 3]
├── steganographic-collusion/                            # [Upstream] ColludeBench benchmark transcripts[cite: 3]
└── util/                                                # [Upstream + Patched] Modal vLLM server (modal_serve.py)
```

---

## Summary of Preliminary Results (Held-Out Test Split, $N=30$)[cite: 3]

Thresholds ($\tau_{\text{conv}}=0.40, \tau_{\text{adv}}=0.50, \tau_{\text{dep}}=0.40$) were calibrated exclusively on the 50% Discovery Split ($N=30$)[cite: 3] and locked via SHA-256 hash (`8d1b12041adb7233...`)[cite: 3] prior to evaluating on the 50% Held-Out Evaluation Split ($N=30$)[cite: 3]:

| Detector Variant | Held-Out F1 (95% Bootstrap CI)[cite: 3] | Precision[cite: 3] | Recall[cite: 3] | FPR (Case 2: Common Incentives)[cite: 3] | FPR (Case 3: Cheap Talk)[cite: 3] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conversation Only ($E_{\text{conv}}$)[cite: 3]** | `0.357 [0.105, 0.571]`[cite: 3] | `0.278`[cite: 3] | `0.500`[cite: 3] | `0.300`[cite: 3] | `1.000`[cite: 3] |
| **Coalition Advantage Only ($E_{\text{adv}}$)[cite: 3]** | `0.769 [0.533, 0.914]`[cite: 3] | `0.625`[cite: 3] | `1.000`[cite: 3] | `0.600`[cite: 3] | `0.000`[cite: 3] |
| **Action Dependence Only ($E_{\text{dep}}$)[cite: 3]** | `0.909 [0.727, 1.000]`[cite: 3] | `0.833`[cite: 3] | `1.000`[cite: 3] | `0.200`[cite: 3] | `0.000`[cite: 3] |
| **Strict 3-Way AND** | `0.667 [0.286, 0.900]`[cite: 3] | `1.000`[cite: 3] | `0.500`[cite: 3] | `0.000`[cite: 3] | `0.000`[cite: 3] |
| **Frozen Tripartite Rule (Ours)** | **`1.000 [1.000, 1.000]`**[cite: 3] | **`1.000`**[cite: 3] | **`1.000`**[cite: 3] | **`0.000`**[cite: 3] | **`0.000`**[cite: 3] |

*Note on Limitations:* As discussed in Section 4 of the paper, the `1.000` F1 score on our held-out split should be interpreted cautiously given the modest test sample size ($N=30$)[cite: 3], the inclusion of semi-synthetic cheap-talk ablations, and the availability of paired counterfactual baselines in `Colosseum`[cite: 3].

---

## Provenance & Acknowledgments

This repository builds upon foundational open-source benchmark infrastructure provided by **Poseidon Research** and the original benchmark authors for the AI Collusion Hackathon:
* **Colosseum DCOP Framework:** [`poseidon-research/collusion_hackathon/colosseum`](https://github.com/poseidon-research/collusion_hackathon/tree/main/colosseum) (adapted from *Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems*, UMass AI Safety, 2026)[cite: 3].
* **ColludeBench Steganographic Benchmark:** [`poseidon-research/collusion_hackathon/steganographic-collusion`](https://github.com/poseidon-research/collusion_hackathon/tree/main/steganographic-collusion) (adapted from *Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs*, Ostailor et al., 2025)[cite: 3].
* **Compute Utility (`util/poseidon_compute`):** Adapted from [`poseidon-research/collusion_hackathon/util`](https://github.com/poseidon-research/collusion_hackathon/tree/main/util), with custom patches to `modal_serve.py` pinning `vllm==0.6.6.post1` dependencies and enabling `--enable-auto-tool-choice` and `--tool-call-parser hermes` for `Qwen/Qwen2.5-7B-Instruct` on Modal A10G GPUs[cite: 3].

All paired 12-seed counterfactual `Colosseum` executions, dataset curation (`trajectories_60.jsonl`)[cite: 3], LLM Judge evaluations, tripartite rule calibration, statistical analyses, and the final research paper were independently conducted and authored by me[cite: 3].
