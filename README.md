# SWE-Mutation: Can LLMs Generate Reliable Test Suites in Software Engineering?

<p align="center">
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-ACL%202026%20Findings-b31b1b"></a>
  <a href="#"><img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-SWE--Mutation-blue"></a>
  <a href="#"><img alt="Languages" src="https://img.shields.io/badge/Languages-10-green"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow"></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/Code%20%26%20Data-Coming%20Soon-orange"></a>
</p>

> **SWE-Mutation: Can LLMs Generate Reliable Test Suites in Software Engineering?**
> Yuxuan Sun, Yuze Zhao, Yufeng Wang, Yao Du, Zhiyuan Ma, Jinbo Wang, Mengdi Zhang, Kai Zhang, Zhenya Huang\*
> *Findings of ACL 2026*

This repository hosts the official code and data of **SWE-Mutation**, a repository-level benchmark that evaluates whether LLM-generated **test suites** are reliable and discriminative enough to be used as verification oracles for software engineering tasks. Instead of measuring a test suite against a single golden solution, SWE-Mutation confronts it with **systematically mutated** buggy solutions produced by an **agentic, language-agnostic mutation framework**, and asks: *how many realistic bugs can your tests actually catch?*

> Code and data are **coming soon**. Star / watch the repository to get notified.

---

## News

- **[2026/04]** Paper released. Code and benchmark data are being cleaned up and will be released here shortly.

---

## Motivation

Progress on the software engineering (SE) ability of LLMs — both for trajectory synthesis and for RL reward signals — is increasingly bottlenecked **not** by the scarcity of correct solutions, but by the scarcity of **reliable test suites**. Current practice treats tests as a fixed oracle, yet:

- Human-written test suites are expensive and often incomplete.
- LLM-generated test suites tend to be **superficial**, easily passing trivial mutants but missing realistic bugs.
- Existing mutation benchmarks rely on rule-based operators or few-shot prompting, which produce **unrealistic** faults that even weak test suites can kill.

SWE-Mutation measures test-suite quality by how well it resists **agent-crafted, semantically realistic mutants** on real GitHub repositories.

![overview](docs/figures/overview.png)
<p align="center"><i>(Figure placeholder — replace with Figure 2 of the paper on release.)</i></p>

---

## Key Features

- **Repository-Level**, executable environments built on real GitHub repos (not isolated snippets).
- **Agentic Mutation Framework** — `Locate → Mutation → Judge → Self-Play` — generates complex, realistic semantic mutants instead of trivial rule-based edits.
- **Multilingual** — a Python core plus a 9-language subset (**C, C++, Java, TypeScript, JavaScript, Rust, Go, PHP, Ruby**), built on top of SWE-bench Verified and SWE-bench-Multilingual.
- **Two Tasks**: *Test Generation* (from scratch) and *Test Repair* (augment an incomplete suite) — the latter closer to real-world SE workflows.
- **Agent-Framework Ready** — out-of-the-box support for [Mini-SWE-Agent](https://github.com/SWE-agent/mini-SWE-agent) and Claude Code.
- **Discriminative Metric**: Relative Detection Rate (**RDR**) on top of Pass@1 and Verified Reproduction Rate (**VRR**).

---

## Benchmark Statistics

| Split                     | Instances | Mutants | Languages                                 |
|---------------------------|-----------|---------|-------------------------------------------|
| SWE-Mutation (Python)     | 500       | 1,664   | Python                                    |
| SWE-Mutation-Multilingual | 300       | 972     | C, C++, Java, TS, JS, Rust, Go, PHP, Ruby |
| **Total**                 | **800**   | **2,636** | **10**                                  |

Each instance ships with **3–5 mutants** selected by a self-play procedure so that they evade at least 3 out of 10 sampled model-generated test suites, ensuring non-trivial difficulty.

---

## Agentic Mutation Framework

Given the golden solution and golden test suite of an instance, the framework autonomously produces realistic mutants through four modules:

1. **Locate** — restrict edits to files touched by the golden patch; parse them with Tree-sitter and overlay Fail-to-Pass execution traces to guide mutation to the bug-triggering logic.
2. **Mutation** — inject a bug following one of five human-error-inspired strategies (see paper Appendix A), with explicit rationale.
3. **Judge** — enforce three constraints: edits stay within allowed files, the patch applies & compiles, and at least one F2P test fails.
4. **Self-Play** — sample `N` candidate mutants, evaluate them against 10 temperature-sampled model-generated test suites, and keep the top-50% that survive against ≥3 suites.

The generator is Claude Sonnet 4 by default. The paper includes a robustness study swapping it for DeepSeek-V3.1 and Qwen3-Coder (RDR shifts within 1.5 pp, Spearman rank correlation of evaluators ≥ 0.93), showing conclusions are **not** driven by a same-family bias.

---

## Tasks & Metrics

### Tasks
- **Test Generation**: produce a complete test suite from scratch, given only the target file path.
- **Test Repair**: augment / fix an incomplete or flawed existing test suite.

### Metrics
- **Pass@1** — the generated patch applies cleanly and executes without compilation errors.
- **VRR** (Verified Reproduction Rate) — the suite **fails** on the buggy repo **and passes** on the golden code.
- **RDR** (Relative Detection Rate) — fraction of *previously-surviving* mutants killed by the model-generated suite:

$$
\mathrm{RDR} \;=\; \frac{\sum_{i=1}^{N} \left| M^{(i)}_{\text{gen}} \setminus M^{(i)}_{\text{base}} \right|}{\sum_{i=1}^{N} \left| M^{(i)} \setminus M^{(i)}_{\text{base}} \right|}
$$

For test generation, $M_{\text{base}} = \emptyset$ and RDR reduces to the absolute mutation score.

Instance-level bootstrap 95% CIs (10,000 resamples) and Wilcoxon signed-rank tests between the top model and each competitor are reported in the paper.

---

## Main Results

### Test Repair (Python)

| Model              | Mini-SWE-Agent Pass@1 / VRR / RDR | Claude Code Pass@1 / VRR / RDR |
|--------------------|-----------------------------------|--------------------------------|
| Claude-sonnet-4.5  | 97.20 / 42.60 / **79.30**         | 99.80 / **59.20** / **81.15**  |
| Claude-sonnet-3.7  | 94.40 / 29.80 / 62.58             | 97.60 / 52.80 / 66.59          |
| DeepSeek-V3.1      | 96.60 / 33.00 / 66.41             | 96.80 / 58.20 / 68.36          |
| Qwen3-Coder-480B   | 87.60 / 38.00 / 68.99             | 96.40 / 50.00 / 70.21          |
| Kimi-K2            | 83.80 / 40.60 / 74.58             | 86.00 / 54.40 / 74.19          |
| GLM-4.6            | 83.40 / 29.40 / 71.26             | 95.60 / 49.80 / 73.54          |
| GPT-oss-120B       | 74.80 / 24.80 / 36.31             | 86.80 / 36.40 / 39.28          |

### Test Generation (Python)

| Model              | Mini-SWE-Agent Pass@1 / VRR / RDR | Claude Code Pass@1 / VRR / RDR |
|--------------------|-----------------------------------|--------------------------------|
| Claude-sonnet-4.5  | 96.20 / **29.80** / **63.70**     | 98.00 / **40.40** / **71.71**  |
| Claude-sonnet-3.7  | 88.40 / 20.60 / 37.47             | 95.40 / 28.80 / 38.60          |
| DeepSeek-V3.1      | 88.20 / 10.20 / 36.15             | 94.00 / 20.40 / 39.09          |
| Qwen3-Coder-480B   | 86.20 / 12.40 / 33.33             | 95.20 / 26.80 / 33.21          |
| Kimi-K2            | 79.40 / 14.60 / 42.59             | 83.60 / 19.20 / 45.12          |
| GLM-4.6            | 74.60 / 15.20 / 39.79             | 86.20 / 25.40 / 42.11          |
| GPT-oss-120B       | 59.80 / 8.00 / 25.61              | 65.60 / 19.20 / 28.73          |

### Mutation Strategy Ablation (Test Generation, RDR)

| Model              | Rule-Based | Few-shot LLM | **Agentic (Ours)** |
|--------------------|-----------:|-------------:|-------------------:|
| Claude-sonnet-4.5  | 75.43      | 69.52        | **63.70 ↓**        |
| Claude-sonnet-3.7  | 73.25      | 55.25        | **37.47 ↓**        |
| DeepSeek-V3.1      | 72.92      | 52.86        | **36.15 ↓**        |
| Qwen3-Coder        | 72.16      | 50.18        | **33.33 ↓**        |
| Kimi-K2            | 74.12      | 62.43        | **42.59 ↓**        |
| GLM-4.6            | 73.88      | 59.55        | **39.79 ↓**        |
| GPT-oss-120B       | 55.55      | 35.27        | **25.61 ↓**        |

Average RDR drops from **71.04% → 39.81%** when switching from conventional mutants to our agentic mutants — evidence that previous benchmarks severely overestimate test-suite quality.

Full results on SWE-Mutation-Multilingual (9 languages) and additional ablations are reported in the paper.

---

## Repository Layout (planned)

> **🚧 Coming Soon.** The codebase, evaluation harness and dataset are being refactored for public release. The planned layout is:

```
SWE-Mutation/
├── data/                      # 🚧 coming soon — mutants, instance metadata, splits
│   ├── swe_mutation/          #     Python split (500 instances, 1,664 mutants)
│   └── swe_mutation_multi/    #     9-language split (300 instances, 972 mutants)
├── framework/                 # 🚧 agentic mutation framework (Locate / Mutation / Judge / Self-Play)
├── agents/                    # 🚧 adapters for Mini-SWE-Agent and Claude Code
├── evaluation/                # 🚧 Pass@1 / VRR / RDR scorers, bootstrap + Wilcoxon
├── scripts/                   # 🚧 reproduction scripts for tables in the paper
└── docs/
```

---

## Quick Start

> **🚧 Coming Soon.** Installation, dataset download, and evaluation commands will be documented here once the release is ready. The intended usage will look roughly like:

```bash
# coming soon
pip install swe-mutation
swe-mutation evaluate \
  --model claude-sonnet-4.5 \
  --agent mini-swe-agent \
  --task test_repair \
  --split python
```

---

## Citation

If you find SWE-Mutation useful for your research, please cite:

```bibtex
@inproceedings{sun2026swemutation,
  title     = {{SWE}-Mutation: Can {LLM}s Generate Reliable Test Suites in Software Engineering?},
  author    = {Sun, Yuxuan and Zhao, Yuze and Wang, Yufeng and Du, Yao and
               Ma, Zhiyuan and Wang, Jinbo and Zhang, Mengdi and
               Zhang, Kai and Huang, Zhenya},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

---

## Acknowledgements

SWE-Mutation is built on top of [SWE-bench](https://github.com/princeton-nlp/SWE-bench), [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) and [SWE-bench-Multilingual](https://github.com/SWE-bench/SWE-bench-Multilingual), and leverages [Mini-SWE-Agent](https://github.com/SWE-agent/mini-SWE-agent) and Claude Code as agentic scaffolds. We thank their authors for open-sourcing these resources.

---

## Contact

For questions about the paper or benchmark, please open an issue or reach out to:
**Yuxuan Sun** — `sunyuxuan@mail.ustc.edu.cn`
