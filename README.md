<div align="center">

# ORDER: Task-Conditioned Routing for Retrieval-Augmented Generation

A query-conditioned router that picks per-cluster chunking, metadata, and source policies, turning a one-size-fits-all RAG into a multi-source, multi-strategy retriever.

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-orange?style=flat-square)](https://2026.emnlp.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg?style=flat-square)](#license--acknowledgements)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.x-7c4dff?style=flat-square)](https://www.trychroma.com/)
[![Cohere](https://img.shields.io/badge/Cohere-embed--v4.0-39594d?style=flat-square)](https://docs.cohere.com/)

<p align="center">
  <img src="paper_EMNLP/images/header_1.png" width="92%" alt="ORDER framework overview"><br/>
  <em>Figure 1 — ORDER routes each query through three switches at inference time: chunking granularity, metadata policy, and source allocation. Index construction and per-cluster strategy discovery are done once, offline.</em>
</p>

</div>

> This repository is anonymised for EMNLP 2026 double-blind review. Please do not attempt to identify the authors.

---

## Overview

Retrieval-Augmented Generation pipelines typically treat every query the same way: one chunking strategy, one metadata index, one retriever, one source pool. On heterogeneous, size-imbalanced corpora this fails twice: long-form sources contaminate the top-k for queries that should never touch them, while small sources are starved by the dominant index.

ORDER replaces the one-size-fits-all assumption with a task-conditioned router that flips three switches per query:

1. **Which chunking granularity?** — TC-Chunking (C2)
2. **Which metadata filter / rerank?** — TC-Metadata (C3) and joint C × M (C4)
3. **Which source(s), and in what proportion?** — QRe + UMS

All four mechanisms share one offline footprint: a UMAP-10D + HDBSCAN cluster map learned over training-question embeddings, plus per-cluster strategy scores. At inference the query is embedded, projected, assigned to its nearest centroid, and served from the index/policy that cluster prefers.

Key properties:

- **Task-conditioned indexing.** Each semantic cluster of queries gets its own chunking strategy, metadata policy, or joint pair, with no per-query re-indexing. All 14 (or 14 × 14 = 196) configurations are built once, offline, and selected at inference by nearest-centroid lookup.
- **Query router (QRe).** A logistic classifier on Cohere `embed-v4.0` question embeddings separates press-only from parliamentary-touching queries (99.4 % accuracy on held-out queries).
- **Uniform multi-source allocator (UMS).** Distributes the top-k budget across sources to mitigate source starvation. Composes with QRe.
- **Multi-hop retrieval.** Recall@3 of 52.3 on the HistoriQA-ThirdRepublic multi-hop set, vs 31.7 for HippoRAG v2 and LinearRAG with the same embedding budget.
- **Reproducible.** 5 seeds (`[42, 123, 2021, 7, 9999]`), Bonferroni paired *t*-tests, 50/50 train/test split, end-to-end pipeline in seven `python -m ...` commands.

QRe and UMS target structurally distinct failure modes — source contamination vs. source starvation — and compose without overlap.

---

## Table of contents

1. [Framework overview](#framework-overview)
2. [Research questions](#research-questions)
3. [Results from the paper](#results-from-the-paper)
4. [The four mechanisms in detail](#the-four-mechanisms-in-detail)
5. [Dataset: HistoriQA-ThirdRepublic](#dataset-historiqa-thirdrepublic)
6. [Repository layout](#repository-layout)
7. [Setup](#setup)
8. [End-to-end reproduction](#end-to-end-reproduction)
9. [Notebook walkthroughs](#notebook-walkthroughs)
10. [Paper ↔ code mapping](#paper--code-mapping)
11. [Limitations](#limitations)
12. [Citation](#citation)
13. [License & acknowledgements](#license--acknowledgements)

---

## Framework overview

ORDER splits retrieval optimisation along two axes:

| Phase                  | Cost     | What ORDER does                                                                                                                                                            |
| ---------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Offline** (once)     | Heavy    | Build 14 chunkings of *Les Débats*; embed each into ChromaDB (Cohere `embed-v4.0`); cluster training queries; score every (cluster × strategy) pair; train QRe classifier. |
| **Online** (per-query) | Light    | 1 embedding call → 1 UMAP projection → 1 nearest-centroid lookup → 1 classifier prediction → retrieve from the chosen index with the chosen metadata policy and source mix. |

<p align="center">
  <img src="paper_EMNLP/images/qre_ums_example_compact_v2.png" width="86%" alt="QRe + UMS worked example"><br/>
  <em>Figure 2 — Worked example: QRe gates source eligibility from the query class; UMS divides the top-k budget uniformly across the eligible sources, addressing both contamination and starvation.</em>
</p>

### Inference algorithm (paper Algorithm 1)

```
Input: query q, UMAP transform T, centroids {μ_c}, strategy map s*(·), budget k
1. z_q  ← T( embed(q) )                       # one UMAP projection
2. ĉ    ← argmin_c  ‖z_q − μ_c‖₂              # one nearest-centroid lookup
3. cfg  ← s*(ĉ)                               # cluster's chunking / metadata / source policy
4. return Retrieve(q, cfg, k)
```

The same template instantiates **C2** (𝒮 = 14 chunkings), **C3** (𝒮 = 14 metadata strategies), and **C4** (𝒮 = 𝒮_chunk × 𝒮_meta, 196 pairs).

---

## Research questions

The paper is organised around five questions; this repo provides one runnable artefact per RQ.

| RQ      | Question                                                                                              | Reproduced by                                                                                                       |
| ------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **RQ1** | Does query-conditioned **chunking** beat a global best?                                               | [scripts/run_C2_multiseed_kfold.py](scripts/run_C2_multiseed_kfold.py) · analysis: [notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb](notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb) |
| **RQ2** | Does query-conditioned **metadata indexing** on *Les Débats* beat a uniform global baseline?          | [scripts/run_C3_multiseed_kfold.py](scripts/run_C3_multiseed_kfold.py) · analysis: [notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb](notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb) |
| **RQ3** | Does **joint** optimisation of chunking × metadata beat either single axis?                           | [scripts/run_C4_joint_multiseed.py](scripts/run_C4_joint_multiseed.py) · cache builder: [scripts/c4_joint/](scripts/c4_joint/) |
| **RQ4** | Does query-conditioned **source selection** beat a single-index baseline?                             | **Analysis notebook:** [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb) |
| **RQ5** | Are **QRe** (router) and **UMS** (allocator) complementary remedies for size-imbalanced retrieval?    | **Analysis notebook:** [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb) |

---

## Results from the paper

### Table 1 — Task-conditioned indexing (RQ1 / RQ2 / RQ3)

Coverage@k / Recall@k on the held-out test set (*n* = 438, 50/50 split, 5 seeds). Each retriever is compared to its Global Baseline (`10000_hierarchical`, no metadata) across three task-conditioned variants. \* = *p* < 0.05 (Bonferroni-paired *t*-test); **ns** = not significant.

| Retriever         | Indexing strategy   | Cov\@3      | Cov\@5      | Rec\@3      | Rec\@5      |
| ----------------- | ------------------- | ----------- | ----------- | ----------- | ----------- |
| **Naïve**         | Global Baseline     | 36.9        | 45.4        | 36.9        | 45.6        |
|                   | + TC-Chunking       | 42.3\*      | 51.7\*      | 42.3\*      | 51.7\*      |
|                   | + TC-Metadata       | **45.8\***  | **55.2\***  | **45.9\***  | **55.3\***  |
|                   | + TC-Joint          | *45.7*\*    | *54.7*\*    | *45.5*\*\*  | *54.5*\*    |
| **QRe**           | Global Baseline     | 48.5        | 58.2        | 48.7        | 58.4        |
|                   | + TC-Chunking       | 48.5 ns     | 57.9 ns     | 48.5 ns     | 58.0 ns     |
|                   | + TC-Metadata       | 48.5 ns     | 58.2 ns     | 48.7 ns     | 58.4 ns     |
|                   | + TC-Joint          | 48.3 ns     | 57.6 ns     | 48.2 ns     | 57.4 ns     |
| **UMS**           | Global Baseline     | 46.8        | 58.7        | 47.0        | 58.8        |
|                   | + TC-Chunking       | 47.0 ns     | 58.5 ns     | 47.2 ns     | 58.7 ns     |
|                   | + TC-Metadata       | **52.8\***  | **63.2\***  | **53.0\***  | **63.4\***  |
|                   | + TC-Joint          | *52.8*\*    | *63.1*\*    | *53.0*\*    | *63.4*\*    |
| **QRe & UMS**     | Global Baseline     | 54.7        | 64.5        | 54.9        | 64.7        |
|                   | + TC-Chunking       | 54.7 ns     | 64.5 ns     | 54.9 ns     | 64.7 ns     |
|                   | + TC-Metadata       | 54.7 ns     | 64.5 ns     | 54.9 ns     | 64.7 ns     |
|                   | + TC-Joint          | 54.7 ns     | 64.5 ns     | 54.9 ns     | 64.7 ns     |

*Bold = best per metric within the retriever block; italics = second-best.*

Notes:

- Under the Naïve retriever, all three task-conditioned variants help significantly (+5.4 / +8.9 / +8.8 pp Cov@3).
- TC-Metadata is the only variant that persists under a non-trivial retriever (+6.0 pp Cov@3 under UMS).
- TC-Chunking gains vanish once QRe is active: chunking and routing address the same failure mode (source contamination), so the gains do not stack.
- The joint variant matches but does not strictly dominate TC-Metadata, reflecting a winner's-curse effect on the 14 × 14 = 196-pair search space.

### Table 2 — Retrieval strategies on multi-hop questions (RQ4 / RQ5)

Recall@k on the multi-hop subset of HistoriQA-ThirdRepublic, split by question type — *cross-newspaper* (314 q., both gold passages in newspapers) and *newspaper → Débats* (571 q., one newspaper + one parliamentary document).

| Configuration                          | Overall R\@3 | R\@5     | R\@10    | Cross-news. R\@3 | News.→Débats R\@3 |
| -------------------------------------- | ------------ | -------- | -------- | ---------------- | ----------------- |
| Naïve Retriever                        | 35.9         | 44.0     | 54.4     | 14.3             | 47.8              |
| BM25 (lexical)                         | 23.1         | 29.3     | 37.8     | 4.6              | 33.3              |
| HippoRAG v2                            | 31.7         | 41.9     | 53.0     | 11.6             | 42.7              |
| LinearRAG                              | 31.7         | 41.9     | 53.0     | 11.6             | 42.7              |
| **S-RAG: QRe**                         | 48.0         | 57.0     | 68.4     | 48.4             | 47.8              |
| **S-RAG: UMS**                         | *50.2*       | *57.6*   | *69.4*   | 40.8             | 55.4              |
| **S-RAG: QRe & UMS** (ours, full)      | **52.3**     | **58.4** | **72.3** | 46.7             | 55.4              |

Where each gain comes from:

- QRe lifts retrieval on cross-newspaper questions (parliamentary contamination is the bottleneck) from 14.3 to 48.4 R@3.
- UMS lifts newspaper → Débats questions (press is starved by the much larger parliamentary index) from 47.8 to 55.4 R@3.
- The combined router composes both gains: there is no setting in which QRe-only or UMS-only beats QRe & UMS.

Full analysis (per-source breakdowns and ablations over the QRe classifier) is in [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb).

---

## The four mechanisms in detail

### 1. TC-Chunking (C2) — *which granularity?*

The 14 candidate chunking strategies, grouped by boundary signal:

| Group              | Strategy                       | Boundary signal                                           |
| ------------------ | ------------------------------ | --------------------------------------------------------- |
| **A — Baseline**   | `10000_hierarchical` †         | Section headers + recursive splitter (production default) |
|                    | `fixed_{500, 1k, 2k, 10k}`     | Character windows only                                    |
| **B — Speaker**    | `S2_atomic`                    | One segment per speaker turn                              |
|                    | `S3_window3`                   | Sliding 3-turn window                                     |
|                    | `S4_president`                 | Presidential interventions                                |
| **C — Procedural** | `S8_vote`                      | Vote resolution markers                                   |
|                    | `S10_amendment`                | Amendment lifecycle                                       |
| **D — Topic**      | `S9_topic`                     | Agenda-item transitions                                   |
|                    | `S9_topic_noVotes`             | Agenda + vote-list removal                                |
|                    | `S11_hybrid`                   | Topic outer / speaker-turn inner                          |
|                    | `S11_hybrid_noVotes`           | Hybrid + vote-list removal                                |

† Production baseline against which all C2 deltas are measured. Re-chunking is restricted to *Les Débats* — newspaper collections keep article-level segmentation throughout.

Implementation: [scripts/segmentation_strategies/](scripts/segmentation_strategies/) · Multi-seed harness: [scripts/run_C2_multiseed_kfold.py](scripts/run_C2_multiseed_kfold.py) · **Analysis notebook:** [notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb](notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb).

### 2. TC-Metadata (C3) — *which units are eligible?*

A second, orthogonal axis: 14 metadata strategies in three families operating on the *Les Débats* index:

| Family       | Examples                                                                              | Action                                            |
| ------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Baseline** | `baseline` †                                                                          | No filter, no rerank                              |
| **Filter**   | `exclude_noisy`, `debates_only`, `debates_and_legal`, `min_speakers_2`, `min_length_500`, `exclude_noisy_min_length` | Boolean predicate as ChromaDB `where` clause     |
| **Rerank**   | `rerank_type_boost`, `rerank_type_penalty`, …                                          | Add a signed bonus / penalty δ to cosine distance for documents whose `doc_type` matches a boost or penalty set |
| **Hybrid**   | `filter + rerank` compositions                                                         | Apply both                                        |

Full predicates, α values, and type-set definitions: paper Appendix E ([paper_EMNLP/latex/E_metadata_strats.tex](paper_EMNLP/latex/E_metadata_strats.tex)). Multi-seed harness: [scripts/run_C3_multiseed_kfold.py](scripts/run_C3_multiseed_kfold.py) · **Analysis notebook:** [notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb](notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb).

### 3. Joint chunking × metadata (C4)

Each cluster selects its best **pair** (c\*, m\*) from the 14 × 14 = 196-pair Cartesian grid. Scoring all pairs at every seed × cluster is expensive, so C4 is split into three stages:

1. **Cache** ([scripts/c4_joint/c4_1_build_cache.py](scripts/c4_joint/c4_1_build_cache.py)) — score every (chunking, metadata) pair on the full training set, once.
2. **Score** ([scripts/c4_joint/c4_2_score_clusters.py](scripts/c4_joint/c4_2_score_clusters.py)) — read the cache, project to per-cluster scores per seed.
3. **Evaluate** ([scripts/c4_joint/c4_3_evaluate_test.py](scripts/c4_joint/c4_3_evaluate_test.py)) — assign held-out queries via nearest centroid, evaluate the selected pair.

Driver: [scripts/run_C4_joint_multiseed.py](scripts/run_C4_joint_multiseed.py).

### 4. QRe (Query Rerouting) + UMS (Uniform Multi-Source) — *which sources?*

The retrieval-side mechanisms live entirely in [scripts/retrieval_utils.py](scripts/retrieval_utils.py) as four routing functions, all sharing the same input signature for clean composition:

| Function                                          | Strategy                                                                         |
| ------------------------------------------------- | -------------------------------------------------------------------------------- |
| `apply_no_rerouting`                              | Naïve baseline — top-N by cosine distance across the union of all collections    |
| `apply_query_rerouting`                           | **QRe** — keep only the classifier-predicted collections, then top-N within them |
| `apply_forced_equal_from_all_collections`         | **UMS** — equal per-source quota, ignoring the classifier                        |
| `apply_forced_multicollection`                    | **QRe & UMS** — equal per-source quota inside the classifier-predicted set       |

QRe is trained in [notebooks/31_classify_questions.ipynb](notebooks/31_classify_questions.ipynb) (99.4 % accuracy on a binary press-vs-parliamentary task) over the question embeddings produced in [notebooks/30_embed_questions.ipynb](notebooks/30_embed_questions.ipynb).

The full end-to-end QRe + UMS evaluation — Recall@k tables, per-source breakdowns, and the cross-newspaper vs newspaper→Débats ablations — is in [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb).

---

## Dataset: HistoriQA-ThirdRepublic

ORDER is benchmarked on the **1887 slice** of HistoriQA-ThirdRepublic (LREC-COLING 2026), a French historian-validated benchmark over three heterogeneous sources from the Third Republic.

| Source                                | Role                                        | Chunks  |
| ------------------------------------- | ------------------------------------------- | ------- |
| **Les Débats parlementaires** (Chambre des Députés) | Long-form parliamentary transcripts (OCR) | 3,229   |
| **Le Gaulois**                        | Monarchist/conservative newspaper           | 78      |
| **L'Intransigeant**                   | Socialist newspaper                         | 79      |
| **Total**                             |                                             | **3,386** |

| Question type                         | Count | Description                                                      |
| ------------------------------------- | ----- | ---------------------------------------------------------------- |
| Single-hop (SH)                       | 897   | Answerable from a single chunk                                   |
| Multi-hop — Generic                   | 541   | Synthesis of two documents                                       |
| Multi-hop — Comparative               | 192   | Comparison across sources                                        |
| Multi-hop — BridgeEntity              | 152   | First doc supplies context to interpret the second               |
| **Total**                             | **1,782** |                                                              |

The three structural properties that **drive** the ORDER design:

1. **Length heterogeneity** — *Débats* chunks are an order of magnitude longer than newspaper articles → no single chunking granularity fits all.
2. **Corpus imbalance** — *Les Débats* is **> 6×** larger than both newspapers combined and supplies **80.6 %** of Naïve-baseline top-3 hits even on newspaper-only gold.
3. **Cross-source multi-hop reasoning** — multi-hop questions require evidence from multiple corpora; top-k cosine ranking cannot guarantee this when one corpus dominates.

Properties (1) → TC-Chunking. Properties (2)–(3) → QRe + UMS.

**Where to get the data**

HistoriQA-ThirdRepublic is the companion benchmark cited in the paper (LREC-COLING 2026). For the review period, dataset URLs and the preprint citation have been withheld to preserve double-blind anonymity; they will be restored in the camera-ready version. Reviewers with access to the supplementary materials will find the JSONL question and corpus files there.

Drop the downloaded files under `data/` so the layout matches what [scripts/config.py](scripts/config.py) expects:

```
data/
├── corpus_1887/                # raw daily issues of Les Débats parlementaires
├── legaulois_1887/             # Le Gaulois daily issues
├── lintransigeant_1887/        # L'Intransigeant daily issues
└── questions/                  # train / dev / test question files (JSONL)
```

All other folders under `data/` (`corpus_1887_splitted_v2/`, `embeddings_*/`, `bm25_*/`, `RETRIEVER_results_*/`, `multiseed_results/`, …) are **derived artefacts** regenerated by the pipeline. They are gitignored.

---

## Repository layout

<details>
<summary>Click to expand the full tree</summary>

```
ORDER_EMNLP2026/
├── scripts/                                  ─── core pipeline ──────────────
│   ├── config.py                             # paths + API keys (loads .env)
│   ├── chunk_debats.py                       # split Les Débats into sections
│   ├── generate_all_chunk_variants.py        # 14 chunking strategies on Les Débats
│   ├── embed_chunk_variants.py               # embed each variant → ChromaDB
│   ├── build_indexes.py                      # Cohere / OpenRouter + BM25 indexes (3 sources)
│   ├── retrieval_utils.py                    # retrievers + QRe / UMS routing + metrics
│   ├── run_C2_multiseed_kfold.py             # C2 — TC-Chunking (multi-seed)
│   ├── run_C3_multiseed_kfold.py             # C3 — TC-Metadata (multi-seed)
│   ├── run_C4_joint_multiseed.py             # C4 — Joint chunking × metadata
│   ├── c4_joint/                             # C4 internals: cache → score → evaluate
│   │   ├── c4_1_build_cache.py
│   │   ├── c4_2_score_clusters.py
│   │   └── c4_3_evaluate_test.py
│   ├── segmentation_strategies/              # 14 chunking strategy definitions (+ README)
│   ├── SOTA/
│   │   ├── HippoRAGv2-main/                  # HippoRAG v2 baseline (B1 / B2 indexing scripts)
│   │   └── LinearRAG-main/                   # LinearRAG baseline (own requirements.txt)
│   └── utils/                                # ChromaDB / IO helpers
│
├── notebooks/                                ─── analysis notebooks ─────────
│   ├── 10_retriever_evaluation.ipynb         # dense retriever sanity check
│   ├── 11_bm25_retrieval_evaluation.ipynb    # BM25 baseline
│   ├── 12_retrieval_evaluation_generic.ipynb # generic retrieval comparison
│   ├── 13_linearRAG_evaluation.ipynb         # LinearRAG eval
│   ├── 14_hipporag_MH.ipynb                  # HippoRAG v2 multi-hop eval
│   ├── 19_collection_explorer.ipynb          # interactive corpus exploration
│   ├── 30_embed_questions.ipynb              # build QRe question embeddings
│   ├── 31_classify_questions.ipynb           # train / inspect the QRe classifier
│   ├── 32_retriever_evaluation_query_rerouting.ipynb   # QRe + UMS evaluation (RQ4–RQ5)
│   └── single_seed/                          # single-seed walkthroughs
│       ├── 40_C2_task_conditioned_chunking.ipynb       # C2 analysis (RQ1)
│       └── 41_C3_task_conditioned_metadata.ipynb       # C3 analysis (RQ2)
│
├── paper_EMNLP/                              # LaTeX source + figures
│   ├── latex/                                # 0_main.tex, sections, appendices
│   └── images/                               # framework diagrams + figures
├── data/                                     # raw + derived data (mostly gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

</details>

The three analysis notebooks that reproduce the main result tables and figures of the paper are listed under [Notebook walkthroughs](#notebook-walkthroughs) below.

---

## Setup

The codebase targets **Python ≥ 3.10**. All scripts are executed from the project root using the **`base` conda environment**.

### 1. Clone

```powershell
git clone <repo-url> ORDER_EMNLP2026
cd ORDER_EMNLP2026
```

### 2. Python dependencies

```powershell
conda activate base
pip install -r requirements.txt
```

> **ChromaDB pinning.** Stay on the **0.5.x** line — collections written with ≥ 0.6 are not backward-compatible with this repo.

### 3. API keys (`.env`)

Copy `.env.example` to `.env` at the repo root:

```text
COHERE_API_KEY=sk-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...     # optional, only for the OpenRouter embedding path
```

Loaded by [scripts/config.py](scripts/config.py) via `python-dotenv`. The `.env` file is gitignored.

### 4. Data

Follow the [Dataset](#-dataset-historiqa-thirdrepublic) section to drop the HistoriQA-ThirdRepublic 1887 files under `data/`.

### 5. (Optional) SOTA baseline dependencies

The HippoRAG v2 and LinearRAG baselines have their own deps; install only if you intend to reproduce those rows of Table 2:

```powershell
pip install -r scripts/SOTA/LinearRAG-main/requirements.txt
# HippoRAG v2: see scripts/SOTA/HippoRAGv2-main/ for upstream instructions
```

---

## End-to-end reproduction

Seven `python -m ...` commands from the project root, with the `base` conda environment active. Steps 1–4 are one-shot indexing; 5–7 are the contribution experiments.

```powershell
# 1. Split Les Débats parlementaires into per-section files
python -m scripts.chunk_debats

# 2. Generate the 14 chunking variants on Les Débats
python -m scripts.generate_all_chunk_variants

# 3. Embed every chunk variant into ChromaDB (Cohere embed-v4.0 by default)
python -m scripts.embed_chunk_variants

# 4. Build base embeddings + BM25 indexes for the 3 sources
python -m scripts.build_indexes --all
#    (use --provider openrouter or --force to override the defaults)

# 5. C2 — Task-Conditioned Chunking (multi-seed)            → Table 1, "+ TC-Chunking" rows
python -m scripts.run_C2_multiseed_kfold

# 6. C3 — Task-Conditioned Metadata (multi-seed)            → Table 1, "+ TC-Metadata" rows
python -m scripts.run_C3_multiseed_kfold

# 7. C4 — Joint chunking × metadata (cache + multi-seed)    → Table 1, "+ TC-Joint" rows
python -m scripts.c4_joint.c4_1_build_cache
python -m scripts.run_C4_joint_multiseed
```

**SOTA baselines** (rows of Table 2):

```powershell
python -m scripts.SOTA.HippoRAGv2-main.B1_hipporag_indexing
python scripts/SOTA/LinearRAG-main/run.py
```

**QRe + UMS** (Table 2, main contribution): open [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb) and run all cells. The notebook loads the universal retrieval cache produced by step 4 and evaluates all four routing strategies end-to-end.

---

## Notebook walkthroughs

| Notebook                                                                                                                  | What it shows                                                              | RQ        |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------- |
| [notebooks/10_retriever_evaluation.ipynb](notebooks/10_retriever_evaluation.ipynb)                                        | Naïve dense retriever sanity check on the single-hop set                   | baseline  |
| [notebooks/11_bm25_retrieval_evaluation.ipynb](notebooks/11_bm25_retrieval_evaluation.ipynb)                              | BM25 lexical baseline (Table 2, row 2)                                     | baseline  |
| [notebooks/12_retrieval_evaluation_generic.ipynb](notebooks/12_retrieval_evaluation_generic.ipynb)                        | Naïve vs BM25 on the generic multi-hop slice                               | baseline  |
| [notebooks/13_linearRAG_evaluation.ipynb](notebooks/13_linearRAG_evaluation.ipynb)                                        | LinearRAG SOTA baseline (Table 2, row 4)                                   | SOTA      |
| [notebooks/14_hipporag_MH.ipynb](notebooks/14_hipporag_MH.ipynb)                                                          | HippoRAG v2 SOTA baseline (Table 2, row 3)                                 | SOTA      |
| [notebooks/19_collection_explorer.ipynb](notebooks/19_collection_explorer.ipynb)                                          | Interactive exploration of ChromaDB collections + UMAP projections          | tooling   |
| [notebooks/30_embed_questions.ipynb](notebooks/30_embed_questions.ipynb)                                                  | Build Cohere embeddings for QRe training                                   | QRe       |
| [notebooks/31_classify_questions.ipynb](notebooks/31_classify_questions.ipynb)                                            | Train the QRe classifier — 99.4 % accuracy, confusion matrix, ablations    | QRe       |
| **[notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb)**    | **QRe + UMS end-to-end retrieval evaluation (Table 2 in this README)**     | **RQ4–5** |
| **[notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb](notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb)** | **C2 single-seed walkthrough — per-cluster strategy maps, Δ-bar charts**   | **RQ1**   |
| **[notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb](notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb)** | **C3 single-seed walkthrough — per-cluster filter/rerank choices**         | **RQ2**   |

---

## Paper ↔ code mapping

| Paper artefact                                      | Produced by                                                                                                             |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| §3 Dataset stats (Table — corpus stats)             | [notebooks/19_collection_explorer.ipynb](notebooks/19_collection_explorer.ipynb)                                        |
| §4.1 14 chunking strategies (Appendix A)            | [scripts/segmentation_strategies/](scripts/segmentation_strategies/)                                                    |
| §4.2 14 metadata strategies (Appendix E)            | [scripts/retrieval_utils.py](scripts/retrieval_utils.py) (rerank/filter helpers)                                         |
| **Table 1** — TC-Chunking / TC-Metadata / TC-Joint  | [scripts/run_C2_multiseed_kfold.py](scripts/run_C2_multiseed_kfold.py) · [scripts/run_C3_multiseed_kfold.py](scripts/run_C3_multiseed_kfold.py) · [scripts/run_C4_joint_multiseed.py](scripts/run_C4_joint_multiseed.py) |
| **Table 2** — Retrieval strategies / SOTA           | [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb) (S-RAG rows); SOTA rows from [notebooks/13_linearRAG_evaluation.ipynb](notebooks/13_linearRAG_evaluation.ipynb), [notebooks/14_hipporag_MH.ipynb](notebooks/14_hipporag_MH.ipynb) |
| §5 QRe classifier (training + 99.4 % accuracy)      | [notebooks/30_embed_questions.ipynb](notebooks/30_embed_questions.ipynb), [notebooks/31_classify_questions.ipynb](notebooks/31_classify_questions.ipynb) |
| §5 QRe + UMS routing functions                      | [scripts/retrieval_utils.py](scripts/retrieval_utils.py) (`apply_no_rerouting`, `apply_query_rerouting`, `apply_forced_equal_from_all_collections`, `apply_forced_multicollection`) |
| Appendix B — Clustering evaluation                  | [notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb](notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb) (UMAP + HDBSCAN sections) |
| Appendix C — Cluster qualitative findings           | [notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb](notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb), [notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb](notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb) |
| Appendix D — Classifier details                     | [notebooks/31_classify_questions.ipynb](notebooks/31_classify_questions.ipynb)                                          |
| Appendix G — Metrics definitions                    | [scripts/retrieval_utils.py](scripts/retrieval_utils.py) (`compute_recall_metrics_dataframe`, `compute_coverage_row_list`) |

---

## Limitations

- **Single year (1887).** Results may not transfer directly to the full Third Republic; the methodology, however, is corpus-agnostic.
- **French-only.** All embeddings, classifier features, and chunking heuristics are tuned for 19th-century French.
- **One historian-in-the-loop.** Question quality validation relies on a single domain expert; broader validation would strengthen external validity.
- **C4 winner's curse.** Joint optimisation over 196 pairs introduces estimation noise that can offset the theoretical benefit of jointly optimising both index-side axes.
- **Cohere embedding dependency.** The QRe classifier and clustering are calibrated for `embed-v4.0`; switching providers requires re-training (the OpenRouter path is implemented but not benchmarked in the paper).

---

## Citation

If you find ORDER useful in your work, please cite (BibTeX placeholder — will be updated on acceptance):

```bibtex
@inproceedings{order2026,
  title     = {ORDER: Task-Conditioned Routing for Retrieval-Augmented Generation},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

And the companion dataset (citation withheld for double-blind review):

```bibtex
@unpublished{historiqa2026anonymous,
  title  = {HistoriQA-ThirdRepublic: Multi-Hop Question Answering Corpus for Historical Research, Parliamentary Debates from the French Third Republic},
  author = {Anonymous},
  note   = {Accepted at LREC-COLING 2026; citation details withheld for double-blind review},
  year   = {2026}
}
```

---

## License & acknowledgements

- **Anonymity.** This repository is anonymised for EMNLP 2026 double-blind review. Identifying URLs, author names, and the companion-dataset preprint reference have been withheld and will be restored in the camera-ready version.
- **Code** — released for review purposes; final license will accompany the camera-ready version.
- **Dataset** — HistoriQA-ThirdRepublic is released by its authors under **CC-BY 4.0**.
- **Historical sources** — public domain, courtesy of the Bibliothèque nationale de France ([Gallica](https://gallica.bnf.fr/)).
- **SOTA baselines** — [HippoRAG v2](scripts/SOTA/HippoRAGv2-main/) and [LinearRAG](scripts/SOTA/LinearRAG-main/) are reused under their respective upstream licenses; see each subfolder.
- **Embedding provider** — Cohere `embed-v4.0` via the Cohere API.

ORDER is research code accompanying an EMNLP 2026 submission. Bug reports and reproduction issues are welcome via the repository issue tracker.