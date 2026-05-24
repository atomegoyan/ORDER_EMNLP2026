# ORDER: Task-Conditioned Routing for RAG

> Code & experiments for the EMNLP 2026 submission **“ORDER: Task-Conditioned
> Routing for Retrieval-Augmented Generation.”**
> Companion dataset (**HistoriQA-ThirdRepublic**):
> [project page](https://atomegoyan.github.io/historiqa-thirdrepublic/) ·
> [GitHub](https://github.com/atomegoyan/historiqa-thirdrepublic)

ORDER is a routing layer placed between the query and a heterogeneous,
multi-source RAG index. Instead of treating every query the same way, it
embeds each question, clusters the training queries in that embedding space,
and learns **per-cluster** retrieval decisions — which chunking granularity to
hit (**C2 — TC-Chunking**), which source/metadata filter to apply
(**C3 — TC-Metadata**), or both jointly (**C4 — Joint C×M**) — together with
a supervised query router (**QRe**) and a uniform multi-source allocator
(**UMS**) for unseen queries at test time.

On the 1887 slice of *HistoriQA-ThirdRepublic* (Les Débats parlementaires,
Le Gaulois, L’Intransigeant), the headline gains over a naïve dense
retriever are:

| Setting                                    | Δ Coverage\@3 vs Naïve |
| ------------------------------------------ | ---------------------- |
| **C2** — TC-Chunking only                  | **+5.4 pp**            |
| **C3** — TC-Metadata only                  | **+8.9 pp** (Naïve) · **+6.1 pp** under UMS |
| **C4** — Joint chunking × metadata         | **+8.8 pp**            |
| **QRe + UMS** (multi-hop, Recall\@3)       | **+16.4 pp** vs Naïve · **+20.6 pp** vs HippoRAG v2 |

The QRe classifier reduces to a binary press-vs-parliamentary decision
and reaches **99.4 % accuracy** on held-out queries; QRe and UMS
attack structurally distinct failure modes (source *contamination*
vs.\ source *starvation*) and compose strictly.

---

## Research questions

The paper is organised around five questions; this repo provides one
runnable artefact per RQ:

| RQ      | Question                                                                                              | Reproduced by                                                                                                                              |
| ------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **RQ1** | Does query-conditioned **chunking** beat a global best?                                               | [scripts/run_C2_multiseed_kfold.py](scripts/run_C2_multiseed_kfold.py)                                                                     |
| **RQ2** | Does query-conditioned **metadata indexing** on *Les Débats* beat a uniform global baseline?          | [scripts/run_C3_multiseed_kfold.py](scripts/run_C3_multiseed_kfold.py)                                                                     |
| **RQ3** | Does **joint** optimisation of chunking × metadata beat either single axis?                           | [scripts/run_C4_joint_multiseed.py](scripts/run_C4_joint_multiseed.py) (+ [scripts/c4_joint/](scripts/c4_joint/))                          |
| **RQ4** | Does query-conditioned **source selection** beat a single-index baseline?                             | [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb)                         |
| **RQ5** | Are **QRe** (router) and **UMS** (allocator) complementary remedies for size-imbalanced retrieval?    | [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb)                         |

---

## Results from the paper

### Task-conditioned indexing (RQ1–RQ3)

Coverage@k / Recall@k on the held-out test set (*n* = 438, 50/50 split,
5 seeds). Each retriever is compared to its Global Baseline
(`10000_hierarchical`, no metadata) across three task-conditioned
variants. \* = *p* < 0.05 (Bonferroni-paired *t*-test); ns = not significant.

| Retriever        | Indexing strategy   | Cov\@3      | Cov\@5      | Rec\@3      | Rec\@5      |
| ---------------- | ------------------- | ----------- | ----------- | ----------- | ----------- |
| **Naïve**        | Global Baseline     | 36.9        | 45.4        | 36.9        | 45.6        |
|                  | + TC-Chunking       | 42.3\*      | 51.7\*      | 42.3\*      | 51.7\*      |
|                  | + TC-Metadata       | **45.8\***  | **55.2\***  | **45.9\***  | **55.3\***  |
|                  | + TC-Joint          | _45.7_\*    | _54.7_\*    | _45.5_\*\*  | _54.5_\*    |
| **QRe**          | Global Baseline     | 48.5        | 58.2        | 48.7        | 58.4        |
|                  | + TC-Chunking       | 48.5 ns     | 57.9 ns     | 48.5 ns     | 58.0 ns     |
|                  | + TC-Metadata       | 48.5 ns     | 58.2 ns     | 48.7 ns     | 58.4 ns     |
|                  | + TC-Joint          | 48.3 ns     | 57.6 ns     | 48.2 ns     | 57.4 ns     |
| **UMS**          | Global Baseline     | 46.8        | 58.7        | 47.0        | 58.8        |
|                  | + TC-Chunking       | 47.0 ns     | 58.5 ns     | 47.2 ns     | 58.7 ns     |
|                  | + TC-Metadata       | **52.8\***  | **63.2\***  | **53.0\***  | **63.4\***  |
|                  | + TC-Joint          | _52.8_\*    | _63.1_\*    | _53.0_\*    | _63.4_\*    |
| **QRe & UMS**    | Global Baseline     | 54.7        | 64.5        | 54.9        | 64.7        |
|                  | + TC-Chunking       | 54.7 ns     | 64.5 ns     | 54.9 ns     | 64.7 ns     |
|                  | + TC-Metadata       | 54.7 ns     | 64.5 ns     | 54.9 ns     | 64.7 ns     |
|                  | + TC-Joint          | 54.7 ns     | 64.5 ns     | 54.9 ns     | 64.7 ns     |

*Bold = best per metric within the retriever block; italics = second-best.*

**Read this table this way:**
- Under the Naïve Retriever, all three task-conditioned variants help
  significantly (+5.4 pp / +8.9 pp / +8.8 pp Cov@3 respectively).
- TC-Metadata is the only variant that **persists** under a non-trivial
  retriever — it still gains +6.0 pp Cov@3 under UMS, where parliamentary
  chunks occupy a guaranteed retrieval quota and filtering noisy
  document types directly improves that quota.
- TC-Chunking gains vanish once QRe is active: chunking and routing
  fix the *same* failure mode (source contamination), so the gains do
  not stack.
- The joint variant matches but does not strictly dominate TC-Metadata
  — a winner’s-curse effect from the 14 × 14 = 196-pair search space.

### Retrieval strategies on multi-hop questions (RQ4–RQ5)

Recall@k on the multi-hop subset of HistoriQA-ThirdRepublic, split by
question type — *cross-newspaper* (314 q., both gold passages in
newspapers) and *newspaper → Débats* (571 q., one newspaper + one
parliamentary document).

| Configuration                          | Overall R\@3 | R\@5     | R\@10    | Cross-news. R\@3 | News.→Débats R\@3 |
| -------------------------------------- | ------------ | -------- | -------- | ---------------- | ----------------- |
| Naïve Retriever                        | 35.9         | 44.0     | 54.4     | 14.3             | 47.8              |
| BM25 (lexical)                         | 23.1         | 29.3     | 37.8     | 4.6              | 33.3              |
| HippoRAG v2                            | 31.7         | 41.9     | 53.0     | 11.6             | 42.7              |
| LinearRAG                              | 31.7         | 41.9     | 53.0     | 11.6             | 42.7              |
| **S-RAG: QRe**                         | 48.0         | 57.0     | 68.4     | 48.4             | 47.8              |
| **S-RAG: UMS**                         | _50.2_       | _57.6_   | _69.4_   | 40.8             | 55.4              |
| **S-RAG: QRe & UMS** (ours, full)      | **52.3**     | **58.4** | **72.3** | 46.7             | 55.4              |

QRe lifts retrieval on cross-newspaper questions (where parliamentary
contamination is the bottleneck) from 14.3 → 48.4 Recall@3, while UMS
lifts newspaper→Débats questions (where the press is starved by the
much larger parliamentary index) from 47.8 → 55.4 Recall@3. The
combined router strictly composes both gains.

---

## Key contributions and where they live

| Paper                          | Idea                                                                                                     | Code entry point                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| §4.1 **C2 — TC-Chunking**      | Cluster queries in embedding space; pick the best chunking strategy per cluster (14 candidates).         | [scripts/run_C2_multiseed_kfold.py](scripts/run_C2_multiseed_kfold.py)                                        |
| §4.2 **C3 — TC-Metadata**      | Same clusters; pick the best per-cluster metadata filter / source mix.                                   | [scripts/run_C3_multiseed_kfold.py](scripts/run_C3_multiseed_kfold.py)                                        |
| §4.3 **C4 — Joint C × M**      | Joint optimisation over (chunking strategy × metadata policy) per cluster, cached scoring for tractability. | [scripts/run_C4_joint_multiseed.py](scripts/run_C4_joint_multiseed.py) · [scripts/c4_joint/](scripts/c4_joint/) |
| §5 **QRe** (Query Rerouting)   | Supervised classifier that maps a held-out query to a training cluster.                                  | [scripts/retrieval_utils.py](scripts/retrieval_utils.py) (`apply_query_rerouting`)                            |
| §5 **UMS** (Uniform Multi-Src) | Fallback allocator that spreads top-k uniformly across sources when the router is uncertain.             | [scripts/retrieval_utils.py](scripts/retrieval_utils.py) (`apply_forced_equal_from_all_collections`)          |

---

## Repository layout

```
ORDER_EMNLP2026/
├── scripts/
│   ├── config.py                          # paths + API keys (loads .env)
│   ├── chunk_debats.py                    # split Les Débats parlementaires into sections
│   ├── generate_all_chunk_variants.py     # 14 chunking strategies on Les Débats
│   ├── embed_chunk_variants.py            # embed each variant → ChromaDB
│   ├── build_indexes.py                   # Cohere/OpenRouter + BM25 indexes (3 sources)
│   ├── retrieval_utils.py                 # retrievers, routing functions (QRe, UMS), metrics
│   ├── run_C2_multiseed_kfold.py          # C2 multi-seed experiment
│   ├── run_C3_multiseed_kfold.py          # C3 multi-seed experiment
│   ├── run_C4_joint_multiseed.py          # C4 multi-seed experiment
│   ├── c4_joint/                          # C4 internals: cache → score → evaluate
│   │   ├── c4_1_build_cache.py
│   │   ├── c4_2_score_clusters.py
│   │   └── c4_3_evaluate_test.py
│   ├── segmentation_strategies/           # the 14 chunking strategy definitions (see its README)
│   ├── SOTA/
│   │   ├── HippoRAGv2-main/               # B1/B2 indexing scripts for HippoRAG v2 baseline
│   │   └── LinearRAG-main/                # LinearRAG baseline (its own requirements.txt)
│   └── utils/                             # ChromaDB / IO helpers
├── notebooks/
│   ├── 10_retriever_evaluation.ipynb      # dense retriever sanity check
│   ├── 11_bm25_retrieval_evaluation.ipynb # BM25 baseline
│   ├── 12_retrieval_evaluation_generic.ipynb
│   ├── 13_linearRAG_evaluation.ipynb
│   ├── 14_hipporag_MH.ipynb               # HippoRAG multi-hop eval
│   ├── 19_collection_explorer.ipynb
│   ├── 30_embed_questions.ipynb           # build question embeddings used by QRe
│   ├── 31_classify_questions.ipynb        # train / inspect the QRe classifier
│   ├── 32_retriever_evaluation_query_rerouting.ipynb
│   └── single_seed/                       # single-seed prototypes for C2 & C3
│       ├── 40_C2_task_conditioned_chunking.ipynb
│       └── 41_C3_task_conditioned_metadata.ipynb
├── paper_EMNLP/                           # LaTeX source
├── data/                                  # (gitignored: large; see Data section)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Methodology in brief

ORDER splits retrieval optimisation along two axes:

- **Offline, document-conditioned axis** — heavy lifting: build 14
  chunking variants of *Les Débats* (sliding windows, hierarchical
  splits, fixed-size, semantic chunkers; see
  [scripts/segmentation_strategies/](scripts/segmentation_strategies/)),
  embed each variant into a ChromaDB collection with Cohere
  `embed-v4.0`, and pre-compute per-cluster scores over the
  *(chunking × metadata)* grid.
- **Online, query-conditioned axis** — at inference: embed the query,
  assign it to a training cluster (UMAP-10D + HDBSCAN learned on the
  training split, *min_cluster_size = 5*, *min_samples = 5*), and
  retrieve from the index/metadata configuration the cluster prefers.
  The marginal online cost is one UMAP projection plus one
  nearest-centroid lookup.

**Experimental protocol.** All multi-seed scripts use the same five
seeds — `[42, 123, 2021, 7, 9999]` — and a 50/50 train/test split of the
HistoriQA-ThirdRepublic 1887 question set. Significance is reported
with a Bonferroni-corrected paired *t*-test across seeds. Routing
configurations (`apply_no_rerouting`, `apply_query_rerouting`,
`apply_forced_multicollection`,
`apply_forced_equal_from_all_collections`) live in
[scripts/retrieval_utils.py](scripts/retrieval_utils.py) and are the
single source of truth for QRe and UMS.

---

## Setup

The codebase targets Python ≥ 3.10. The author runs everything in the
`base` conda environment on Windows.

```powershell
conda activate base
pip install -r requirements.txt
```

Then create a `.env` file at the repo root by copying `.env.example`:

```text
COHERE_API_KEY=sk-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...     # optional, only for the OpenRouter embedding path
```

`scripts/config.py` reads these via `python-dotenv`. ChromaDB is pinned
to the 0.5.x line — collections written with later majors are not
backward-compatible.

---

## Data

The corpus and questions are released as part of
**HistoriQA-ThirdRepublic** (see links at the top). Download the 1887
slice and drop the archives under `data/`, so that the layout below
matches what `scripts/config.py` expects:

```
data/
├── corpus_1887/                # raw daily issues of Les Débats parlementaires
├── legaulois_1887/             # Le Gaulois daily issues
├── lintransigeant_1887/        # L'Intransigeant daily issues
└── questions/                  # train / dev / test question files (JSONL)
```

All other folders under `data/` (`corpus_1887_splitted_v2/`,
`embeddings_*/`, `bm25_*/`, `RETRIEVER_results_*/`, `multiseed_results/`,
…) are **derived artefacts** and are regenerated by the pipeline below.
They are intentionally gitignored.

---

## End-to-end reproduction

Run from the repository root with the `base` conda environment active.
Steps 1–4 are one-shot indexing; 5–7 are the contribution experiments.

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

# 5. C2 — Task-Conditioned Chunking (multi-seed)
python -m scripts.run_C2_multiseed_kfold

# 6. C3 — Task-Conditioned Metadata (multi-seed)
python -m scripts.run_C3_multiseed_kfold

# 7. C4 — Joint chunking × metadata
python -m scripts.c4_joint.c4_1_build_cache
python -m scripts.run_C4_joint_multiseed
```

**SOTA baselines** (optional, install extra deps from
`scripts/SOTA/LinearRAG-main/requirements.txt` and the HippoRAG v2
upstream first):

```powershell
python -m scripts.SOTA.HippoRAGv2-main.B1_hipporag_indexing
python scripts/SOTA/LinearRAG-main/run.py
```

**Figures & tables** are produced by the notebooks under
[notebooks/](notebooks/). The mapping below points each paper artefact
to the file that generates it.

---

## Paper ↔ code mapping

| Paper artefact                                      | Produced by                                                                                                             |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Table — Naïve / BM25 retriever baseline             | [notebooks/10_retriever_evaluation.ipynb](notebooks/10_retriever_evaluation.ipynb), [notebooks/11_bm25_retrieval_evaluation.ipynb](notebooks/11_bm25_retrieval_evaluation.ipynb) |
| Generic retrieval comparison                        | [notebooks/12_retrieval_evaluation_generic.ipynb](notebooks/12_retrieval_evaluation_generic.ipynb)                      |
| **C2** — TC-Chunking absolute + Δ tables            | [scripts/run_C2_multiseed_kfold.py](scripts/run_C2_multiseed_kfold.py) (single-seed walkthrough: [notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb](notebooks/single_seed/40_C2_task_conditioned_chunking.ipynb)) |
| **C3** — TC-Metadata absolute + Δ tables            | [scripts/run_C3_multiseed_kfold.py](scripts/run_C3_multiseed_kfold.py) (single-seed walkthrough: [notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb](notebooks/single_seed/41_C3_task_conditioned_metadata.ipynb)) |
| **C4** — Joint C × M tables                         | [scripts/run_C4_joint_multiseed.py](scripts/run_C4_joint_multiseed.py) (+ [scripts/c4_joint/](scripts/c4_joint/))       |
| QRe classifier — training & inspection              | [notebooks/30_embed_questions.ipynb](notebooks/30_embed_questions.ipynb), [notebooks/31_classify_questions.ipynb](notebooks/31_classify_questions.ipynb) |
| QRe routing — end-to-end retrieval impact           | [notebooks/32_retriever_evaluation_query_rerouting.ipynb](notebooks/32_retriever_evaluation_query_rerouting.ipynb)      |
| HippoRAG v2 baseline                                | [scripts/SOTA/HippoRAGv2-main/](scripts/SOTA/HippoRAGv2-main/), [notebooks/14_hipporag_MH.ipynb](notebooks/14_hipporag_MH.ipynb) |
| LinearRAG baseline                                  | [scripts/SOTA/LinearRAG-main/](scripts/SOTA/LinearRAG-main/), [notebooks/13_linearRAG_evaluation.ipynb](notebooks/13_linearRAG_evaluation.ipynb) |

The 14 chunking strategies themselves are documented in
[scripts/segmentation_strategies/](scripts/segmentation_strategies/).

---

## Citation

```bibtex
@inproceedings{order2026,
  title     = {ORDER: Task-Conditioned Routing for Retrieval-Augmented Generation},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```

*(BibTeX entry will be updated upon publication.)*

---

## License & acknowledgements

Code released for review purposes. Final license will accompany the
camera-ready version. ORDER builds on the
[HistoriQA-ThirdRepublic](https://github.com/atomegoyan/historiqa-thirdrepublic)
dataset and reuses HippoRAG v2 and LinearRAG as SOTA baselines; see the
respective subfolders under [scripts/SOTA/](scripts/SOTA/) for their
original licenses.
