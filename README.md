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

| Setting                         | Δ Coverage\@3 vs Naïve |
| ------------------------------- | ---------------------- |
| **C2** — TC-Chunking only       | **+5.4 pp**            |
| **C3** — TC-Metadata only       | **+10.8 pp**           |
| **QRe + UMS** (full routing)    | **+16.4 pp**           |

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
