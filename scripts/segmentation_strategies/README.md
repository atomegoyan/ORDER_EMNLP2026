# Segmentation Strategies for `doc_2_section`

> **Living document** — updated each time a new cluster of questions is analysed.
> Last update: 2026-03-19 (clusters 9 & 16).

## Purpose

`doc_2_section` is a full parliamentary-debate section from the *Journal Officiel*
(Third Republic, 1887). It can reach **170 k+ characters**. For multi-hop question
generation and RAG retrieval the section must be split into focused, retrievable
chunks. This package implements and evaluates multiple splitting strategies.

---

## Strategies

| ID | Name | Header | Description | Extra args |
|----|------|--------|-------------|------------|
| S1 | `header_based` | ALL-CAPS titles | Split on ≥10 uppercase-char headers. Usually yields **1 segment** (most sections have only one top-level header). | — |
| S2 | `speaker_turn_atomic` | `M. Name.` | One segment per speaker turn. Finest granularity (~600–950 chars median). Preamble before the first speaker is attached to that turn. | — |
| S3 | `speaker_turn_windowed` | `M. Name.` | Sliding window of *n* consecutive turns (overlap=1). Good precision/context trade-off. | `n_turns`, `overlap` |
| S4 | `president_mediated` | `M. le président.` | Group everything between two presidential interventions (official topic/speaker hand-offs). | — |
| S5 | `anchor_expansion` | fuzzy locate | Locate `doc_2` inside the section via exact or fuzzy match, then expand outward to speaker-turn boundaries. Always yields ≥1 segment containing the evidence. | `source_chunk`, `context_turns` |
| S6 | `semantic_texttiling` | embeddings | Cosine-similarity valley detection between sentence windows (TextTiling with dense embeddings). | `embed_fn`, `window_size`, `threshold_percentile` |
| S7 | `llm_thematic` | LLM | Ask an LLM to identify thematic boundaries (optional, slow). | `llm_fn`, `max_chars` |
| S8 | `chapter_vote_split` | `(Adopté.)` | Split on budget-chapter vote markers: `(Adopté.)`, `(Rejeté.)`, and OCR variants. One segment per chapter discussion + vote. | — |

### When to use which

| Scenario | Recommended |
|---|---|
| Maximise coverage, no extra APIs needed | **S2** or **S3 (n=3)** |
| Budget-debate rapid-vote sections | **S8** (chapter votes) + **S3** for the long inter-vote stretches |
| Precision retrieval (RAG gold label) | **S5** anchor expansion with `ctx=1` |
| Rich thematic segmentation | **S6** or **S7** (requires embeddings / LLM) |
| Quick whole-section baseline | **S1** |

---

## Clusters analysed

### Cluster 9 — Prison / penitentiary reform debates

| Aspect | Value |
|---|---|
| Questions | 22 valid |
| Unique sections | 1 (79 245 chars) |
| doc_2 lengths | 4 000 – 10 000 chars |
| doc_1_section | present |
| Structural markers | ~50 speaker turns, ~18 president turns, sparse chapter votes |
| Key OCR issues | `me8skur'`, merged words, garbled accents |

**Structural notes:**
- All questions reference the same large section (single budget topic).
- The section starts with an ALL-CAPS header followed immediately by
  `M. le président.` — 3 records had `doc_2[:150]` spanning that boundary.
  Fixed by merging the preamble into the first segment in S2/S4.

### Cluster 16 — Budget allocations: chaplains, secret funds, asylums

| Aspect | Value |
|---|---|
| Questions | 9 valid |
| Unique sections | 2 (170 930 chars, 75 292 chars) |
| doc_2 lengths | 5 356 – 8 226 chars |
| doc_1_section | **absent** (null) — only the newspaper clip is available |
| Structural markers | 61–69 speaker turns, 18–29 president turns, **60 "Adopté"** in section A, 35 chapter markers |
| Key OCR issues | `me8skur'`, `M. Maurice. Fauce.` (for Maurice-Faure), `ingufâ` |

**Structural notes:**
- Section A (170 k chars) is a massive multi-topic budget session covering
  chapters 6–33+, mixing rapid-vote blocks (`«Chap. N. — …, NNN fr.» — (Adopté.)`)
  with extended debates on chaplain suppression (Bourneville, Maurice-Faure).
- Section B (75 k chars) covers secret funds (M. Achard) and departmental elections.
- S8 (`chapter_vote_split`) is especially relevant here: it produces 37.6 segments
  per section, though highly bimodal (many short vote-confirmation lines ~105 chars,
  and a few large discussion segments up to 57 k chars).
- The evidence chunks always sit in the large discursive segments, not in the
  rapid-vote lines → S8 could be combined with S3 as a two-pass strategy.

---

## Evaluation results

All results below are **100 % coverage** (each `doc_2` found in at least one segment).

### Cluster 9

| Strategy | Avg segments | Median chars |
|---|---:|---:|
| S1 header | 1.0 | 79 245 |
| S2 atomic | 50.3 | 623 |
| S3 window-3 | 25.4 | 3 181 |
| S3 window-5 | 12.9 | 5 325 |
| S4 president | 12.2 | 1 492 |
| S5 anchor ctx=0 | 3.0 | 15 845 |
| S5 anchor ctx=1 | 3.0 | 18 750 |
| S8 chapter-vote | 16.0 | 126 |

### Cluster 16

| Strategy | Avg segments | Median chars |
|---|---:|---:|
| S1 header | 1.0 | 170 928 |
| S2 atomic | 61.9 | 955 |
| S3 window-3 | 31.4 | 5 525 |
| S3 window-5 | 16.2 | 9 404 |
| S4 president | 19.2 | 2 410 |
| S5 anchor ctx=0 | 3.0 | ~16 000 |
| S5 anchor ctx=1 | 3.0 | ~18 750 |
| S8 chapter-vote | 37.6 | 105 |

### Retrieval results (embeddings, cluster 9 — Cohere embed-v4.0)

See the notebook for the full recall@k bar charts. These results were obtained
with the embedding-based `compare_strategies` call in cell 25 of the notebook.

---

## Question categories observed

### Cross-cluster patterns

All questions are **multi-hop**: they require information from both a newspaper
article (`doc_1`) and a parliamentary transcript (`doc_2`).

| Category | Description | Clusters seen |
|---|---|---|
| **Argument extraction** | "What argument does M. X use to support Y?" | 9, 16 |
| **Position comparison** | "How does M. X distinguish A from B?" | 9, 16 |
| **Press reaction** | "How does the press comment on X?" / "What irony does the press note?" | 16 |
| **Legislative outcome** | "What is the result of the vote on X?" | 9, 16 |
| **Reform context** | "What broader reform does M. X invoke?" | 16 |
| **Press tone/critique** | "What tone does the press adopt?" / "What criticism does the press address?" | 16 |

### Cluster-specific themes

**Cluster 9** — Focused on penitentiary / prison administration reform.
Questions ask about inspector general roles, budget allocations for prison
services, and specific legislative proposals.

**Cluster 16** — Focused on the 1887 budget debate across three topics:
1. Suppression of chaplain subsidies in departmental prisons (Maurice-Faure)
2. Suppression of chaplains in asylums, specifically Charenton (Bourneville)
3. Reduction and control of secret funds for the Ministry of Interior (Achard)

Questions emphasise the **press–parliament dynamic**: what arguments are made
in the Chamber, how the press reports them, and what editorial stance the
press takes (irony, criticism, tone).

---

## Known issues / edge cases

1. **Preamble–speaker boundary**: when `doc_2` starts with an ALL-CAPS header
   immediately followed by the first `M. le président.`, the evidence straddles
   the split boundary. Fixed in S2 and S4 by extending the first segment back
   to offset 0.

2. **S8 bimodal distribution**: many very short segments (vote lines) and a few
   very large ones. A two-pass approach (S8 → S3 on large segments) is needed
   for uniform granularity.

3. **OCR speaker-name mangling**: `Maurice-Faure` → `Maurice. Fauce.`, which
   splits the hyphenated surname into what looks like two separate speaker turns.
   Current regex is broad enough to handle this, but it produces a false match.

4. **`doc_1_section` absent in cluster 16**: newspaper article sections are null,
   so cross-document section alignment is not possible for this cluster.

---

## File inventory

| File | Description |
|---|---|
| `strategies.py` | All 8 strategies + `STRATEGIES` registry |
| `evaluate.py` | Coverage, size stats, retrieval rank, `compare_strategies` |
| `__init__.py` | Package exports |
| `segmentation_exploration.ipynb` | Interactive notebook for all evaluations |
| `README.md` | This document |
