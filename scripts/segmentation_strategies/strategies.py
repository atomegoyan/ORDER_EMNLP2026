"""
Segmentation strategies for doc_2_section chunks.

Each strategy takes a `section` string (a large parliamentary debate section)
and returns a list of segment strings.

Strategies
----------
S1  header_based           – Split on ALL-CAPS section headers
S2  speaker_turn_atomic    – One segment per speaker turn  (M. Name.)
S3  speaker_turn_windowed  – Sliding window of N consecutive speaker turns
S4  president_mediated     – Group turns between two "M. le président" calls
S5  anchor_expansion       – Locate a source chunk inside the section, expand
                             to natural speaker-turn boundaries
S6  semantic_texttiling    – Cosine-similarity valley detection on sentences
S7  llm_thematic           – LLM-based thematic segmentation (optional/slow)
S9  topic_boundary         – Split on agenda-item / topic markers
S10 amendment_cycle        – One segment per amendment lifecycle
S11 hybrid_topic_window    – Two-pass: topic boundary then windowed turns
"""

from __future__ import annotations

import re
import textwrap
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Matches the start of a speaker turn, e.g. "M. Raynal." or "M. le président."
# Also catches abbreviated names like "M. de Mahy." and un-accented OCR variants.
_SPEAKER_RE = re.compile(
    r"""
    (?:^|\n)                    # start of string or new line
    (M\.                        # "M."  (monsieur)
    (?:me\.?)?                  # optional "me" -> "Mme"
    \s+
    (?:le\s+président|la\s+présidente  # president variants
    |de\s+[A-ZÉÈÀÊÔÎÏÜa-zéèàêôûîïü\-']+  # de / la / etc.
    |[A-ZÉÈÀÊÔ][A-Za-zÉÈÀÊÔÎÏÜéèàêôûîïü\-'\s]+?
    )
    \.?)                        # closing dot (may be missing in OCR)
    \s                          # followed by whitespace
    """,
    re.VERBOSE | re.MULTILINE,
)

# Matches ALL-CAPS section headers (≥10 uppercase chars) typical in J.O. debates
_HEADER_RE = re.compile(
    r"""
    (?:^|\n\n?)                 # section break
    (
      [A-ZÉÈÀÊÔÎÏÜ\s\d\-\.\—\:\'«»]{10,}  # all-caps block
    )
    \n                          # followed by newline
    """,
    re.VERBOSE,
)

# President speaking → marks a logical topic boundary
_PRESIDENT_RE = re.compile(
    r"(?:^|\n)(M\.?\s+le\s+président\.?)",
    re.IGNORECASE,
)


def _split_on_pattern(text: str, pattern: re.Pattern) -> List[str]:
    """Split `text` at each match of `pattern`; keep leading match text."""
    parts = []
    prev = 0
    for m in pattern.finditer(text):
        start = m.start() if m.start() == 0 else m.start() + (1 if text[m.start()] == "\n" else 0)
        chunk = text[prev:m.start()].strip()
        if chunk:
            parts.append(chunk)
        prev = m.start() if text[m.start()] != "\n" else m.start() + 1
    tail = text[prev:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _find_speaker_boundaries(text: str) -> List[int]:
    """Return character offsets of every speaker-turn start in `text`."""
    offsets = []
    for m in _SPEAKER_RE.finditer(text):
        offsets.append(m.start() if m.start() == 0 else m.start() + 1)
    return offsets


# ---------------------------------------------------------------------------
# S1: Header-based segmentation
# ---------------------------------------------------------------------------

def header_based(section: str) -> List[str]:
    """
    Split on ALL-CAPS section headers (≥10 uppercase characters).
    These correspond to procedural sections like
       "SUITE DE LA DISCUSSION DU PROJET DE LOI..."
    Mirrors the logic in generate_chunks.chunking_strategy_1.
    """
    segments = re.split(
        r"\n(?=[A-ZÉÈÀÊÔÎÏÜ\s\d\-\.\—\:\'«»]{10,}\n)",
        section,
    )
    cleaned = [s.strip() for s in segments if s.strip()]
    return cleaned if cleaned else [section.strip()]


# ---------------------------------------------------------------------------
# S2: Atomic speaker-turn segmentation
# ---------------------------------------------------------------------------

def speaker_turn_atomic(section: str) -> List[str]:
    """
    One segment per speaker turn.  Boundaries are detected by the regex
    ``M. [Name].`` at the start of a line.  Produces the finest-grained
    segmentation, useful as a baseline.
    """
    # Normalise line endings
    text = section.replace("\r\n", "\n")

    # Find every speaker-turn boundary offset
    boundaries = _find_speaker_boundaries(text)
    if not boundaries:
        return [text.strip()]

    segments = []

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        # Extend the first turn back to 0 to absorb any all-caps header /
        # preamble so that we preserve original characters (no stitching).
        effective_start = 0 if i == 0 else start
        chunk = text[effective_start:end].strip()
        if chunk:
            segments.append(chunk)

    return segments


# ---------------------------------------------------------------------------
# S3: Windowed speaker-turn grouping
# ---------------------------------------------------------------------------

def speaker_turn_windowed(
    section: str,
    n_turns: int = 3,
    overlap: int = 1,
) -> List[str]:
    """
    Group `n_turns` consecutive speaker turns per segment, advancing by
    ``n_turns - overlap`` turns each step.  Provides context around each
    turn while staying focused.

    Parameters
    ----------
    n_turns : int
        Number of speaker turns per window (default 3).
    overlap : int
        Number of turns shared between consecutive windows (default 1).
    """
    turns = speaker_turn_atomic(section)
    if len(turns) <= n_turns:
        return ["\n\n".join(turns)]

    step = max(1, n_turns - overlap)
    segments = []
    i = 0
    while i < len(turns):
        window = turns[i : i + n_turns]
        segments.append("\n\n".join(window))
        i += step

    return segments


# ---------------------------------------------------------------------------
# S4: President-mediated grouping
# ---------------------------------------------------------------------------

def president_mediated(section: str) -> List[str]:
    """
    Each segment groups everything between two consecutive
    "M. le président" interventions.  The president's lines demarcate
    official topic/speaker hand-offs in the J.O. format.
    """
    text = section.replace("\r\n", "\n")

    # Positions of president lines
    pres_positions = [m.start() for m in _PRESIDENT_RE.finditer(text)]

    if not pres_positions:
        return [text.strip()]

    segments = []

    for i, start in enumerate(pres_positions):
        end = pres_positions[i + 1] if i + 1 < len(pres_positions) else len(text)
        # For the first president segment extend back to 0 to absorb any
        # all-caps header preamble — same logic as speaker_turn_atomic.
        effective_start = 0 if i == 0 else start
        chunk = text[effective_start:end].strip()
        if chunk:
            segments.append(chunk)

    return segments


# ---------------------------------------------------------------------------
# S5: Source-chunk anchor expansion
# ---------------------------------------------------------------------------

def _fuzzy_find(needle: str, haystack: str, max_error_rate: float = 0.05) -> Optional[Tuple[int, int]]:
    """
    Try exact substring match first; fall back to a sliding-window
    approximate match (Hamming-distance proxy) for OCR-noisy text.

    Returns (start, end) character offsets in `haystack`, or None if not found.
    """
    # 1. Exact match
    idx = haystack.find(needle)
    if idx != -1:
        return idx, idx + len(needle)

    # 2. Try with first 200 chars of needle (anchors on start)
    anchor = needle[:200].strip()
    idx = haystack.find(anchor)
    if idx != -1:
        return idx, min(idx + len(needle), len(haystack))

    # 3. Sliding window approximate match on first 100 chars
    short = needle[:100]
    n = len(short)
    best_pos, best_err = 0, n
    for i in range(max(0, len(haystack) - n)):
        window = haystack[i : i + n]
        errs = sum(c1 != c2 for c1, c2 in zip(short, window))
        if errs < best_err:
            best_err = errs
            best_pos = i
    if best_err / n <= max_error_rate:
        return best_pos, min(best_pos + len(needle), len(haystack))

    return None


def anchor_expansion(
    section: str,
    source_chunk: str,
    context_turns: int = 1,
) -> List[str]:
    """
    Locate `source_chunk` inside `section`, then expand the window by
    `context_turns` speaker turns on each side.

    This always produces at least one segment that contains the
    evidence chunk, making it the precision anchor for retrieval evaluation.

    Parameters
    ----------
    source_chunk : str
        The `doc_2` text that must appear in the returned segment.
    context_turns : int
        Number of additional speaker turns to include on each side (default 1).

    Returns
    -------
    List[str]
        A list containing the anchor segment (and surrounding context turns
        if context_turns > 0) plus any remaining parts of the section.
    """
    text = section.replace("\r\n", "\n")

    loc = _fuzzy_find(source_chunk, text)
    if loc is None:
        # Fallback: return the whole section as one segment
        return [text.strip()]

    src_start, src_end = loc

    # Get all speaker-turn boundaries
    boundaries = _find_speaker_boundaries(text)
    if not boundaries:
        return [text.strip()]

    # Find which turns overlap with [src_start, src_end]
    turn_ranges: List[Tuple[int, int]] = []
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        turn_ranges.append((b, end))

    overlapping = [
        i for i, (ts, te) in enumerate(turn_ranges)
        if ts < src_end and te > src_start
    ]

    if not overlapping:
        return [text.strip()]

    # Expand by context_turns on each side
    first_turn = max(0, overlapping[0] - context_turns)
    last_turn = min(len(turn_ranges) - 1, overlapping[-1] + context_turns)

    anchor_start = turn_ranges[first_turn][0]
    anchor_end = turn_ranges[last_turn][1]

    # Ensure the anchor segment actually starts where doc_2 starts
    # (doc_2 might begin with a header that precedes the first speaker turn).
    effective_start = min(anchor_start, src_start)
    anchor_seg = text[effective_start:anchor_end].strip()

    # Also return the rest of the section as additional segments so callers
    # can compare against other strategies.
    segments = []
    before = text[:anchor_start].strip()
    after = text[anchor_end:].strip()
    if before:
        segments.append(before)
    segments.append(anchor_seg)
    if after:
        segments.append(after)

    return segments


# ---------------------------------------------------------------------------
# S8: Chapter-vote boundary segmentation
# ---------------------------------------------------------------------------

# Matches budget-chapter vote results in J.O. format:
#   «Chap. 21. — ..., 480,600 fr. » — (Adopté.) or (Rejeté.)
# Also handles OCR variants (comma instead of period, verbose form).
_CHAPTER_VOTE_RE = re.compile(
    r'\([^\n)]{0,100}'
    r'(?:adopt\u00e9e?|adoptee?|rejete?\u00e9e?|rejet\u00e9e?'
    r'|mis\s+aux\s+voix[^\n)]{0,40}adopt\u00e9e?)'
    r'[^\n)]{0,30}\)',
    re.IGNORECASE,
)


def chapter_vote_split(section: str) -> List[str]:
    """
    Split on budget-chapter vote markers: ``(Adopté.)`` / ``(Rejeté.)``.

    In *Journal Officiel* budget debates (Troisième République format)
    each chapter ends with one of:

    * ``—\u00a0(Adopt\u00e9.)``   — standard form
    * ``(Adopt\u00e9,)``           — OCR comma variant
    * ``(Le chapitre N est mis aux voix et adopt\u00e9)``

    Each segment produced here covers one chapter's discussion + its
    closing vote.  Sections that contain no vote markers are returned
    as a single segment.

    This complements S3/S4 well for rapid budget-vote sessions where
    many chapters are adopted in quick succession, and for long sections
    that mix discursive debates with interspersed chapter votes.
    """
    text = section.replace("\r\n", "\n")

    vote_ends = [m.end() for m in _CHAPTER_VOTE_RE.finditer(text)]

    if not vote_ends:
        return [text.strip()]

    segments: List[str] = []
    prev = 0
    for end in vote_ends:
        chunk = text[prev:end].strip()
        if chunk:
            segments.append(chunk)
        prev = end

    # Trailing discussion after the last vote (no vote marker yet)
    tail = text[prev:].strip()
    if tail:
        segments.append(tail)

    return [s for s in segments if s] or [text.strip()]


# ---------------------------------------------------------------------------
# Vote-list stripping preprocessor
# ---------------------------------------------------------------------------

# Matches vote name-lists: "ONT VOTÉ POUR :", "ONT VOTÉ CONTRE :",
# "N'ONT PAS PRIS PART AU VOTE", "ABSENTS PAR CONGÉ", etc.
# These blocks contain hundreds of deputy names and are noise for retrieval.
_VOTE_LIST_HEADER_RE = re.compile(
    r"(ONT\s+VOT[EÉ]\s+(?:POUR|CONTRE)"
    r"|N.ONT\s+PAS\s+PRIS\s+PART\s+AU\s+VOTE"
    r"|ABSENTS?\s+PAR\s+CONG[EÉ])",
    re.IGNORECASE,
)


def strip_vote_lists(section: str) -> str:
    """
    Remove deputy name-lists from vote roll-calls.

    Blocks like ``ONT VOTÉ POUR : MM. Abeille. Achard. ...`` are replaced
    with a short placeholder ``[liste de vote omise]`` so that the
    semantic signal (a vote occurred) is preserved without the noise of
    hundreds of names.
    """
    text = section.replace("\r\n", "\n")
    matches = list(_VOTE_LIST_HEADER_RE.finditer(text))
    if not matches:
        return text

    # Each vote-list block runs from its header until the next vote-list
    # header, a double newline followed by a non-name line, or end of text.
    result_parts: List[str] = []
    prev_end = 0

    for i, m in enumerate(matches):
        # Keep text before this vote list
        result_parts.append(text[prev_end:m.start()])
        # Determine where the list ends: next header or end of text
        if i + 1 < len(matches):
            block_end = matches[i + 1].start()
        else:
            block_end = len(text)
        result_parts.append("[liste de vote omise]")
        prev_end = block_end

    result_parts.append(text[prev_end:])
    return "".join(result_parts)


# ---------------------------------------------------------------------------
# S9: Topic-boundary / agenda-item segmentation
# ---------------------------------------------------------------------------

# Matches agenda-item boundaries in J.O. format:
#   "L'ordre du jour appelle..."
#   "La discussion (générale) est ouverte..."
#   "Je mets aux voix..." / "Je consulte la Chambre..."
#   "DÉPÔT D'UN PROJET DE LOI" / "DÉPÔT D'UN PROJET DE RÉSOLUTION"
#   "SUITE DE LA DISCUSSION..."
_TOPIC_BOUNDARY_RE = re.compile(
    r"(?:^|\n)"
    r"("
    r"(?:M\.?\s+le\s+président\.?\s+)?L.ordre\s+du\s+jour\s+appelle"
    r"|(?:M\.?\s+le\s+président\.?\s+)?La\s+discussion(?:\s+g[eé]n[eé]rale)?\s+est\s+ouverte"
    r"|SUITE\s+DE\s+LA\s+DISCUSSION"
    r"|D[EÉ]P[OÔ]T\s+D.UN\s+PROJET\s+DE\s+(?:LOI|R[EÉ]SOLUTION)"
    r"|(?:M\.?\s+le\s+président\.?\s+)?Je\s+(?:mets\s+aux\s+voix|consulte\s+la\s+Chambre)"
    r"|FIXATION\s+DE\s+L.ORDRE\s+DU\s+JOUR"
    r"|D[EÉ]P[OÔ]T\s+DE\s+(?:PROPOSITIONS|RAPPORTS)"
    r")",
    re.IGNORECASE,
)


def topic_boundary(section: str) -> List[str]:
    """
    Split on agenda-item / topic markers in J.O. format.

    Produces coarser segments than ``president_mediated`` (S4) because
    only *topic-level* transitions are used as boundaries, not every
    presidential intervention.  Falls back to S4 when no topic markers
    are found.
    """
    text = section.replace("\r\n", "\n")

    offsets = []
    for m in _TOPIC_BOUNDARY_RE.finditer(text):
        start = m.start()
        if start > 0 and text[start] == "\n":
            start += 1
        offsets.append(start)

    if not offsets:
        return president_mediated(text)

    # Deduplicate and sort
    offsets = sorted(set(offsets))

    segments: List[str] = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        # Extend first segment back to 0 to absorb any header/preamble
        effective_start = 0 if i == 0 else start
        chunk = text[effective_start:end].strip()
        if chunk:
            segments.append(chunk)

    # Absorb any text before the first topic marker
    if offsets and offsets[0] > 0 and segments:
        preamble = text[:offsets[0]].strip()
        if preamble:
            segments[0] = preamble + "\n\n" + segments[0]

    # If only 1 segment was produced (single-topic section), fall back
    # to president_mediated for finer granularity.
    if len(segments) <= 1:
        return president_mediated(text)

    return segments


# ---------------------------------------------------------------------------
# S10: Amendment-cycle segmentation
# ---------------------------------------------------------------------------

# Matches amendment introductions
_AMENDMENT_START_RE = re.compile(
    r"(?:^|\n)"
    r"("
    r"(?:L.)?[Aa]mendement\s+de\s+M"
    r"|Celui\s+de\s+M\.\s+"
    r"|Il\s+y\s+a\s+(?:sur\s+ce\s+chapitre\s+)?(?:un|deux|trois|quatre|plusieurs)\s+amendement"
    r")",
    re.IGNORECASE,
)

# Matches amendment resolution (adoption or rejection)
_AMENDMENT_END_RE = re.compile(
    r"("
    r"La\s+Chambre(?:\s+des\s+d[eé]put[eé]s)?\s+(?:a\s+)?adopt[eé]"
    r"|La\s+Chambre(?:\s+des\s+d[eé]put[eé]s)?\s+(?:n.a\s+pas|a)\s+(?:adopt[eé]|rejet[eé])"
    r"|\([^\n)]{0,60}(?:adopt[eé]|rejet[eé])[^\n)]{0,30}\)"
    r"|est\s+(?:adopt[eé]|rejet[eé])"
    r")",
    re.IGNORECASE,
)


def amendment_cycle(section: str) -> List[str]:
    """
    One segment per amendment lifecycle: presentation → debate → vote.

    Detects amendment introductions (``amendement de M. [Name]``,
    ``La parole est à M. [Name]``) and their resolution (``(Adopté.)``,
    ``La Chambre a adopté``).  Each segment covers the full cycle.

    Falls back to ``chapter_vote_split`` (S8) when no amendment markers
    are found.
    """
    text = section.replace("\r\n", "\n")

    starts = [m.start() + (1 if m.start() > 0 and text[m.start()] == "\n" else 0)
              for m in _AMENDMENT_START_RE.finditer(text)]

    if not starts:
        return chapter_vote_split(text)

    # Find all amendment-end positions
    ends = [m.end() for m in _AMENDMENT_END_RE.finditer(text)]

    segments: List[str] = []
    prev = 0

    for i, s in enumerate(starts):
        # Emit any text before this amendment as its own segment
        if s > prev:
            pre = text[prev:s].strip()
            if pre:
                segments.append(pre)

        # Find the first end that comes after this start
        next_start = starts[i + 1] if i + 1 < len(starts) else len(text)
        matched_end = None
        for e in ends:
            if e > s and e <= next_start:
                matched_end = e
                # Take the last end before next_start (complete the cycle)

        if matched_end:
            chunk = text[s:matched_end].strip()
            prev = matched_end
        else:
            # No closing vote found — extend to next amendment or end
            chunk = text[s:next_start].strip()
            prev = next_start

        if chunk:
            segments.append(chunk)

    # Trailing text after last amendment
    tail = text[prev:].strip()
    if tail:
        segments.append(tail)

    return segments if segments else [text.strip()]


# ---------------------------------------------------------------------------
# S11: Hybrid topic → windowed-turns segmentation
# ---------------------------------------------------------------------------

def hybrid_topic_window(
    section: str,
    n_turns: int = 3,
    overlap: int = 1,
    max_segment_chars: int = 3000,
) -> List[str]:
    """
    Two-pass hierarchical segmentation:

    1. Split by agenda-item / topic boundaries (S9 logic).
    2. Within each topic segment, apply ``speaker_turn_windowed`` if the
       segment exceeds *max_segment_chars*.

    Produces medium-sized, topic-coherent segments with speaker-level
    granularity.
    """
    topic_segments = topic_boundary(section)

    result: List[str] = []
    for seg in topic_segments:
        if len(seg) > max_segment_chars:
            sub_chunks = speaker_turn_windowed(seg, n_turns=n_turns, overlap=overlap)
            result.extend(sub_chunks)
        else:
            result.append(seg)

    return result if result else [section.strip()]


# ---------------------------------------------------------------------------
# S6: Semantic TextTiling (embedding-based)
# ---------------------------------------------------------------------------

def semantic_texttiling(
    section: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    window_size: int = 3,
    threshold_percentile: float = 25,
    min_segment_chars: int = 200,
) -> List[str]:
    """
    Detect thematic boundaries using cosine-similarity valleys between
    consecutive sentence windows (TextTiling-style, but with dense embeddings).

    Parameters
    ----------
    embed_fn : callable
        Function that takes a list of strings and returns a list of float
        vectors (shape [n, d]).  Compatible with Cohere, OpenAI, or any
        sentence-transformers model.
    window_size : int
        Number of sentences to average into each side of the comparison
        window.
    threshold_percentile : float
        Percentile of similarity scores below which a boundary is declared.
    min_segment_chars : int
        Minimum segment length; segments shorter than this are merged into
        the previous one.

    Returns
    -------
    List[str]
    """
    import numpy as np

    # Sentence-level splitting (simple heuristic; keeps OCR-noise tolerance)
    sentences = re.split(r"(?<=[.!?»])\s+|\n{2,}", section)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2 * window_size + 1:
        return [section.strip()]

    vectors = embed_fn(sentences)
    vectors = np.array(vectors, dtype=np.float32)
    # L2-normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors = vectors / norms

    # Compute similarity between left and right windows at each gap
    n = len(sentences)
    similarities = []
    for i in range(window_size, n - window_size):
        left = vectors[i - window_size : i].mean(axis=0)
        right = vectors[i : i + window_size].mean(axis=0)
        sim = float(np.dot(left, right))
        similarities.append((i, sim))

    if not similarities:
        return [section.strip()]

    sim_values = [s for _, s in similarities]
    threshold = float(np.percentile(sim_values, threshold_percentile))

    boundary_positions = [i for i, s in similarities if s <= threshold]

    # Convert sentence indices to character offsets
    # Rebuild character offsets per sentence
    char_offsets = []
    pos = 0
    for sent in sentences:
        idx = section.find(sent, pos)
        char_offsets.append(idx if idx != -1 else pos)
        pos = char_offsets[-1] + len(sent)

    segments = []
    prev_char = 0
    for sent_idx in boundary_positions:
        char_end = char_offsets[sent_idx]
        chunk = section[prev_char:char_end].strip()
        if len(chunk) >= min_segment_chars:
            segments.append(chunk)
            prev_char = char_end
    # Last segment
    tail = section[prev_char:].strip()
    if tail:
        segments.append(tail)

    return [s for s in segments if s] or [section.strip()]


# ---------------------------------------------------------------------------
# S7: LLM-assisted thematic segmentation (optional)
# ---------------------------------------------------------------------------

_LLM_SEGMENT_PROMPT = """Voici un extrait de débat parlementaire de la Troisième République française (1887).
Identifie les frontières thématiques principales dans ce texte : chaque fois que le sujet de discussion change significativement (ex. passage d'un thème législatif à un autre, changement de ministre interrogé, nouveau chapitre budgétaire), indique le numéro de caractère approximatif dans le texte où commence ce nouveau segment, et donne lui un titre court.

Réponds UNIQUEMENT avec un JSON de la forme :
[
  {{"start_char": 0, "title": "Introduction / contexte"}},
  {{"start_char": 1234, "title": "Discussion sur les ports"}},
  ...
]

Ne génère aucun autre texte. Texte à segmenter (tronqué à 8000 caractères) :

{text}
"""


def llm_thematic(
    section: str,
    llm_fn: Callable[[str], str],
    max_chars: int = 8000,
) -> List[str]:
    """
    Ask an LLM to identify thematic boundaries and segment accordingly.

    Parameters
    ----------
    llm_fn : callable
        Function that takes a prompt string and returns the LLM response
        string (plain text or JSON).
    max_chars : int
        Character limit passed to the LLM (the full section may be longer).

    Returns
    -------
    List[str]
    """
    import json

    truncated = section[:max_chars]
    prompt = _LLM_SEGMENT_PROMPT.format(text=truncated)
    response = llm_fn(prompt)

    try:
        boundaries = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        # Attempt to extract JSON array from freeform response
        m = re.search(r"\[.*\]", response, re.DOTALL)
        if m:
            try:
                boundaries = json.loads(m.group())
            except json.JSONDecodeError:
                return [section.strip()]
        else:
            return [section.strip()]

    if not boundaries or not isinstance(boundaries, list):
        return [section.strip()]

    starts = sorted(
        [int(b.get("start_char", 0)) for b in boundaries if isinstance(b, dict)]
    )

    segments = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(section)
        chunk = section[start:end].strip()
        if chunk:
            segments.append(chunk)

    return segments if segments else [section.strip()]


# ---------------------------------------------------------------------------
# Registry: map strategy name → callable
# ---------------------------------------------------------------------------

STRATEGIES = {
    "S1_header_based": header_based,
    "S2_speaker_atomic": speaker_turn_atomic,
    "S3_speaker_window_3": lambda s: speaker_turn_windowed(s, n_turns=3, overlap=1),
    "S3_speaker_window_5": lambda s: speaker_turn_windowed(s, n_turns=5, overlap=1),
    "S4_president_mediated": president_mediated,
    "S8_chapter_vote": chapter_vote_split,
    "S9_topic_boundary": topic_boundary,
    "S10_amendment_cycle": amendment_cycle,
    "S11_hybrid_topic_window": hybrid_topic_window,
    # Vote-stripped variants: compose strip_vote_lists with base strategy
    "S9_topic_noVotes": lambda s: topic_boundary(strip_vote_lists(s)),
    "S11_hybrid_noVotes": lambda s: hybrid_topic_window(strip_vote_lists(s)),
    # S5 and S6 require extra args; use their functions directly.
    # S7 requires an LLM fn; use llm_thematic() directly.
}
