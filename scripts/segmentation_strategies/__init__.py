"""segmentation_strategies package"""
from .strategies import (
    header_based,
    speaker_turn_atomic,
    speaker_turn_windowed,
    president_mediated,
    chapter_vote_split,
    anchor_expansion,
    semantic_texttiling,
    llm_thematic,
    strip_vote_lists,
    topic_boundary,
    amendment_cycle,
    hybrid_topic_window,
    STRATEGIES,
)
from .evaluate import (
    coverage_score,
    coverage_rate,
    size_distribution,
    aggregate_size_stats,
    retrieval_rank,
    recall_at_k,
    compare_strategies,
    load_records,
    load_sections_deduplicated,
    print_summary_table,
)
