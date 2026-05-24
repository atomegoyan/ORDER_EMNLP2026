"""
Utilities for loading and preparing multi-hop question datasets.
"""

import json
import pandas as pd


def create_questions_dataset(
    jsonl_path: str,
    column_mapping: dict,
    source1_name: str = "Source 1",
    source2_name: str = "Source 2",
    question_type: str = "Generic",
) -> pd.DataFrame:
    """Load a multi-hop question JSONL file and expand into one row per question.

    Args:
        jsonl_path: Path to the JSONL file.
        column_mapping: Dict mapping logical keys to actual column names in the file.
            Required keys: 'id_1', 'id_2', 'doc_1', 'doc_2'.
            Optional: 'metadata_1', 'metadata_2'.
        source1_name: Human-readable label for the first source.
        source2_name: Human-readable label for the second source.
        question_type: Type label (e.g. 'MH - Generic', 'MH - BridgeEntity').

    Returns:
        DataFrame with one row per question. IDs are normalised to str.
    """
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if isinstance(entry.get("questions"), str):
                entry["questions"] = json.loads(entry["questions"])
            entries.append(entry)

    df = pd.DataFrame(entries)
    question_rows = []

    for _, row in df.iterrows():
        base_row = row.drop("questions").to_dict()
        for col in column_mapping.values():
            base_row.pop(col, None)

        for q_idx, q in enumerate(row["questions"]["questions"]):
            question_row = base_row.copy()

            id_2_value = row[column_mapping["id_2"]]
            if isinstance(id_2_value, list):
                id_2_value = id_2_value[0] if id_2_value else None
            if id_2_value is None:
                continue

            question_row.update(
                {
                    "question_index": q_idx,
                    "question": q["question"],
                    "llm_answer": q.get("answer", ""),
                    "question_type": question_type,
                    "id_1": str(row[column_mapping["id_1"]]),
                    "id_2": str(id_2_value),
                    "source_1": source1_name,
                    "source_2": source2_name,
                    "doc_1": row.get(column_mapping["doc_1"], ""),
                    "doc_2": row.get(column_mapping["doc_2"], ""),
                    "metadata_1": row.get(column_mapping.get("metadata_1"), {}),
                    "metadata_2": row.get(column_mapping.get("metadata_2"), {}),
                    "gold_ids": [
                        str(row[column_mapping["id_1"]]),
                        str(id_2_value),
                    ],
                }
            )
            question_rows.append(question_row)

    questions_df = pd.DataFrame(question_rows)
    print(
        f"Created dataset with {len(questions_df)} questions "
        f"from {len(df)} original entries"
    )
    return questions_df
