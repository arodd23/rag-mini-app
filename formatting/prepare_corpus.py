"""
prepare_corpus.py

Purpose:
1. Load the GovReport dataset from Hugging Face.
2. Combine train, validation, and test splits.
3. Remove empty or extremely short reports.
4. Create a reproducible 5,000-document experimental corpus.
5. Save the experimental corpus as JSON.
6. Optionally save the full GovReport corpus as JSON.

Run:
    python prepare_corpus.py
"""

import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "ccdv/govreport-summarization"
DATASET_CONFIG = "document"

EXPERIMENT_SIZE = 5_000
RANDOM_SEED = 42

# Prevent unusually tiny documents from entering the corpus.
MIN_DOCUMENT_CHARS = 5_000

OUTPUT_DIR = Path("data/dataset")

EXPERIMENT_OUTPUT = OUTPUT_DIR / "documents.json"
FULL_OUTPUT = OUTPUT_DIR / "govreport_full.json"

# Change to False if you do not want to save all ~19,500 reports.
SAVE_FULL_CORPUS = True


# ============================================================
# HELPERS
# ============================================================


def normalize_text(text: str) -> str:
    """
    Perform minimal text cleanup.

    We intentionally avoid aggressive preprocessing because the
    project later compares different chunking strategies.
    """

    if not isinstance(text, str):
        return ""

    return text.strip()


def save_json(documents: list[dict], output_path: Path) -> None:
    """Save documents as formatted UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_documents(dataset) -> list[dict]:
    """
    Convert GovReport rows into the format used by the RAG project.

    Output format:
    {
        "doc_id": 0,
        "text": "...",
        "source_split": "train",
        "source_id": "..."
    }
    """

    documents = []

    for row_index, row in enumerate(dataset):
        text = normalize_text(row["report"])

        if len(text) < MIN_DOCUMENT_CHARS:
            continue

        source_id = row.get("id")

        documents.append(
            {
                "doc_id": len(documents),
                "text": text,
                "source_id": source_id,
            }
        )

    return documents


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("=" * 60)
    print("PREPARING GOVREPORT CORPUS")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load GovReport
    # --------------------------------------------------------

    print("\n[1/6] Loading GovReport from Hugging Face...")

    dataset_dict = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    print("Dataset loaded.")

    for split_name, split in dataset_dict.items():
        print(f"  {split_name}: {len(split):,} reports")

    # --------------------------------------------------------
    # Step 2: Add original split metadata
    # --------------------------------------------------------

    print("\n[2/6] Preserving source split information...")

    datasets_with_split = []

    for split_name, split in dataset_dict.items():

        split = split.add_column(
            "source_split",
            [split_name] * len(split),
        )

        datasets_with_split.append(split)

    # --------------------------------------------------------
    # Step 3: Combine splits
    # --------------------------------------------------------

    print("\n[3/6] Combining dataset splits...")

    full_dataset = concatenate_datasets(datasets_with_split)

    print(
        f"Combined dataset size: "
        f"{len(full_dataset):,} reports"
    )

    # --------------------------------------------------------
    # Step 4: Filter short documents
    # --------------------------------------------------------

    print(
        f"\n[4/6] Filtering reports shorter than "
        f"{MIN_DOCUMENT_CHARS:,} characters..."
    )

    full_dataset = full_dataset.filter(
        lambda row: (
            isinstance(row["report"], str)
            and len(row["report"].strip()) >= MIN_DOCUMENT_CHARS
        )
    )

    print(
        f"Reports remaining: "
        f"{len(full_dataset):,}"
    )

    if len(full_dataset) < EXPERIMENT_SIZE:
        raise ValueError(
            f"Only {len(full_dataset):,} valid reports remain, "
            f"but EXPERIMENT_SIZE is {EXPERIMENT_SIZE:,}."
        )

    # --------------------------------------------------------
    # Step 5: Build full local corpus
    # --------------------------------------------------------

    print("\n[5/6] Converting reports to project format...")

    full_documents = []

    for row in full_dataset:

        text = normalize_text(row["report"])

        full_documents.append(
            {
                "doc_id": len(full_documents),
                "text": text,
                "source_id": row.get("id"),
                "source_split": row["source_split"],
            }
        )

    print(
        f"Created {len(full_documents):,} document records."
    )

    # --------------------------------------------------------
    # Step 6: Create fixed experimental corpus
    # --------------------------------------------------------

    print(
        f"\n[6/6] Selecting fixed "
        f"{EXPERIMENT_SIZE:,}-document corpus..."
    )

    shuffled_dataset = full_dataset.shuffle(
        seed=RANDOM_SEED
    )

    experiment_dataset = shuffled_dataset.select(
        range(EXPERIMENT_SIZE)
    )

    experiment_documents = []

    for row in experiment_dataset:

        text = normalize_text(row["report"])

        experiment_documents.append(
            {
                "doc_id": len(experiment_documents),
                "text": text,
                "source_id": row.get("id"),
                "source_split": row["source_split"],
            }
        )

    save_json(
        experiment_documents,
        EXPERIMENT_OUTPUT,
    )

    print(
        f"Experimental corpus saved to: "
        f"{EXPERIMENT_OUTPUT}"
    )

    # --------------------------------------------------------
    # Optional: save entire corpus
    # --------------------------------------------------------

    if SAVE_FULL_CORPUS:

        print("\nSaving full GovReport corpus...")

        save_json(
            full_documents,
            FULL_OUTPUT,
        )

        print(
            f"Full corpus saved to: "
            f"{FULL_OUTPUT}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    lengths = [
        len(document["text"])
        for document in experiment_documents
    ]

    average_chars = sum(lengths) / len(lengths)

    minimum_chars = min(lengths)
    maximum_chars = max(lengths)

    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)

    print(
        f"Experimental documents : "
        f"{len(experiment_documents):,}"
    )

    print(
        f"Average characters      : "
        f"{average_chars:,.0f}"
    )

    print(
        f"Shortest document       : "
        f"{minimum_chars:,}"
    )

    print(
        f"Longest document        : "
        f"{maximum_chars:,}"
    )

    print(
        f"Random seed             : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Experimental corpus     : "
        f"{EXPERIMENT_OUTPUT}"
    )

    if SAVE_FULL_CORPUS:
        print(
            f"Full corpus             : "
            f"{FULL_OUTPUT}"
        )

    print("\nCorpus preparation complete.")


if __name__ == "__main__":
    main()