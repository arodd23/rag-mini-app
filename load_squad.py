from datasets import load_dataset
from pathlib import Path
import json


# -----------------------------
# Settings
# -----------------------------
NUM_EXAMPLES = 100
RANDOM_SEED = 42

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# This is the ONE file you review
REVIEW_PATH = DATA_DIR / "squad_review.json"

# These are generated from the review file
DOCS_PATH = DATA_DIR / "docs.txt"
METADATA_PATH = DATA_DIR / "squad_metadata.jsonl"


def create_or_load_review_file():
    """
    Creates one stable randomized review file.
    If it already exists, reuse it.
    """

    if REVIEW_PATH.exists():
        print(f"Using existing review file: {REVIEW_PATH}")

        with REVIEW_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    print("Loading SQuAD v2 from Hugging Face...")
    dataset = load_dataset("rajpurkar/squad_v2", split="train")

    print(f"Shuffling once with seed={RANDOM_SEED} and selecting {NUM_EXAMPLES} examples...")
    sampled_dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(NUM_EXAMPLES))

    seen_contexts = {}
    review_records = []

    for sample_id, item in enumerate(sampled_dataset):
        title = item["title"]
        context = item["context"].strip()
        question = item["question"].strip()
        answers = item["answers"]["text"]
        is_impossible = len(answers) == 0

        # Same context can appear with multiple questions.
        # This keeps the same doc_id for duplicate contexts.
        if context not in seen_contexts:
            seen_contexts[context] = len(seen_contexts)

        doc_id = seen_contexts[context]

        review_records.append({
            "sample_id": sample_id,
            "doc_id": doc_id,
            "title": title,
            "context": context,
            "question": question,
            "answers": answers,
            "is_impossible": is_impossible
        })

    with REVIEW_PATH.open("w", encoding="utf-8") as f:
        json.dump(review_records, f, indent=2, ensure_ascii=False)

    print(f"Saved review file to {REVIEW_PATH}")

    return review_records


def write_docs_and_metadata(review_records):
    """
    Builds docs.txt and squad_metadata.jsonl from the one review file.
    """

    seen_doc_ids = set()

    with DOCS_PATH.open("w", encoding="utf-8") as docs_file, METADATA_PATH.open("w", encoding="utf-8") as meta_file:
        for item in review_records:
            doc_id = item["doc_id"]
            title = item["title"]
            context = item["context"]

            # Write each unique document context only once
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)

                docs_file.write(f"\n\n--- Document {doc_id}: {title} ---\n")
                docs_file.write(context)
                docs_file.write("\n")

            # Write every question/answer metadata item
            meta_record = {
                "sample_id": item["sample_id"],
                "doc_id": item["doc_id"],
                "title": item["title"],
                "question": item["question"],
                "answers": item["answers"],
                "is_impossible": item["is_impossible"]
            }

            meta_file.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

    print(f"Saved documents to {DOCS_PATH}")
    print(f"Saved metadata/questions to {METADATA_PATH}")


def main():
    review_records = create_or_load_review_file()
    write_docs_and_metadata(review_records)

    unique_docs = len(set(item["doc_id"] for item in review_records))
    total_questions = len(review_records)
    answerable = sum(not item["is_impossible"] for item in review_records)
    unanswerable = sum(item["is_impossible"] for item in review_records)

    print()
    print("Dataset summary:")
    print(f"Review file: {REVIEW_PATH}")
    print(f"Total examples/questions: {total_questions}")
    print(f"Unique documents/contexts: {unique_docs}")
    print(f"Answerable questions: {answerable}")
    print(f"Unanswerable questions: {unanswerable}")


if __name__ == "__main__":
    main()