import json
from pathlib import Path


metadata_path = Path("data/squad_metadata.jsonl")
sample_path = Path("data/squad_sample.json")

if not sample_path.exists():
    raise FileNotFoundError(
        "Missing data/squad_sample.json. Run python load_squad.py first."
    )

if not metadata_path.exists():
    raise FileNotFoundError(
        "Missing data/squad_metadata.jsonl. Run python load_squad.py first."
    )


MAX_QUESTIONS = 15

print(f"Showing first {MAX_QUESTIONS} questions from saved randomized sample.")
print(f"Sample file: {sample_path}")
print(f"Metadata file: {metadata_path}\n")

with metadata_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= MAX_QUESTIONS:
            break

        item = json.loads(line)

        question = item.get("question", "")
        answers = item.get("answers", [])
        title = item.get("title", "")
        doc_id = item.get("doc_id", "")
        is_impossible = item.get("is_impossible", False)

        print("=" * 80)
        print(f"Question {i + 1}")
        print(f"Title: {title}")
        print(f"Doc ID: {doc_id}")
        print(f"Impossible?: {is_impossible}")
        print(f"Question: {question}")

        if answers:
            print(f"Answer(s): {answers}")
        else:
            print("Answer(s): No answer / unanswerable")

print("\nCopy one of the answerable questions into ask.py or the Streamlit app.")