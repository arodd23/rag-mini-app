import json
from pathlib import Path

from rag_core import RAGSystem


RAW_PATH = Path("data/eval_set_v0_raw.jsonl")
OUT_PATH = Path("data/eval_set_v0_raw_with_chunks.jsonl")


def main():
    rag = RAGSystem()
    raw_text = rag.load_documents()
    chunks = rag.chunk_text(raw_text)

    fixed = 0

    with RAW_PATH.open("r", encoding="utf-8") as in_file, OUT_PATH.open("w", encoding="utf-8") as out_file:
        for line in in_file:
            item = json.loads(line)
            chunk_id = item["gold_chunk_id"]

            item["chunk_text"] = chunks[chunk_id]

            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            fixed += 1

    print(f"Added chunk_text to {fixed} records.")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()