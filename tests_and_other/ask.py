import textwrap
from rag_core import RAGSystem


def main():
    rag = RAGSystem()
    rag.build_index()

    while True:
        question = input("\nAsk a question, or type 'exit': ")

        if question.lower().strip() in ["exit", "quit", "q"]:
            break

        answer, retrieved = rag.answer_question(question)

        print("\nANSWER:")
        print(textwrap.fill(answer, width=100))

        print("\nRETRIEVED CHUNKS:")
        for item in retrieved:
            print(f"\nChunk {item['chunk_id']} | Score: {item['score']:.4f}")
            print(item["text"])


if __name__ == "__main__":
    main()