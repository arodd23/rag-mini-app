from datasets import load_dataset
from pathlib import Path


# Load the dataset from Hugging Face
dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")

# Make sure data folder exists
Path("data").mkdir(exist_ok=True)

# Output file
output_path = Path("data/docs.txt")

# Use the train split
train_data = dataset["train_sft"]

# Limit for Phase 1 so it stays fast
MAX_CONVERSATIONS = 500

with output_path.open("w", encoding="utf-8") as f:
    for i, example in enumerate(train_data):
        if i >= MAX_CONVERSATIONS:
            break

        f.write(f"\n\n--- Conversation {i} ---\n")

        # Print the example once so we can inspect structure if needed
        if i == 0:
            print("First example:")
            print(example)

        # Most chat datasets store messages in a field like "messages"
        if "messages" in example:
            for message in example["messages"]:
                role = message.get("role", "unknown")
                content = message.get("content", "")

                f.write(f"{role.upper()}: {content}\n")

        else:
            # Fallback if the dataset has a different structure
            f.write(str(example))

print(f"Saved conversations to {output_path}")