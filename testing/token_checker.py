import json
from tiktoken import encoding_for_model

enc = encoding_for_model("gpt-3.5-turbo")
max_tokens = 4096

with open("/Users/yarik/Desktop/AromaKiss Project/LLM/training_data_unique.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        full_text = ""
        for m in data["messages"]:
            full_text += m["content"] + "\n"
        tokens = len(enc.encode(full_text))
        if tokens > max_tokens:
            print(f"Line {i + 1} has {tokens} tokens — TOO LONG")
        else:
            print(f"Line {i + 1} is OK: {tokens} tokens")
