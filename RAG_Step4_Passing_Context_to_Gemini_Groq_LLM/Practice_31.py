from Practice_28 import rag_answer

TLDR_FORMAT = (
    "Answer using ONLY the provided context. If the answer is missing, reply exactly: "
    "\"I don't know based on the provided documents.\" "
    "Format: first line 'TL;DR: <max 12 words>', then exactly 2 bullets starting with '* '. Nothing else."
)

def check_format(text):
    lines = [l for l in text.splitlines() if l.strip()]
    return {"starts_with_tldr": bool(lines) and lines[0].startswith("TL;DR:"),
            "bullets": sum(l.strip().startswith("* ") for l in lines)}

for q in ['What is progressive overload?', 'How do training, nutrition and sleep work together for performance?']:
    reply = rag_answer(q, system_prompt=TLDR_FORMAT, show_sources=False)
    print(f"\nQ: {q}\n{reply}\nFormat check: {check_format(reply)}")
