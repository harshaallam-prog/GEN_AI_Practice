import os
import re
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

file_path = "my_fitness_notes.txt"
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""Sports Nutrition

Carbohydrates fuel high intensity exercise, so endurance athletes often eat extra before long sessions. Protein
after training provides the building blocks that muscles need to repair themselves.

Hydration matters because even mild dehydration reduces strength and concentration. Drinking small amounts
regularly works better than gulping a large amount at once.

Recovery and Sleep

Sleep is when the body releases growth hormone and repairs tissue. Most athletes perform better when they sleep
seven to nine hours each night.

Gentle activities such as walking, stretching, or swimming speed up recovery by increasing blood flow without
adding much stress to tired muscles.""")

with open(file_path, "r", encoding="utf-8") as f:
    my_text = f.read()
print("Loaded", file_path, "->", len(my_text), "characters")

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    encode, decode = _enc.encode, _enc.decode
except Exception:                      # offline fallback: treat words as tokens
    encode = lambda t: t.split(" ")
    decode = lambda toks: " ".join(toks)

def fixed_chunks(text, size=200, overlap=30):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out

def sentence_chunks(text, n=1):
    s = [x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x]
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

def token_chunks(text, max_tokens=40, overlap=8):
    toks = encode(text)
    out, start = [], 0
    while start < len(toks):
        out.append(decode(toks[start:start + max_tokens]))
        start += max_tokens - overlap
    return out

strategies = {
    "fixed-size": fixed_chunks(my_text),
    "character": CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=30).split_text(my_text),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]).split_text(my_text),
    "sentence": sentence_chunks(my_text),
    "token": token_chunks(my_text),
}

biggest = max(len(c) for c in strategies.values())
for name, ch in strategies.items():
    print(f"{name:<11} | {'#' * int(30 * len(ch) / biggest):<30} | {len(ch)}")
