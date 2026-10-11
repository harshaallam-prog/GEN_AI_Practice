import os
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

first_text = """Principles of Athletic Training

Progressive overload means gradually increasing the demand placed on the body so that muscles, bones, and the
heart keep adapting. Adding a little more weight, distance, or speed each week is the classic method.

Specificity says that the body adapts to the exact kind of work it practises. A sprinter therefore trains
differently from a marathon runner, even though both want to become faster.

Warming up raises muscle temperature and prepares the joints for movement. A short dynamic routine before
training lowers the chance of strains and improves performance.

Rest days are part of the programme, not a break from it. Muscles grow stronger during recovery, so training
every day without rest often leads to fatigue and injury.

Tracking sessions in a simple log shows progress over time. Even a few notes about distance, weight, and how
you felt can reveal useful patterns.
"""
second_text = """Sports Nutrition

Carbohydrates fuel high intensity exercise, so endurance athletes often eat extra before long sessions. Protein
after training provides the building blocks that muscles need to repair themselves.

Hydration matters because even mild dehydration reduces strength and concentration. Drinking small amounts
regularly works better than gulping a large amount at once.
"""
third_text = """Recovery and Sleep

Sleep is when the body releases growth hormone and repairs tissue. Most athletes perform better when they sleep
seven to nine hours each night.

Gentle activities such as walking, stretching, or swimming speed up recovery by increasing blood flow without
adding much stress to tired muscles.
"""
sources = {"training_principles.txt": first_text, "sports_nutrition.txt": second_text, "recovery_and_sleep.txt": third_text}

os.makedirs("docs_fitness", exist_ok=True)
for name, text in sources.items():
    with open(os.path.join("docs_fitness", name), "w", encoding="utf-8") as f:
        f.write(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=60, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="fitness_chunks", embedding_function=embedder)

def add_in_batches(source, text, batch_size=3):
    chunks = splitter.split_text(text)
    for start in range(0, len(chunks), batch_size):
        part = chunks[start:start + batch_size]
        collection.add(
            documents=part,
            metadatas=[{"source": source, "chunk_index": start + j, "length": len(c)} for j, c in enumerate(part)],
            ids=[f"{source}#{start + j}" for j in range(len(part))],
        )
    print(f"{source}: {len(chunks)} chunks added in batches of {batch_size}")

for name, text in sources.items():
    add_in_batches(name, text)

print("\nPer-source statistics")
for name in sources:
    got = collection.get(where={"source": name}, include=["metadatas"])
    lengths = [m["length"] for m in got["metadatas"]]
    print(f"  {name:<28} chunks={len(lengths)} longest={max(lengths)} shortest={min(lengths)}")

res = collection.query(query_texts=["Why does sleep help athletes recover?"], n_results=2, include=["documents", "metadatas"])
for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    print("\n->", meta["source"], "#", meta["chunk_index"], "\n  ", doc[:100].replace("\n", " "))
