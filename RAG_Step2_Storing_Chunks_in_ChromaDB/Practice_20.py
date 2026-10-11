import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

article_text = """Principles of Athletic Training

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

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
import re
collection = client.create_collection(name="fitness_chunks", embedding_function=embedder)

def recursive_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=260, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]).split_text(text)

def sentence_chunks(text, n=2):
    s = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

for name, fn in (("recursive", recursive_chunks), ("sentence", sentence_chunks)):
    pieces = fn(article_text)
    collection.add(documents=pieces, ids=[f"{name}-{i}" for i in range(len(pieces))],
                   metadatas=[{"strategy": name} for _ in pieces])
    print(f"{name:<10}: {len(pieces)} chunks")

for q in ['What is progressive overload?', 'Why are rest days important?', 'How does protein help after training?']:
    print("\n" + q)
    for name in ("recursive", "sentence"):
        r = collection.query(query_texts=[q], n_results=1, where={"strategy": name}, include=["documents", "distances"])
        print(f"  {name:<10} {r['distances'][0][0]:.3f} {r['documents'][0][0][:75]!r}")
