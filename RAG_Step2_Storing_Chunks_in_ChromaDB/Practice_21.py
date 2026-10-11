import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def validate(source, text):
    problems = []
    if not source.endswith(".txt"):
        problems.append("source name should end with .txt")
    if len(text.strip()) < 50:
        problems.append("text is too short to be useful")
    return problems

def ingest(collection, splitter, source, text, batch=4):
    issues = validate(source, text)
    if issues:
        print(f"Skipped {source!r}: {'; '.join(issues)}")
        return 0
    chunks = splitter.split_text(text)
    for s in range(0, len(chunks), batch):
        piece = chunks[s:s + batch]
        collection.add(documents=piece,
                       metadatas=[{"source": source, "chunk_index": s + k} for k in range(len(piece))],
                       ids=[f"{source}-{s + k}" for k in range(len(piece))])
    print(f"Ingested {source}: {len(chunks)} chunks")
    return len(chunks)

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(name="fitness_chunks", embedding_function=embedder)
splitter = RecursiveCharacterTextSplitter(chunk_size=230, chunk_overlap=25, separators=["\n\n", "\n", ". ", " ", ""])

doc_a = """Sports Nutrition

Carbohydrates fuel high intensity exercise, so endurance athletes often eat extra before long sessions. Protein
after training provides the building blocks that muscles need to repair themselves.

Hydration matters because even mild dehydration reduces strength and concentration. Drinking small amounts
regularly works better than gulping a large amount at once.
"""

doc_b = """Recovery and Sleep

Sleep is when the body releases growth hormone and repairs tissue. Most athletes perform better when they sleep
seven to nine hours each night.

Gentle activities such as walking, stretching, or swimming speed up recovery by increasing blood flow without
adding much stress to tired muscles.
"""

doc_c = """Equipment Register Notes

The adjustable dumbbell set is registered under equipment code EQ-4521 and is currently the most used item in
the gym.

The speed skipping rope is registered under equipment code EQ-7788 and is replaced within three business days
when worn.

Any maintenance request must quote the exact equipment code so the staff can find the right item quickly.
"""

ingest(collection, splitter, "sports_nutrition.txt", doc_a)
ingest(collection, splitter, "recovery_and_sleep.txt", doc_b)
ingest(collection, splitter, "equipment_register.txt", doc_c)
ingest(collection, splitter, "notes.md", doc_a)       # rejected: wrong extension
ingest(collection, splitter, "empty.txt", "too short")  # rejected: too short
print("Total chunks:", collection.count())
res = collection.query(query_texts=["How much sleep do athletes need?"], n_results=1, include=["metadatas"])
print("Best source for 'How much sleep do athletes need?':", res["metadatas"][0][0]["source"])
