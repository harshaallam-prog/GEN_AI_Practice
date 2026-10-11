import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = {
    "training_principles.txt": """Principles of Athletic Training

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
""",
    "sports_nutrition.txt": """Sports Nutrition

Carbohydrates fuel high intensity exercise, so endurance athletes often eat extra before long sessions. Protein
after training provides the building blocks that muscles need to repair themselves.

Hydration matters because even mild dehydration reduces strength and concentration. Drinking small amounts
regularly works better than gulping a large amount at once.
"""
}

splitter = RecursiveCharacterTextSplitter(chunk_size=280, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="fitness_chunks", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

def index_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
    )

for _source, _text in documents.items():
    index_document(_source, _text)
print("Total chunks stored:", collection.count())

class Retriever:
    def __init__(self, collection, threshold=0.25):
        self.col = collection
        self.threshold = threshold

    @staticmethod
    def similarity(distance):
        return 1 - distance

    def search(self, query, n=5, threshold=None, where=None, max_words=None):
        cutoff = self.threshold if threshold is None else threshold
        kwargs = dict(query_texts=[query], n_results=n, include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where
        res = self.col.query(**kwargs)

        hits, words = [], 0
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = self.similarity(dist)
            if sim < cutoff:
                continue
            if max_words is not None and words + len(doc.split()) > max_words:
                break
            words += len(doc.split())
            hits.append({"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"], "similarity": round(sim, 4)})
        return hits

retriever = Retriever(collection)

third_doc = """Recovery and Sleep

Sleep is when the body releases growth hormone and repairs tissue. Most athletes perform better when they sleep
seven to nine hours each night.

Gentle activities such as walking, stretching, or swimming speed up recovery by increasing blood flow without
adding much stress to tired muscles.
"""
new_source = "recovery_and_sleep.txt"
index_document(new_source, third_doc)
print("Total chunks now:", collection.count())

early = {"$and": [{"source": new_source}, {"chunk_index": {"$lt": 1}}]}
hits = retriever.search("How much sleep do athletes need?", n=5, threshold=-1.0, where=early)
print("Only the first chunk of the new document:")
for h in hits:
    print(f"   chunk {h['chunk_index']} sim={h['similarity']} | {h['text'][:80]!r}")

everything = retriever.search("How does gentle activity help recovery?", n=3, threshold=-1.0)
print("\nUnfiltered top 3:", [(h["source"], h["chunk_index"]) for h in everything])
