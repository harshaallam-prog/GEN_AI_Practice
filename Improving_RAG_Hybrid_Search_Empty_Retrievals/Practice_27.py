import re
from rank_bm25 import BM25Okapi
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
""",
    "equipment_register.txt": """Equipment Register Notes

The adjustable dumbbell set is registered under equipment code EQ-4521 and is currently the most used item in
the gym.

The speed skipping rope is registered under equipment code EQ-7788 and is replaced within three business days
when worn.

Any maintenance request must quote the exact equipment code so the staff can find the right item quickly.
""",
}

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "what", "whats", "which",
             "who", "how", "why", "when", "where", "can", "could", "would", "should", "you", "your", "me", "my", "i",
             "we", "our", "it", "its", "this", "that", "of", "for", "to", "in", "on", "at", "by", "with", "about",
             "and", "or", "please", "tell", "s"}

class HybridIndex:
    def __init__(self, name, chunk_size=210, overlap=35):
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.col = chromadb.Client().create_collection(name=name, embedding_function=embedder,
                                                       metadata={"hnsw:space": "cosine"})
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
        self.texts, self.meta, self.bm25 = [], [], None

    @staticmethod
    def tokens(text, drop=False):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in toks if not (drop and t in STOPWORDS)]

    def add(self, source, text):
        chunks = self.splitter.split_text(text)
        meta = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.col.add(documents=chunks, metadatas=meta, ids=[f"{source}|{i}" for i in range(len(chunks))])
        self.texts += chunks
        self.meta += meta
        self.bm25 = BM25Okapi([self.tokens(t) for t in self.texts])
        return len(chunks)

    def semantic(self, query, n=5):
        r = self.col.query(query_texts=[query], n_results=min(n, self.col.count()),
                           include=["documents", "metadatas", "distances"])
        return [{"text": t, "source": m["source"], "similarity": round(1 - x, 4)}
                for t, m, x in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def keyword(self, query, n=5, drop=False):
        q = self.tokens(query, drop)
        if not q:
            return []
        sc = self.bm25.get_scores(q)
        idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:n]
        return [{"text": self.texts[i], "source": self.meta[i]["source"], "bm25": round(float(sc[i]), 4)}
                for i in idx if sc[i] > 0]

    def hybrid(self, query, n=3, k=60, floor=-1.0, drop=False):
        scores, info = {}, {}
        sem = [r for r in self.semantic(query, 10) if r["similarity"] >= floor]
        for ranking in (sem, self.keyword(query, 10, drop)):
            for pos, r in enumerate(ranking):
                scores[r["text"]] = scores.get(r["text"], 0) + 1 / (k + pos + 1)
                info.setdefault(r["text"], r)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"text": t, "source": info[t]["source"], "rrf": round(s, 5)} for t, s in top]

    def smart(self, query, n=4, threshold=0.5):
        hits = [r for r in self.semantic(query, n) if r["similarity"] >= threshold]
        if hits:
            return {"status": "ok", "method": "semantic", "chunks": hits}
        hits = self.hybrid(query, n)
        if hits:
            return {"status": "ok", "method": "hybrid_fallback", "chunks": hits}
        return {"status": "empty", "method": None, "chunks": [],
                "message": "I don't know based on the provided documents. Covered files: "
                           + ", ".join(sorted({m["source"] for m in self.meta}))}

index = HybridIndex("fitness_chunks_hybrid")
for _src, _txt in documents.items():
    index.add(_src, _txt)
print("Total chunks stored:", index.col.count())

def smart_v2(query, n=4, threshold=0.5, floor=0.1):
    first = index.smart(query, n, threshold)
    if first["status"] == "ok" and first["method"] == "semantic":
        return first
    hits = index.hybrid(query, n, floor=floor)
    if hits:
        return {"status": "ok", "method": "hybrid", "chunks": hits}
    stripped = " ".join(index.tokens(query, drop=True))
    if stripped:
        hits = index.hybrid(stripped, n, floor=floor, drop=True)
        if hits:
            return {"status": "ok", "method": "hybrid_stripped", "chunks": hits, "query_used": stripped}
    return {"status": "empty", "method": None, "chunks": [],
            "message": "I don't know based on the provided documents."}

for q in ['What is progressive overload?', 'Tell me about EQ-4521', 'Could you please tell me what the EQ-7788 item is about?', 'How do I reset my email password?']:
    out = smart_v2(q)
    print(f"{q[:48]:<50} -> {out['status']:<6} {out['method']}", out.get("query_used", ""))

print("\nTokens before stopword removal:", index.tokens("Could you please tell me what the EQ-7788 item is about?"))
print("Tokens after stopword removal :", index.tokens("Could you please tell me what the EQ-7788 item is about?", drop=True))
