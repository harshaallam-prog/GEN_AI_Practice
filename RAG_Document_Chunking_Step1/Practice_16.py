# Which sizes keep a whole paragraph together?
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

paragraphs = [p for p in article_text.split("\n\n") if p.strip()]
longest = max(len(p) for p in paragraphs)
print("Paragraphs:", len(paragraphs), "| longest paragraph:", longest, "chars")

for size in (150, 300, longest + 20):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = sp.split_text(article_text)
    split_paras = sum(1 for p in paragraphs if not any(p.strip() in c for c in chunks))
    print(f"chunk_size={size:<4} -> {len(chunks):<2} chunks, paragraphs broken across chunks: {split_paras}")
