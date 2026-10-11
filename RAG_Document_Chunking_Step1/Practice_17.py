# 1 vs 3 sentences per chunk, comparing word counts
import re

article_text = """Progressive overload means gradually increasing the demand placed on the body so that muscles, bones, and the
heart keep adapting. Adding a little more weight, distance, or speed each week is the classic method.

Specificity says that the body adapts to the exact kind of work it practises. A sprinter therefore trains
differently from a marathon runner, even though both want to become faster.

Warming up raises muscle temperature and prepares the joints for movement. A short dynamic routine before
training lowers the chance of strains and improves performance.

Rest days are part of the programme, not a break from it. Muscles grow stronger during recovery, so training
every day without rest often leads to fatigue and injury.

Tracking sessions in a simple log shows progress over time. Even a few notes about distance, weight, and how
you felt can reveal useful patterns."""

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def chunk_by_sentences(text, n):
    s = split_sentences(text)
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

for n in (1, 3):
    chunks = chunk_by_sentences(article_text, n)
    words = [len(c.split()) for c in chunks]
    print(f"n={n}: {len(chunks)} chunks | words per chunk min/avg/max = {min(words)}/{sum(words)/len(words):.1f}/{max(words)}")
    print("   longest chunk:", max(chunks, key=len)[:80], "...")
