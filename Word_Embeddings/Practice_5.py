# Odd-one-out and a simple "A minus B" query with Word2Vec
from gensim.models import Word2Vec

tokens = [['sprint', 'is', 'a', 'fast', 'run'], ['jog', 'is', 'a', 'slow', 'run'], ['swim', 'builds', 'full', 'body', 'strength'], ['row', 'builds', 'full', 'body', 'strength'], ['squat', 'trains', 'the', 'legs'], ['lunge', 'trains', 'the', 'legs'], ['plank', 'trains', 'the', 'core'], ['crunch', 'trains', 'the', 'core'], ['sprint', 'needs', 'a', 'warm', 'up'], ['jog', 'needs', 'a', 'warm', 'up']]

model = Word2Vec(tokens, vector_size=15, window=3, min_count=1, workers=1, epochs=150, seed=3)

group = ["sprint", "jog", "lunge"]
print("Odd one out in", group, "->", model.wv.doesnt_match(group))
print("Positive='sprint', negative='lunge':", [(w, round(s, 3)) for w, s in model.wv.most_similar(positive=["sprint"], negative=["lunge"], topn=3)])
