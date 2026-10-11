# A small reusable SimilarityChecker class with caching
from gensim.models import Word2Vec

class SimilarityChecker:
    def __init__(self, sentences):
        self.model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1, seed=9, epochs=100)
        self._cache = {}

    def score(self, a, b):
        key = tuple(sorted((a, b)))
        if key not in self._cache:
            self._cache[key] = float(self.model.wv.similarity(a, b))
        return self._cache[key]

    def closest(self, word, n=3):
        return [(w, round(s, 3)) for w, s in self.model.wv.most_similar(word, topn=n)]

checker = SimilarityChecker([['sprint', 'is', 'a', 'fast', 'run'], ['jog', 'is', 'a', 'slow', 'run'], ['swim', 'builds', 'full', 'body', 'strength'], ['row', 'builds', 'full', 'body', 'strength'], ['squat', 'trains', 'the', 'legs'], ['lunge', 'trains', 'the', 'legs'], ['plank', 'trains', 'the', 'core'], ['crunch', 'trains', 'the', 'core'], ['sprint', 'needs', 'a', 'warm', 'up'], ['jog', 'needs', 'a', 'warm', 'up']])
for a, b in [('sprint', 'jog'), ('swim', 'row'), ('squat', 'lunge')]:
    print(a, "<->", b, round(checker.score(a, b), 3))
print("Closest to 'sprint':", checker.closest("sprint"))
print("Cache entries:", len(checker._cache))
