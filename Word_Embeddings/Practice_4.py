# Build one-hot vectors by hand with NumPy and verify against scikit-learn
import numpy as np
from sklearn.preprocessing import OneHotEncoder

words = ['sprint', 'jog', 'swim', 'cycle', 'sprint', 'lift', 'swim', 'row']

vocab = sorted(set(words))
index = {w: i for i, w in enumerate(vocab)}

manual = np.zeros((len(words), len(vocab)))
for row, w in enumerate(words):
    manual[row, index[w]] = 1

sk = OneHotEncoder(sparse_output=False).fit_transform([[w] for w in words])

print("Vocabulary:", vocab)
print("Manual one-hot:\n", manual.astype(int))
print("Matches scikit-learn:", np.array_equal(manual, sk))
