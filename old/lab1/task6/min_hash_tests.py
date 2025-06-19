import pandas as pd
import matplotlib.pyplot as plt

from min_hash import MinHash


def true_jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def error_rate(mh, set1, set2):
    sig1 = mh.compute_signature(set1)
    sig2 = mh.compute_signature(set2)
    estimated = mh.similarity(sig1, sig2)
    true = true_jaccard_similarity(set1, set2)
    return abs(true - estimated) / true * 100 if true > 0 else 0


with open('english_words_10k.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]

set1 = set(words[:4000])
set2 = set(words[3000:7000])

k_values = [10, 50, 100, 200, 500]

results = {}

for k in k_values:
    mh = MinHash(k)
    err = error_rate(mh, set1, set2)
    results[k] = err

df = pd.DataFrame.from_dict(results, orient='index',
                            columns=['Error Rate (%)'])
df.index.name = 'k'
print("Таблица результатов:")
print(df)

plt.figure(figsize=(8, 6))
plt.plot(k_values, [results[k] for k in k_values], marker='o')
plt.title('Зависимость ошибки от k')
plt.xlabel('k (количество хеш-функций)')
plt.ylabel('Относительная ошибка (%)')
plt.grid(True)
plt.show()
