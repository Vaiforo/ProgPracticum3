import pandas as pd
import matplotlib.pyplot as plt

from hyperloglog import HyperLogLog

with open('english_words_10k.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]
unique_words = list(set(words))
n_unique = len(unique_words)

b_values = [4, 6, 8, 10, 12]

results = {}

for b in b_values:
    hll = HyperLogLog(b)
    for word in words:
        hll.add(word)
    estimated = hll.estimate()
    error = abs(n_unique - estimated) / n_unique * 100
    results[b] = error

df = pd.DataFrame.from_dict(results, orient='index', columns=['Relative error (%)'])
df.index.name = 'b'
print("Таблица результатов:")
print(df)

plt.figure(figsize=(8, 6))
plt.plot(b_values, [results[b] for b in b_values], marker='o')
plt.title('Зависимость относительной ошибки от b')
plt.xlabel('b (количество бит для регистров)')
plt.ylabel('Относительная ошибка (%)')
plt.grid(True)
plt.show()
