import pandas as pd
import matplotlib.pyplot as plt

from bloom_filter import BloomFilter


def false_positive_rate(m, k, train_words, test_words):
    bf = BloomFilter(m, k)
    for word in train_words:
        bf.add(word)
    false_positives = 0
    for word in test_words:
        if bf.check(word) and word not in train_words:
            false_positives += 1
    return (false_positives / len(test_words)) * 100


with open('english_words_10k.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]
train_words = words[:5000]
test_words = words[5000:]

m_values = [5000, 10000, 15000, 20000, 25000]
k_values = [1, 2, 3, 4, 5]

results = {}
for m in m_values:
    for k in k_values:
        fp_rate = false_positive_rate(m, k, train_words, test_words)
        results[(m, k)] = fp_rate

df = pd.DataFrame.from_dict(results, orient='index', columns=['False positive rate (%)'])
df.index = pd.MultiIndex.from_tuples(df.index, names=['m', 'k'])
print("Таблица результатов:")
print(df)

fixed_k = 1
fp_rates_m = [results[(m, fixed_k)] for m in m_values]
plt.figure(figsize=(8, 6))
plt.plot(m_values, fp_rates_m, marker='o')
plt.title(f'Зависимость ложноположительных срабатываний от m (k={fixed_k})')
plt.xlabel('m (размер битового массива)')
plt.ylabel('Процент ложноположительных срабатываний (%)')
plt.grid(True)
plt.show()

fixed_m = 5000
fp_rates_k = [results[(fixed_m, k)] for k in k_values]
plt.figure(figsize=(8, 6))
plt.plot(k_values, fp_rates_k, marker='o')
plt.title(f'Зависимость ложноположительных срабатываний от k (m={fixed_m})')
plt.xlabel('k (количество хеш-функций)')
plt.ylabel('Процент ложноположительных срабатываний (%)')
plt.grid(True)
plt.show()
