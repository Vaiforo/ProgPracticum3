import pandas as pd
import matplotlib.pyplot as plt

from quotient_filter import QuotientFilter


def false_positive_rate(qf, train_words, test_words):
    false_positives = 0
    for word in test_words:
        if qf.query(word) and word not in train_words:
            false_positives += 1
    return (false_positives / len(test_words)) * 100


with open('english_words_10k.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]
train_words = words[:5000]
test_words = words[5000:]

q_values = [4, 6, 8, 10, 12]
r_values = [4, 6, 8, 10, 12]

results = {}

for q in q_values:
    for r in r_values:
        qf = QuotientFilter(q, r)
        for word in train_words:
            qf.insert(word)
        fp_rate = false_positive_rate(qf, train_words, test_words)
        results[(q, r)] = fp_rate

df = pd.DataFrame.from_dict(results, orient='index', columns=['False positive rate (%)'])
df.index = pd.MultiIndex.from_tuples(df.index, names=['q', 'r'])
print("Таблица результатов:")
print(df)

fixed_r = 8
fp_rates_q = [results[(q, fixed_r)] for q in q_values]
plt.figure(figsize=(8, 6))
plt.plot(q_values, fp_rates_q, marker='o')
plt.title(f'Зависимость FP rate от q (r={fixed_r})')
plt.xlabel('q (количество бит для частного)')
plt.ylabel('Процент FP (%)')
plt.grid(True)
plt.savefig('fp_rate_vs_q.png')

fixed_q = 8
fp_rates_r = [results[(fixed_q, r)] for r in r_values]
plt.figure(figsize=(8, 6))
plt.plot(r_values, fp_rates_r, marker='o')
plt.title(f'Зависимость FP rate от r (q={fixed_q})')
plt.xlabel('r (количество бит для остатка)')
plt.ylabel('Процент FP (%)')
plt.grid(True)
plt.savefig('fp_rate_vs_r.png')
