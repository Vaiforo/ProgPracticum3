import random
import pandas as pd
import matplotlib.pyplot as plt

from bloom_with_counter import BloomFilterCounter


def false_positive_rate(cbf, train_words, test_words):
    false_positives = 0
    for word in test_words:
        if cbf.check(word) and word not in train_words:
            false_positives += 1
    return (false_positives / len(test_words)) * 100


with open('english_words_10k.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]
train_words = words[:5000]
test_words = words[5000:]

m_values = [1000, 2000, 3000, 4000, 5000]
k_values = [1, 2, 3, 4, 5]

results_add = {}
results_remove = {}

for m in m_values:
    for k in k_values:
        cbf = BloomFilterCounter(m, k)

        for word in train_words:
            cbf.add(word)
        fp_rate_add = false_positive_rate(cbf, train_words, test_words)
        results_add[(m, k)] = fp_rate_add

        remove_words = random.sample(test_words, 500)

        remove_words = [word for word in remove_words if word not in train_words]
        deleted_words = 0
        for word in remove_words:
            if cbf.remove(word):
                deleted_words += 1
        fp_rate_remove = deleted_words / len(remove_words) * 100
        results_remove[(m, k)] = fp_rate_remove

df_add = pd.DataFrame.from_dict(results_add, orient='index', columns=['FP rate after add (%)'])
df_add.index = pd.MultiIndex.from_tuples(df_add.index, names=['m', 'k'])
print("Результаты после добавления:")
print(df_add)

df_remove = pd.DataFrame.from_dict(results_remove, orient='index', columns=['FP rate after remove (%)'])
df_remove.index = pd.MultiIndex.from_tuples(df_remove.index, names=['m', 'k'])
print("\nРезультаты после удаления:")
print(df_remove)

fixed_k = 1
fp_rates_m_add = [results_add[(m, fixed_k)] for m in m_values]
plt.figure(figsize=(8, 6))
plt.plot(m_values, fp_rates_m_add, marker='o')
plt.title(f'Зависимость FP rate от m после добавления (k={fixed_k})')
plt.xlabel('m (размер массива)')
plt.ylabel('Процент FP (%)')
plt.grid(True)
plt.savefig('fp_rate_vs_m_add.png')

fixed_m = 5000
fp_rates_k_add = [results_add[(fixed_m, k)] for k in k_values]
plt.figure(figsize=(8, 6))
plt.plot(k_values, fp_rates_k_add, marker='o')
plt.title(f'Зависимость FP rate от k после добавления (m={fixed_m})')
plt.xlabel('k (количество хеш-функций)')
plt.ylabel('Процент FP (%)')
plt.grid(True)
plt.savefig('fp_rate_vs_k_add.png')

fp_rates_m_remove = [results_remove[(m, fixed_k)] for m in m_values]
plt.figure(figsize=(8, 6))
plt.plot(m_values, fp_rates_m_remove, marker='o')
plt.title(f'Зависимость FP rate от m после удаления (k={fixed_k})')
plt.xlabel('m (размер массива)')
plt.ylabel('Процент FP (%)')
plt.grid(True)
plt.savefig('fp_rate_vs_m_remove.png')

fp_rates_k_remove = [results_remove[(fixed_m, k)] for k in k_values]
plt.figure(figsize=(8, 6))
plt.plot(k_values, fp_rates_k_remove, marker='o')
plt.title(f'Зависимость FP rate от k после удаления (m={fixed_m})')
plt.xlabel('k (количество хеш-функций)')
plt.ylabel('Процент FP (%)')
plt.grid(True)
plt.savefig('fp_rate_vs_k_remove.png')
