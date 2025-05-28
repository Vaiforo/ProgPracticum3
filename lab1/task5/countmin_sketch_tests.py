import pandas as pd
import matplotlib.pyplot as plt

from countmin_sketch import CountMinSketch


def error_rate(cms, train_words, test_words, threshold=1):
    errors = 0
    for word in test_words:
        estimate = cms.estimate(word)
        if estimate >= threshold and word not in train_words:
            errors += 1
    return (errors / len(test_words)) * 100


with open('english_words_10k.txt', 'r') as file:
    words = [line.strip() for line in file.readlines()]
train_words = words[:5000]
test_words = words[5000:]

d_values = [1, 2, 3, 4, 5]
w_values = [100, 500, 1000, 2000, 5000]

results = {}

for d in d_values:
    for w in w_values:
        cms = CountMinSketch(d, w)
        for word in train_words:
            cms.add(word)
        err_rate = error_rate(cms, train_words, test_words)
        results[(d, w)] = err_rate

df = pd.DataFrame.from_dict(results, orient='index', columns=['Error Rate (%)'])
df.index = pd.MultiIndex.from_tuples(df.index, names=['d', 'w'])
print("Таблица результатов:")
print(df)

fixed_w = 1000
err_rates_d = [results[(d, fixed_w)] for d in d_values]
plt.figure(figsize=(8, 6))
plt.plot(d_values, err_rates_d, marker='o')
plt.title(f'Зависимость Error Rate от d (w={fixed_w})')
plt.xlabel('d (количество хеш-функций)')
plt.ylabel('Процент ошибок (%)')
plt.grid(True)
plt.savefig('error_rate_vs_d.png')

fixed_d = 3
err_rates_w = [results[(fixed_d, w)] for w in w_values]
plt.figure(figsize=(8, 6))
plt.plot(w_values, err_rates_w, marker='o')
plt.title(f'Зависимость Error Rate от w (d={fixed_d})')
plt.xlabel('w (ширина массива)')
plt.ylabel('Процент ошибок (%)')
plt.grid(True)
plt.savefig('error_rate_vs_w.png')
