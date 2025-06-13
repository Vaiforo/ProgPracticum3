import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import List

from quotient_filter import QuotientFilterStruct


def false_positive_rate(qf, train_words, test_words):
    false_positives = 0
    for word in test_words:
        if qf.contains(word) and word not in train_words:
            false_positives += 1
    return (false_positives / len(test_words)) * 100


def test_fpr(q_bits: int, r_bits: int, items=1000, tests=5000) -> float:
    qf = QuotientFilterStruct(q_bits, r_bits)

    for i in range(min(items, (1 << q_bits) - 1)):
        qf.add(str(i))

    false_pos = 0
    for _ in range(tests):
        if qf.contains(str(random.randint(10 ** 6, 10 ** 7))):
            false_pos += 1

    return false_pos / tests


def solve(q_options: List[int], r_options: List[int]):
    results = []

    for q in q_options:
        for r in r_options:
            print(f"Тестируем q={q}, r={r}...")
            fpr = test_fpr(q, r)
            results.append({'q': q, 'r': r, 'FPR': fpr})

    df = pd.DataFrame(results)
    print("\nРезультаты тестов:")
    print(df)

    for q in q_options:
        subset = df[df['q'] == q]
        plt.plot(subset['r'], subset['FPR'], label=f'q={q}', marker='o')

    plt.xlabel('Биты остатка (r)')
    plt.ylabel('Вероятность ошибки')
    plt.title('Зависимость FPR от параметров')
    plt.legend()
    plt.grid()
    plt.show()


q_params = [8, 10, 12]
r_params = [4, 6, 8, 10]
solve(q_params, r_params)
