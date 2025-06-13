import random


class MinHash:
    def __init__(self, k):
        self.k = k
        self.hash_functions = [random.randint(0, 1000000) for _ in range(k)]

    def compute_signature(self, set_elements):
        signature = []
        for seed in self.hash_functions:
            min_hash = float('inf')
            for elem in set_elements:
                h = hash((elem, seed))
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def similarity(self, sig1, sig2):
        if len(sig1) != len(sig2):
            raise ValueError("Сигнатуры должны иметь одинаковую длину")
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.k
