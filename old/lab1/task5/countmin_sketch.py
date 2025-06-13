import random


class CountMinSketch:
    def __init__(self, d, w):
        self.d = d
        self.w = w
        self.counters = [[0] * w for _ in range(d)]
        self.hashes = [random.randint(0, 1000000) for _ in range(d)]

    def _hash(self, item, seed):
        h = hash((item, seed))
        return h % self.w

    def add(self, item):
        for i in range(self.d):
            index = self._hash(item, self.hashes[i])
            self.counters[i][index] += 1

    def estimate(self, item):
        estimates = []
        for i in range(self.d):
            index = self._hash(item, self.hashes[i])
            estimates.append(self.counters[i][index])
        return min(estimates)
