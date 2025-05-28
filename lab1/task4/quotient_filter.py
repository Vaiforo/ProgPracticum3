import hashlib


class QuotientFilter:
    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.m = 2 ** q
        self.table = [None] * self.m

    def _hash(self, item):
        h = int(hashlib.sha256(item.encode()).hexdigest(), 16)
        quotient = h >> self.r
        remainder = h & ((1 << self.r) - 1)
        return quotient % self.m, remainder

    def insert(self, item):
        quotient, remainder = self._hash(item)
        self.table[quotient] = remainder

    def query(self, item):
        quotient, remainder = self._hash(item)
        return self.table[quotient] == remainder
