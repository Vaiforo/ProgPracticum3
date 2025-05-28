class BloomFilterCounter:
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.counters = [0] * m
        self.primes = [31, 37, 41, 43, 47, 53, 59, 61, 67, 71][:k]
        if len(self.primes) < k:
            raise ValueError("Не хватает простых чисел")

    def hash_function(self, s, prime):
        h = 0
        for char in s:
            h = (h * prime + ord(char)) % self.m
        return h

    def add(self, element):
        for prime in self.primes:
            index = self.hash_function(element, prime)
            self.counters[index] += 1

    def remove(self, element):
        find_indexes = []
        for prime in self.primes:
            index = self.hash_function(element, prime)
            if self.counters[index] > 0:
                find_indexes.append(index)
        if len(find_indexes) == self.k:
            for index in find_indexes:
                if self.counters[index] > 0:
                    self.counters[index] -= 1
            return True
        return False


    def check(self, element):
        for prime in self.primes:
            index = self.hash_function(element, prime)
            if self.counters[index] == 0:
                return False
        return True

    def union(self, other):
        if self.m != other.m or self.k != other.k:
            raise ValueError("Фильтры должны иметь одинаковые m и k")
        new_filter = BloomFilterCounter(self.m, self.k)
        for i in range(self.m):
            new_filter.counters[i] = self.counters[i] + other.counters[i]
        return new_filter

    def intersection(self, other):
        if self.m != other.m or self.k != other.k:
            raise ValueError("Фильтры должны иметь одинаковые m и k")
        new_filter = BloomFilterCounter(self.m, self.k)
        for i in range(self.m):
            new_filter.counters[i] = min(self.counters[i], other.counters[i])
        return new_filter
