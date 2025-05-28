class BloomFilter:
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.bit_array = [False] * m
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
            self.bit_array[index] = True

    def check(self, element):
        for prime in self.primes:
            index = self.hash_function(element, prime)
            if not self.bit_array[index]:
                return False
        return True

    def union(self, other):
        if self.m != other.m or self.k != other.k:
            raise ValueError("Для объединения m и k должны быть одинаковыми")
        new_bf = BloomFilter(self.m, self.k)
        for i in range(self.m):
            new_bf.bit_array[i] = self.bit_array[i] or other.bit_array[i]
        return new_bf

    def intersection(self, other):
        if self.m != other.m or self.k != other.k:
            raise ValueError("Для нахождения пересечения m и k должны быть одинаковыми")
        new_bf = BloomFilter(self.m, self.k)
        for i in range(self.m):
            new_bf.bit_array[i] = self.bit_array[i] and other.bit_array[i]
        return new_bf
