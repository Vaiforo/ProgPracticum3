import math
import hashlib


class HyperLogLog:
    def __init__(self, b):
        self.b = b
        self.m = 2 ** b
        self.registers = [0] * self.m
        self.alpha = self.get_alpha(self.m)

    @staticmethod
    def get_alpha(m):
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    @staticmethod
    def hash64(s):
        h = hashlib.sha256(s.encode()).digest()
        h_int = int.from_bytes(h[:8], 'big')
        return h_int

    def add(self, element):
        h = self.hash64(element)
        index = h & (self.m - 1)
        h_shifted = h >> self.b
        bin_h_shifted = bin(h_shifted)[2:].zfill(64 - self.b)
        if bin_h_shifted == '0' * (64 - self.b):
            rho = 64 - self.b + 1
        else:
            rho = bin_h_shifted.index('1') + 1
        self.registers[index] = max(self.registers[index], rho)

    def estimate(self):
        Z = sum(2 ** (-self.registers[j]) for j in range(self.m))
        estimate = self.alpha * self.m ** 2 / Z
        V = sum(1 for j in range(self.m) if self.registers[j] == 0)
        if V > 0:
            estimate_linear = self.m * math.log(self.m / V)
            if estimate < 2.5 * self.m:
                estimate = estimate_linear
        return estimate
