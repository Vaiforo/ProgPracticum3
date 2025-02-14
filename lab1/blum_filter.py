class BlumFilter:
    def __init__(self, size: int):
        self.size: int = size
        self.filter: list = [0 for _ in range(size)]

    def add(self, value: str) -> None:
        hashes = self.get_hashes(value)

        self.filter[hashes[0] % self.size] += 1
        self.filter[hashes[1] % self.size] += 1
        self.filter[hashes[2] % self.size] += 1

    def check(self, value: str, hashes: (int, int, int) = None) -> bool:
        if not hashes:
            hashes = self.get_hashes(value)

        return all((self.filter[hashes[0] % self.size],
                    self.filter[hashes[1] % self.size],
                    self.filter[hashes[2] % self.size]))

    def delete(self, value: str) -> None:
        hashes = self.get_hashes(value)

        if self.check(value, hashes):
            self.filter[hashes[0] % self.size] -= 1
            self.filter[hashes[1] % self.size] -= 1
            self.filter[hashes[2] % self.size] -= 1
        else:
            raise ValueError('Value is not in the filter')

    def get_hashes(self, value: str) -> (int, int, int):
        return self.get_hash_sum(value), self.get_hash_sumproduct(value), self.get_hash_product(value)

    def __str__(self):
        return ' '.join(list(map(str, self.filter)))

    @staticmethod
    def get_hash_sum(value: str) -> int:
        return sum(ord(symbol) for symbol in value)

    @staticmethod
    def get_hash_sumproduct(value: str) -> int:
        value_len = len(value)
        return sum(
            ord(symbol) * ord(value[-(i + 1)]) for i, symbol in enumerate(value[:value_len // 2 + value_len % 2]))

    @staticmethod
    def get_hash_product(value: str) -> int:
        return sum(ord(symbol) * ord(value[i + 1]) for i, symbol in enumerate(value[:-1]))
