import random


class Hasher:
    """Генератор хешей для строк"""

    def __init__(self, initial_seed: int, output_bits: int):
        self.seed = initial_seed  # Начальное значение для хеширования
        self.output_bits = output_bits  # Сколько бит должен содержать итоговый хеш
        self.modulus = 2 ** (output_bits + 5)  # Ограничиваем диапазон хеша

    def generate_hash(self, input_string: str) -> int:
        """Генерирует хеш для строки"""
        hash_value = self.seed
        for char in input_string:
            # Простая хеш-функция: умножаем на 31, добавляем код символа
            hash_value = (hash_value * 31 + ord(char)) % self.modulus
        # Берем только нужное количество бит
        return hash_value & ((1 << self.output_bits) - 1)


class QuotientFilterStruct:
    """Реализация Quotient Filter"""

    def __init__(self, quotient_bits: int, remainder_bits: int):
        self.q_bits = quotient_bits  # Бит для выбора ячейки
        self.r_bits = remainder_bits  # Бит для хранения "отпечатка"
        self.size = 1 << quotient_bits  # Всего ячеек = 2^q_bits
        # Каждая ячейка содержит 3 флага и значение
        self.slots = [{'occ': False, 'shift': False, 'cont': False, 'rem': None}
                      for _ in range(self.size)]
        self.hasher = Hasher(17, quotient_bits + remainder_bits)

    def _get_indices(self, full_hash: int):
        """Разбивает хеш на номер ячейки и отпечаток"""
        quotient = (full_hash >> self.r_bits) % self.size
        remainder = full_hash & ((1 << self.r_bits) - 1)
        return quotient, remainder

    def add(self, key: str):
        """Добавляет ключ в фильтр"""
        # Шаг 1: Получаем хеш и разбиваем его
        h = self.hasher.generate_hash(key)
        q, r = self._get_indices(h)

        # Шаг 2: Помечаем ячейку как занятую
        self.slots[q]['occ'] = True

        # Шаг 3: Ищем место для вставки
        idx = q
        while self.slots[idx]['rem'] is not None:
            idx = (idx + 1) % self.size

        # Шаг 4: Устанавливаем флаги
        if idx != q:
            self.slots[idx]['shift'] = True
            if self.slots[(idx - 1) % self.size]['rem'] is not None:
                self.slots[idx]['cont'] = True

        # Шаг 5: Сохраняем отпечаток
        self.slots[idx]['rem'] = r

    def contains(self, key: str) -> bool:
        """Проверяет наличие ключа в фильтре"""
        h = self.hasher.generate_hash(key)
        q, r = self._get_indices(h)

        # Ищем в кластере ячеек
        idx = q
        while True:
            # Проверяем текущую ячейку
            if self.slots[idx]['rem'] == r:
                return True
            # Прекращаем если дошли до конца кластера
            if not self.slots[idx]['cont']:
                break
            idx = (idx + 1) % self.size

        return False

    def print_debug(self):
        """Выводит состояние фильтра"""
        print("Яч | Occ Shift Cont | Отпечаток")
        print("-----------------------------")
        for i, slot in enumerate(self.slots):
            rem = slot['rem'] if slot['rem'] is not None else '-'
            print(f"{i:3} |  {int(slot['occ'])}     {int(slot['shift'])}     {int(slot['cont'])}  | {rem}")


def test_fpr(q_bits: int, r_bits: int, items=1000, tests=5000) -> float:
    """Тестирует вероятность ложных срабатываний"""
    qf = QuotientFilterStruct(q_bits, r_bits)

    # Добавляем тестовые элементы
    for i in range(min(items, (1 << q_bits) - 1)):
        qf.add(str(i))

    # Проверяем случайные элементы (которые не добавляли)
    false_pos = 0
    for _ in range(tests):
        if qf.contains(str(random.randint(10 ** 6, 10 ** 7))):
            false_pos += 1

    return false_pos / tests
