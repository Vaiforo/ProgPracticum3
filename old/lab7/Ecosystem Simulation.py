import random
from abc import ABC, abstractmethod

# Константы для времени суток
MORNING = 0
DAY = 1
EVENING = 2
NIGHT = 3


# Базовый класс для растений
class Plant(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @abstractmethod
    def grow(self, world):
        pass


# Растение, которое растет днем
class Lumiere(Plant):
    def grow(self, world):
        if world.time == DAY:
            self._spread(world)

    def _spread(self, world):
        adjacent = world.get_adjacent_positions(self.x, self.y)
        for pos in adjacent:
            if world.grid[pos[0]][pos[1]] is None and random.random() < 0.5:
                world.grid[pos[0]][pos[1]] = Lumiere(pos[0], pos[1])


# Растение, которое растет ночью
class Obscurite(Plant):
    def grow(self, world):
        if world.time == NIGHT:
            self._spread(world)

    def _spread(self, world):
        adjacent = world.get_adjacent_positions(self.x, self.y)
        for pos in adjacent:
            if world.grid[pos[0]][pos[1]] is None and random.random() < 0.5:
                world.grid[pos[0]][pos[1]] = Obscurite(pos[0], pos[1])


# Растение, которое растет утром и вечером
class Demi(Plant):
    def grow(self, world):
        if world.time == MORNING or world.time == EVENING:
            self._spread(world)

    def _spread(self, world):
        adjacent = world.get_adjacent_positions(self.x, self.y)
        for pos in adjacent:
            if world.grid[pos[0]][pos[1]] is None and random.random() < 0.5:
                world.grid[pos[0]][pos[1]] = Demi(pos[0], pos[1])


# Базовый класс для животных
class Animal(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hunger = 0

    def move(self, world):
        adjacent = world.get_adjacent_positions(self.x, self.y)
        best_pos = None
        best_score = -1
        for pos in adjacent:
            score = self._evaluate_position(world, pos)
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_pos:
            self.x, self.y = best_pos

    @abstractmethod
    def _evaluate_position(self, world, pos):
        pass

    def eat(self, world):
        if isinstance(self, Pauvre):
            if world.grid[self.x][self.y] is not None:
                world.grid[self.x][self.y] = None
                self.hunger = max(0, self.hunger - 5)
            elif self.hunger > 5:
                for animal in world.animals:
                    if isinstance(animal, Pauvre) and animal != self and animal.x == self.x and animal.y == self.y:
                        world.animals.remove(animal)
                        self.hunger = max(0, self.hunger - 10)
                        break
        elif isinstance(self, Malheureux):
            for animal in world.animals:
                if animal != self and animal.x == self.x and animal.y == self.y:
                    world.animals.remove(animal)
                    self.hunger = max(0, self.hunger - 10)
                    break
            else:
                if world.grid[self.x][self.y] is not None and random.random() < 0.5:
                    world.grid[self.x][self.y] = None
                    self.hunger = max(0, self.hunger - 5)

    def increase_hunger(self):
        self.hunger += 1
        if self.hunger >= 10:
            return True  # Животное умирает
        return False


# Животное-травоядное
class Pauvre(Animal):
    def _evaluate_position(self, world, pos):
        if world.grid[pos[0]][pos[1]] is not None:
            return 10  # Приоритет для клеток с растениями
        for animal in world.animals:
            if isinstance(animal, Pauvre) and animal.x == pos[0] and animal.y == pos[1]:
                return 5  # Приоритет для клеток с другими Pauvre
        return 0


# Животное-всеядное
class Malheureux(Animal):
    def _evaluate_position(self, world, pos):
        for animal in world.animals:
            if animal.x == pos[0] and animal.y == pos[1]:
                return 10  # Приоритет для клеток с другими животными
        if world.grid[pos[0]][pos[1]] is not None:
            return 5  # Приоритет для клеток с растениями
        return 0


# Класс мира
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.animals = []
        self.time = MORNING

    def get_adjacent_positions(self, x, y):
        positions = []
        if x > 0:
            positions.append((x - 1, y))
        if x < self.width - 1:
            positions.append((x + 1, y))
        if y > 0:
            positions.append((x, y - 1))
        if y < self.height - 1:
            positions.append((x, y + 1))
        return positions

    def advance_time(self):
        self.time = (self.time + 1) % 4

    def update(self):
        # Рост растений
        for row in self.grid:
            for plant in row:
                if plant is not None:
                    plant.grow(self)

        # Действия животных
        for animal in self.animals[:]:  # Копия списка для безопасного удаления
            animal.move(self)
            animal.eat(self)
            if animal.increase_hunger():
                self.animals.remove(animal)

    def display(self):
        time_names = ["Утро", "День", "Вечер", "Ночь"]
        print(f"Время суток: {time_names[self.time]}")
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.grid[x][y] is None:
                    row.append('.')
                elif isinstance(self.grid[x][y], Lumiere):
                    row.append('L')
                elif isinstance(self.grid[x][y], Obscurite):
                    row.append('O')
                elif isinstance(self.grid[x][y], Demi):
                    row.append('D')
                animals_in_cell = [a for a in self.animals if a.x == x and a.y == y]
                if len(animals_in_cell) > 0:
                    if len(animals_in_cell) == 1:
                        if isinstance(animals_in_cell[0], Pauvre):
                            row[-1] = 'P'
                        elif isinstance(animals_in_cell[0], Malheureux):
                            row[-1] = 'M'
                    else:
                        row[-1] = '+'
            print(' '.join(row))
        print()

    def initialize(self):
        # Размещение растений
        for _ in range(10):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            plant_type = random.choice([Lumiere, Obscurite, Demi])
            self.grid[x][y] = plant_type(x, y)

        # Размещение животных
        for _ in range(5):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.animals.append(Pauvre(x, y))
        for _ in range(5):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.animals.append(Malheureux(x, y))


# Основная функция для запуска симуляции
def main():
    world = World(10, 10)
    world.initialize()
    for step in range(20):
        print(f"Шаг {step + 1}")
        world.display()
        world.update()
        world.advance_time()


if __name__ == "__main__":
    main()
