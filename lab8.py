import random
import unittest

# Глобальные реестры классов растений и животных
REGISTERED_PLANTS = {}
REGISTERED_ANIMALS = {}


class EcoMeta(type):
    """
    Метакласс для регистрации и динамического добавления поведения по DSL.
    """

    def __new__(mcs, classname, bases, attrs):
        cls = super().__new__(mcs, classname, bases, attrs)

        growth_dsl = attrs.get('GROWTH_DSL')
        behavior_dsl = attrs.get('BEHAVIOR_DSL')

        # Регистрация классов
        if growth_dsl:
            REGISTERED_PLANTS[classname] = cls
            rules = {k.strip(): float(v.strip()) for k, v in
                     (item.split(':') for item in growth_dsl.split(','))}

            @classmethod
            def adapt_to_time(cls, time):
                cls.growth_prob = rules.get(time, 0.0)
            cls.adapt_to_time = adapt_to_time

            def spread(self, world):
                if random.random() < self.growth_prob:
                    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    random.shuffle(neighbors)
                    for dx, dy in neighbors:
                        nx, ny = self.x + dx, self.y + dy
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            if world.grid[ny][nx] is None:
                                world.add_entity(type(self)(), nx, ny)
                                break
            cls.spread = spread
            cls.act = lambda self, world: self.spread(world)

        if behavior_dsl:
            REGISTERED_ANIMALS[classname] = cls
            # Пример парсинга поведения: morning:eat=2;evening:eat=0.5;default:eat=1
            rules = {}
            for part in behavior_dsl.split(';'):
                timekey, action = part.split(':')
                if action.startswith('eat='):
                    rules[timekey.strip()] = float(action.split('=')[1])

            @classmethod
            def adapt_to_time(cls, time):
                cls.eat_amount = rules.get(time, rules.get('default', 0.0))
            cls.adapt_to_time = adapt_to_time

            def eat(self, world):
                self.hunger = max(0, self.hunger - self.eat_amount)
            cls.eat = eat

            def move(self, world):
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                dx, dy = random.choice(directions)
                world.move_entity(self, self.x + dx, self.y + dy)
            cls.move = move

            def reproduce(self, world):
                if random.random() < 0.05:
                    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    random.shuffle(neighbors)
                    for dx, dy in neighbors:
                        nx, ny = self.x + dx, self.y + dy
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            if world.grid[ny][nx] is None:
                                world.add_entity(type(self)(), nx, ny)
                                break
            cls.reproduce = reproduce

            def act(self, world):
                self.hunger += 1
                self.eat(world)
                self.move(world)
                self.reproduce(world)
            cls.act = act

        return cls


# Базовые классы с метаклассом
class Plant(metaclass=EcoMeta):
    ENTITY_TYPE = 'plant'

    def __init__(self):
        self.x = None
        self.y = None


class Animal(metaclass=EcoMeta):
    ENTITY_TYPE = 'animal'

    def __init__(self):
        self.x = None
        self.y = None
        self.hunger = 0


class Lumiere(Plant):
    GROWTH_DSL = "morning:0.3,day:0.3,evening:0.0,night:0.0"


class Obscurite(Plant):
    GROWTH_DSL = "night:0.3,evening:0.3,morning:0.0,day:0.0"


class Demi(Plant):
    GROWTH_DSL = "morning:0.15,evening:0.15,day:0.05,night:0.05"


class Pauvre(Animal):
    BEHAVIOR_DSL = "morning:eat=2;evening:eat=0.5;default:eat=1"


class Malheureux(Animal):
    BEHAVIOR_DSL = "morning:eat=1;evening:eat=1;default:eat=0.5"


# Класс мира
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None]*width for _ in range(height)]
        self.entities = []
        self.times = ['morning', 'day', 'evening', 'night']
        self.time_idx = 0

    def add_entity(self, entity, x, y):
        entity.x = x
        entity.y = y
        self.grid[y][x] = entity
        self.entities.append(entity)

    def move_entity(self, entity, nx, ny):
        if 0 <= nx < self.width and 0 <= ny < self.height:
            if self.grid[ny][nx] is None:
                self.grid[entity.y][entity.x] = None
                self.grid[ny][nx] = entity
                entity.x = nx
                entity.y = ny

    def tick(self):
        current_time = self.times[self.time_idx]
        print(f"=== Tick, Time: {current_time} ===")

        # Адаптируем классы под время
        for cls in list(REGISTERED_PLANTS.values()) + list(REGISTERED_ANIMALS.values()):
            cls.adapt_to_time(current_time)

        # Действия всех объектов
        for entity in list(self.entities):
            entity.act(self)

        # Переключаем время суток
        self.time_idx = (self.time_idx + 1) % len(self.times)

    def __str__(self):
        rows = []
        for row in self.grid:
            s = ''
            for cell in row:
                if cell is None:
                    s += '.'
                else:
                    s += cell.__class__.__name__[0]
            rows.append(s)
        return '\n'.join(rows)


# Тесты
class EcosystemMetaTests(unittest.TestCase):

    def test_registration(self):
        self.assertIn('Lumiere', REGISTERED_PLANTS)
        self.assertIn('Malheureux', REGISTERED_ANIMALS)

    def test_methods_presence(self):
        self.assertTrue(hasattr(Lumiere, 'adapt_to_time'))
        self.assertTrue(callable(Pauvre.eat))
        self.assertTrue(callable(Malheureux.reproduce))

    def test_behavior_switching(self):
        Lumiere.adapt_to_time('morning')
        self.assertAlmostEqual(Lumiere.growth_prob, 0.3)
        Pauvre.adapt_to_time('evening')
        self.assertAlmostEqual(Pauvre.eat_amount, 0.5)


if __name__ == "__main__":
    unittest.main(exit=False)

    # Демонстрация работы симуляции
    w = World(8, 4)

    # Добавим растения и животных
    w.add_entity(Lumiere(), 1, 1)
    w.add_entity(Obscurite(), 2, 2)
    w.add_entity(Demi(), 3, 3)

    w.add_entity(Pauvre(), 4, 1)
    w.add_entity(Malheureux(), 5, 2)

    for _ in range(6):
        print(w)
        w.tick()
        print()
