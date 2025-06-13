import random
from enum import Enum, auto

# ----------------------------
# Time Manager и Время суток
# ----------------------------
class TimeOfDay(Enum):
    MORNING = auto()
    DAY = auto()
    EVENING = auto()
    NIGHT = auto()

class TimeManager:
    def __init__(self, change_interval_ticks=5):
        self.current_time = TimeOfDay.MORNING
        self.ticks = 0
        self.change_interval = change_interval_ticks

    def tick(self):
        self.ticks += 1
        if self.ticks >= self.change_interval:
            self.ticks = 0
            self.advance_time()

    def advance_time(self):
        mapping = {
            TimeOfDay.MORNING: TimeOfDay.DAY,
            TimeOfDay.DAY: TimeOfDay.EVENING,
            TimeOfDay.EVENING: TimeOfDay.NIGHT,
            TimeOfDay.NIGHT: TimeOfDay.MORNING,
        }
        self.current_time = mapping[self.current_time]

    def is_day(self):
        return self.current_time in (TimeOfDay.MORNING, TimeOfDay.DAY)

    def is_night(self):
        return self.current_time == TimeOfDay.NIGHT

    def is_low_light(self):
        return self.current_time in (TimeOfDay.MORNING, TimeOfDay.EVENING)

    def __str__(self):
        return self.current_time.name

# ----------------------------
# Растения
# ----------------------------
class PlantBase:
    name = "BasePlant"
    def __init__(self, world, x, y):
        self.world = world
        self.x = x
        self.y = y

    def can_grow(self):
        raise NotImplementedError

    def activity_level(self):
        raise NotImplementedError

    def is_more_active_than(self, other_plant):
        return self.activity_level() > other_plant.activity_level()

    def grow(self):
        if not self.can_grow():
            return
        neighbors = self.world.get_neighbors(self.x, self.y)
        random.shuffle(neighbors)
        for nx, ny in neighbors:
            occupant = self.world.get_plant_at(nx, ny)
            if occupant is None:
                self.world.place_plant(self.__class__, nx, ny)
                break
            else:
                if self.is_more_active_than(occupant):
                    if random.random() < 0.5:
                        self.world.remove_plant(nx, ny)
                        self.world.place_plant(self.__class__, nx, ny)
                        break

class Lumiere(PlantBase):
    name = "Lumiere"
    def can_grow(self):
        return self.world.time_manager.is_day()

    def activity_level(self):
        return 1 if self.can_grow() else 0

class Obscurite(PlantBase):
    name = "Obscurite"
    def can_grow(self):
        return self.world.time_manager.is_night()

    def activity_level(self):
        return 1 if self.can_grow() else 0

class Demi(PlantBase):
    name = "Demi"
    def can_grow(self):
        return self.world.time_manager.is_low_light()

    def activity_level(self):
        return 1 if self.can_grow() else 0

# ----------------------------
# Животные
# ----------------------------
class AnimalBase:
    name = "BaseAnimal"
    def __init__(self, world, x, y):
        self.world = world
        self.x = x
        self.y = y
        self.hunger = 0
        self.group = None

    def step(self):
        raise NotImplementedError

    def move_towards(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        step_x = (dx > 0) - (dx < 0)
        step_y = (dy > 0) - (dy < 0)
        new_x = self.x + step_x
        new_y = self.y + step_y
        if self.world.is_free(new_x, new_y):
            self.x = new_x
            self.y = new_y

    def wander(self):
        dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        new_x = self.x + dx
        new_y = self.y + dy
        if self.world.is_free(new_x, new_y):
            self.x, self.y = new_x, new_y

class Pauvre(AnimalBase):
    name = "Pauvre"
    def __init__(self, world, x, y):
        super().__init__(world, x, y)
        self.aggression = 0

    def step(self):
        self.aggression = min(10, self.hunger)
        if self.world.time_manager.current_time == TimeOfDay.NIGHT:
            return
        plant = self.world.find_nearest_plant(self.x, self.y, "Lumiere")
        if plant:
            self.move_towards(plant.x, plant.y)
            if (self.x, self.y) == (plant.x, plant.y):
                self.world.remove_plant(plant.x, plant.y)
                self.hunger = 0
        else:
            self.wander()
        self.hunger += 1

class Malheureux(AnimalBase):
    name = "Malheureux"
    def step(self):
        current_time = self.world.time_manager.current_time
        if current_time in [TimeOfDay.DAY, TimeOfDay.NIGHT]:
            return
        if self.hunger > 5:
            if random.random() < 0.5:
                return
        prey = self.world.find_nearest_prey(self.x, self.y)
        if prey:
            self.move_towards(prey.x, prey.y)
            if (self.x, self.y) == (prey.x, prey.y):
                self.world.remove_animal(prey)
                self.hunger = 0
        else:
            self.wander()
        self.hunger += 1

# ----------------------------
# Мир
# ----------------------------
class World:
    def __init__(self, width, height, time_change_interval=5):
        self.width = width
        self.height = height
        self.grid = [[None]*height for _ in range(width)]
        self.plants = []
        self.animals = []
        self.time_manager = TimeManager(change_interval_ticks=time_change_interval)

    def is_in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_plant_at(self, x, y):
        if not self.is_in_bounds(x, y):
            return None
        cell = self.grid[x][y]
        if cell and cell in self.plants:
            return cell
        return None

    def get_animal_at(self, x, y):
        for a in self.animals:
            if a.x == x and a.y == y:
                return a
        return None

    def place_plant(self, plant_cls, x, y):
        if not self.is_in_bounds(x, y) or self.get_plant_at(x, y):
            return False
        plant = plant_cls(self, x, y)
        self.plants.append(plant)
        self.grid[x][y] = plant
        return True

    def remove_plant(self, x, y):
        plant = self.get_plant_at(x, y)
        if plant:
            self.plants.remove(plant)
            self.grid[x][y] = None

    def place_animal(self, animal_cls, x, y):
        if not self.is_in_bounds(x, y) or self.get_animal_at(x, y):
            return False
        animal = animal_cls(self, x, y)
        self.animals.append(animal)
        return True

    def remove_animal(self, animal):
        if animal in self.animals:
            self.animals.remove(animal)

    def is_free(self, x, y):
        return self.is_in_bounds(x, y) and not self.get_plant_at(x, y) and not self.get_animal_at(x, y)

    def get_neighbors(self, x, y):
        candidates = [(x+dx, y+dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]]
        return [(nx, ny) for nx, ny in candidates if self.is_in_bounds(nx, ny)]

    def find_nearest_plant(self, x, y, plant_name):
        plants = [p for p in self.plants if p.name == plant_name]
        if not plants:
            return None
        plants = sorted(plants, key=lambda p: abs(p.x - x) + abs(p.y - y))
        return plants[0]

    def find_nearest_prey(self, x, y):
        prey_candidates = [a for a in self.animals if (a.name == "Pauvre")]
        prey_candidates += [p for p in self.plants if p.name in ("Demi", "Obscurite")]
        if not prey_candidates:
            return None
        prey_candidates = sorted(prey_candidates, key=lambda c: abs(c.x - x) + abs(c.y - y))
        return prey_candidates[0]

    def step(self):
        self.time_manager.tick()
        for plant in self.plants[:]:
            plant.grow()
        for animal in self.animals[:]:
            animal.step()

    def __str__(self):
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                cell = self.grid[x][y]
                if cell is None:
                    animal = self.get_animal_at(x, y)
                    if animal:
                        if animal.name == "Pauvre":
                            line += "P"
                        elif animal.name == "Malheureux":
                            line += "M"
                        else:
                            line += "A"
                    else:
                        line += "."
                else:
                    if cell.name == "Lumiere":
                        line += "L"
                    elif cell.name == "Obscurite":
                        line += "O"
                    elif cell.name == "Demi":
                        line += "D"
                    else:
                        line += "?"
            lines.append(line)
        return "\n".join(lines)

# ----------------------------
# Визуализация и отладка
# ----------------------------
def print_world_state(world):
    print(f"Time of day: {world.time_manager}")
    print(world)
    print(f"Plants: {len(world.plants)} | Animals: {len(world.animals)}")
    print("-" * 40)

# ----------------------------
# Основной цикл симуляции
# ----------------------------
def main():
    w = World(20, 15)

    # Инициализация растений
    for _ in range(10):
        x, y = random.randint(0, w.width-1), random.randint(0, w.height-1)
        w.place_plant(random.choice([Lumiere, Obscurite, Demi]), x, y)

    # Инициализация животных
    for _ in range(5):
        x, y = random.randint(0, w.width-1), random.randint(0, w.height-1)
        w.place_animal(random.choice([Pauvre, Malheureux]), x, y)

    # Запуск симуляции 30 шагов
    for step in range(30):
        print(f"Step {step + 1}")
        w.step()
        print_world_state(w)

if __name__ == "__main__":
    main()
