import random
import PySimpleGUI as sg

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
        self.vision_radius = random.uniform(1, 3)
        self.scale = random.uniform(0.5, 1.5)


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
        for cls in list(REGISTERED_PLANTS.values()) + list(REGISTERED_ANIMALS.values()):
            cls.adapt_to_time(current_time)
        for entity in list(self.entities):
            entity.act(self)
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


def draw_world(graph, world):
    graph.erase()
    for entity in world.entities:
        if entity.ENTITY_TYPE == 'plant':
            color = {'Lumiere': '#FFFF00', 'Obscurite': '#0000FF',
                     'Demi': '#808080'}[entity.__class__.__name__]
            graph.draw_rectangle((entity.x, entity.y),
                                 (entity.x+1, entity.y+1), fill_color=color)
        elif entity.ENTITY_TYPE == 'animal':
            color = {'Pauvre': '#FFFF00', 'Malheureux': '#800080'}[
                entity.__class__.__name__]
            center = (entity.x + 0.5, entity.y + 0.5)
            radius = entity.scale * 0.5
            graph.draw_circle(center, radius, fill_color=color)


def get_stats(world):
    stats = {}
    plant_counts = {'Lumiere': 0, 'Obscurite': 0, 'Demi': 0}
    animal_counts = {'Pauvre': 0, 'Malheureux': 0}
    vision_radii = []
    for entity in world.entities:
        if entity.ENTITY_TYPE == 'plant':
            plant_counts[entity.__class__.__name__] += 1
        elif entity.ENTITY_TYPE == 'animal':
            animal_counts[entity.__class__.__name__] += 1
            vision_radii.append(entity.vision_radius)
    stats['plants'] = plant_counts
    stats['animals'] = animal_counts
    stats['avg_vision_radius'] = sum(
        vision_radii) / len(vision_radii) if vision_radii else 0
    return stats


def format_stats(stats):
    s = "Растения:\n"
    for name, count in stats['plants'].items():
        s += f"  {name}: {count}\n"
    s += "Животные:\n"
    for name, count in stats['animals'].items():
        s += f"  {name}: {count}\n"
    s += f"Средний радиус обзора: {stats['avg_vision_radius']:.2f}\n"
    return s


if __name__ == "__main__":
    width = 8
    height = 4
    w = World(width, height)
    w.add_entity(Lumiere(), 1, 1)
    w.add_entity(Obscurite(), 2, 2)
    w.add_entity(Demi(), 3, 3)
    w.add_entity(Pauvre(), 4, 1)
    w.add_entity(Malheureux(), 5, 2)

    layout = [
        [sg.Text('Время:'), sg.Slider(range=(0, 100),
                                      orientation='h', key='-TIME-', enable_events=True)],
        [sg.Graph(canvas_size=(600, 400), graph_bottom_left=(0, 0),
                  graph_top_right=(width, height), key='-MAP-')],
        [sg.Button('Запуск'), sg.Button('Пауза'), sg.Button('Сброс')],
        [sg.Multiline(size=(80, 5), key='-STATS-', disabled=True)]
    ]
    window = sg.Window('Симулятор экосистемы', layout, finalize=True)
    window['-MAP-'].bind('<Motion>', '+MOTION+')
    draw_world(window['-MAP-'], w)
    stats = get_stats(w)
    window['-STATS-'].update(format_stats(stats))

    is_running = False
    vision_circle_id = None

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Запуск':
            is_running = True
        elif event == 'Пауза':
            is_running = False
        elif event == 'Сброс':
            w = World(width, height)
            w.add_entity(Lumiere(), 1, 1)
            w.add_entity(Obscurite(), 2, 2)
            w.add_entity(Demi(), 3, 3)
            w.add_entity(Pauvre(), 4, 1)
            w.add_entity(Malheureux(), 5, 2)
            draw_world(window['-MAP-'], w)
            stats = get_stats(w)
            window['-STATS-'].update(format_stats(stats))
        elif event == '-TIME-' and not is_running:
            tick_num = int(values['-TIME-'])
            w = World(width, height)
            w.add_entity(Lumiere(), 1, 1)
            w.add_entity(Obscurite(), 2, 2)
            w.add_entity(Demi(), 3, 3)
            w.add_entity(Pauvre(), 4, 1)
            w.add_entity(Malheureux(), 5, 2)
            for _ in range(tick_num):
                w.tick()
            draw_world(window['-MAP-'], w)
            stats = get_stats(w)
            window['-STATS-'].update(format_stats(stats))
        elif event == '-MAP-+MOTION+':
            x, y = values['-MAP-']
            cell_x = int(x)
            cell_y = int(y)
            if vision_circle_id:
                window['-MAP-'].delete_figure(vision_circle_id)
                vision_circle_id = None
            if 0 <= cell_x < w.width and 0 <= cell_y < w.height:
                entity = w.grid[cell_y][cell_x]
                if entity and entity.ENTITY_TYPE == 'animal':
                    center = (entity.x + 0.5, entity.y + 0.5)
                    radius = entity.vision_radius
                    vision_circle_id = window['-MAP-'].draw_circle(
                        center, radius, line_color='red', line_width=2)
        if is_running:
            w.tick()
            draw_world(window['-MAP-'], w)
            stats = get_stats(w)
            window['-STATS-'].update(format_stats(stats))

    window.close()
