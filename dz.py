import random
import PySimpleGUI as sg

from lab8 import Demi, EcoMeta, Lumiere, Obscurite, World

REGISTERED_PLANTS = {}
REGISTERED_ANIMALS = {}


class Animal(metaclass=EcoMeta):
    ENTITY_TYPE = 'animal'

    def __init__(self):
        self.x = None
        self.y = None
        self.hunger = 0
        self.vision_radius = random.uniform(1, 3)
        self.scale = random.uniform(0.5, 1.5)


class Pauvre(Animal):
    BEHAVIOR_DSL = "morning:eat=8;evening:eat=2;default:eat=2"


class Malheureux(Animal):
    BEHAVIOR_DSL = "morning:eat=6;evening:eat=2;default:eat=2"


def get_neighbors(world, animal):
    neighbors = []
    for entity in world.entities:
        if entity != animal and entity.ENTITY_TYPE == 'animal':
            distance = ((entity.x - animal.x) ** 2 +
                        (entity.y - animal.y) ** 2) ** 0.5
            if distance <= animal.vision_radius:
                neighbors.append(entity)
    return neighbors


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
    animal_vision_radii = {'Pauvre': [], 'Malheureux': []}
    for entity in world.entities:
        if entity.ENTITY_TYPE == 'plant':
            plant_counts[entity.__class__.__name__] += 1
        elif entity.ENTITY_TYPE == 'animal':
            animal_counts[entity.__class__.__name__] += 1
            animal_vision_radii[entity.__class__.__name__].append(
                entity.vision_radius)
    dct = {}
    for animal in animal_vision_radii:
        dct[animal] = sum(animal_vision_radii[animal]) / len(
            animal_vision_radii[animal]) if animal_vision_radii[animal] else 0
    stats['plants'] = plant_counts
    stats['animals'] = animal_counts
    stats['avg_vision_radius'] = dct
    return stats


def format_stats(stats):
    s = "Растения:\n"
    for name, count in stats['plants'].items():
        s += f"  {name}: {count}\n"
    s += "Животные:\n"
    for name, count in stats['animals'].items():
        s += f"  {name}: {count}\n"
        s += f"  Средний радиус обзора: {stats['avg_vision_radius'][name]:.2f}\n"
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
        [sg.Graph(canvas_size=(400, 300), graph_bottom_left=(0, 0),
                  graph_top_right=(width, height), key='-MAP-')],
        [sg.Button('Запуск'), sg.Button('Пауза'), sg.Button('Сброс')],
        [sg.Multiline(size=(80, 20), key='-STATS-', disabled=True)]
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
                    neighbors = get_neighbors(w, entity)
                    if neighbors:
                        neighbor_info = "Соседи в пределах радиуса обзора:\n"
                        for neighbor in neighbors:
                            neighbor_info += f"  {neighbor.__class__.__name__} на ({neighbor.x}, {neighbor.y})\n"
                    else:
                        neighbor_info = "Нет соседей в пределах радиуса обзора."
                    window['-STATS-'].update(format_stats(stats) +
                                             "\n" + neighbor_info)
                else:
                    window['-STATS-'].update(format_stats(stats))
            else:
                window['-STATS-'].update(format_stats(stats))
        if is_running:
            w.tick()
            draw_world(window['-MAP-'], w)
            stats = get_stats(w)
            window['-STATS-'].update(format_stats(stats))

    window.close()
