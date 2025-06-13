# Импорт необходимых библиотек
import pygame  # для создания игрового интерфейса
import sys  # для работы с системными функциями (например, выход из программы)
import math  # для математических вычислений (углы, расстояния)

# Импорт констант и классов из другого файла
from env_common import (
    GRID_SIZE,  # размер одной клетки сетки
    WINDOW_WIDTH,  # ширина окна
    WINDOW_HEIGHT,  # высота окна
    Block,  # класс для стен
    Goal,  # класс для цели
    Tank,  # класс для танка
    build_wall_coords,  # функция для создания координат стен
    in_collision,  # функция проверки столкновений
)

FPS = 30  # частота кадров в секунду

# Инициализация Pygame
pygame.init()
# Создание игрового окна
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
# Создание объекта для контроля времени
clock = pygame.time.Clock()
# Создание шрифта для отображения текста
font = pygame.font.SysFont("Arial", 20)

# Список стен (блоков) на карте
walls_list = [
    Block(3, 5), Block(4, 5), Block(5, 5),
    Block(10, 2), Block(10, 3), Block(10, 4),
    Block(15, 8), Block(16, 8), Block(17, 8)
]
# Построение координат стен
wall_positions = build_wall_coords(walls_list)

# Расчет максимального количества столбцов и строк в сетке
max_grid_cols = WINDOW_WIDTH // GRID_SIZE
max_grid_rows = WINDOW_HEIGHT // GRID_SIZE

# Создание цели (финиша) в позиции (18, 10)
target = Goal(18, 10)
# Создание танка игрока в позиции (2, 2) с углом поворота 45 градусов
player_tank = Tank(2, 2, angle=45)

# Флаг работы игрового цикла
game_running = True

# Словарь действий (для удобства использования)
ACTIONS = {
    0: "move_forward",  # движение вперед
    1: "move_backward",  # движение назад
    2: "turn_left",  # поворот влево
    3: "turn_right",  # поворот вправо
    4: "shoot"  # выстрел (не реализован)
}


# Функция расчета расстояния между двумя точками
def get_distance(a_x, a_y, b_x, b_y):
    return math.hypot(a_x - b_x, a_y - b_y)


# Функция расчета относительного угла между танком и целью
def get_relative_angle(tank_angle, tank_x, tank_y, goal_x, goal_y):
    dx = goal_x - tank_x  # разница по X
    dy = goal_y - tank_y  # разница по Y
    # Расчет угла к цели в градусах (atan2 возвращает угол в радианах)
    angle_to_goal = math.degrees(math.atan2(-dy, dx)) % 360
    # Расчет относительного угла (разница между углом танка и углом к цели)
    relative_angle = (angle_to_goal - tank_angle) % 360
    # Корректировка угла для диапазона [-180, 180]
    if relative_angle > 180:
        relative_angle -= 360
    return relative_angle


# Проверка достижения цели
def check_goal_reached(tank, goal):
    dist = get_distance(tank.x, tank.y, goal.x, goal.y)
    return dist < 1  # True если расстояние меньше 1 клетки


# Расчет расстояния до ближайшего препятствия в заданном направлении
def distance_to_obstacle(tank_x, tank_y, dx, dy, wall_positions, max_cols, max_rows):
    dist = 0  # начальное расстояние
    x, y = tank_x, tank_y  # текущие координаты
    # Пока координаты в пределах карты
    while 0 <= x < max_cols and 0 <= y < max_rows:
        x += dx  # движение по X
        y += dy  # движение по Y
        dist += 1  # увеличение расстояния
        # Если встретили стену - прерываем цикл
        if (x, y) in wall_positions:
            break
    return dist


# Получение текущего состояния системы
def get_state(tank, goal, wall_positions, max_cols, max_rows):
    # Расстояние до цели
    dist_goal = get_distance(tank.x, tank.y, goal.x, goal.y)
    # Относительный угол к цели
    rel_angle = get_relative_angle(tank.angle, tank.x, tank.y, goal.x, goal.y)
    # Расстояния до препятствий в 4 направлениях:
    # Вперед
    obs_front = distance_to_obstacle(tank.x, tank.y,
                                     int(round(math.cos(math.radians(tank.angle)))),
                                     -int(round(math.sin(math.radians(tank.angle)))),
                                     wall_positions, max_cols, max_rows)
    # Влево
    obs_left = distance_to_obstacle(tank.x, tank.y,
                                    int(round(math.cos(math.radians(tank.angle + 90)))),
                                    -int(round(math.sin(math.radians(tank.angle + 90)))),
                                    wall_positions, max_cols, max_rows)
    # Вправо
    obs_right = distance_to_obstacle(tank.x, tank.y,
                                     int(round(math.cos(math.radians(tank.angle - 90)))),
                                     -int(round(math.sin(math.radians(tank.angle - 90)))),
                                     wall_positions, max_cols, max_rows)
    # Назад
    obs_back = distance_to_obstacle(tank.x, tank.y,
                                    int(round(math.cos(math.radians(tank.angle + 180)))),
                                    -int(round(math.sin(math.radians(tank.angle + 180)))),
                                    wall_positions, max_cols, max_rows)

    return [tank.x, tank.y, tank.angle, dist_goal, rel_angle, obs_front, obs_left, obs_right, obs_back]


# Расчет награды (reward) для алгоритмов обучения
def calculate_reward(prev_dist, current_dist, collision, reached_goal):
    reward = -1  # базовая награда за каждый шаг
    if collision:  # штраф за столкновение
        reward -= 10
    if reached_goal:  # большая награда за достижение цели
        reward += 100
    elif current_dist < prev_dist:  # награда за приближение к цели
        reward += 10
    else:  # штраф за удаление от цели
        reward -= 5
    return reward


# Функция движения танка
def move_tank(tank, action, wall_positions, max_cols, max_rows):
    collision = False  # флаг столкновения
    if action == "turn_left":  # поворот влево
        tank.angle = (tank.angle + 15) % 360
    elif action == "turn_right":  # поворот вправо
        tank.angle = (tank.angle - 15) % 360
    elif action == "move_forward" or action == "move_backward":  # движение вперед/назад
        rad = math.radians(tank.angle)  # угол в радианах
        direction = 1 if action == "move_forward" else -1  # направление движения
        # Расчет новых координат
        dx = int(round(math.cos(rad))) * direction
        dy = -int(round(math.sin(rad))) * direction
        nx, ny = tank.x + dx, tank.y + dy
        # Проверка на возможность перемещения (нет столкновений и в пределах карты)
        if 0 <= nx < max_cols and 0 <= ny < max_rows and not in_collision(nx, ny, wall_positions, max_cols, max_rows):
            tank.x, tank.y = nx, ny  # обновление позиции
        else:
            collision = True  # обнаружено столкновение
    elif action == "shoot":  # выстрел (заглушка)
        pass
    return collision  # возвращаем флаг столкновения


# Функция отрисовки игрового мира
def draw_environment():
    screen.fill((30, 30, 30))  # заливка фона (темно-серый)

    # Отрисовка сетки
    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (0, y), (WINDOW_WIDTH, y))

    # Отрисовка стен
    for block in walls_list:
        block.draw(screen)

    # Отрисовка цели
    target.draw(screen)

    # Отрисовка танка
    tank_img = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(tank_img, (0, 0, 255), (0, 0, GRID_SIZE, GRID_SIZE))  # синий квадрат
    rotated_img = pygame.transform.rotate(tank_img, -player_tank.angle)  # поворот изображения
    rect = rotated_img.get_rect(center=(player_tank.x * GRID_SIZE + GRID_SIZE / 2,
                                        player_tank.y * GRID_SIZE + GRID_SIZE / 2))
    screen.blit(rotated_img, rect)

    # Отображение текущего состояния
    state = get_state(player_tank, target, wall_positions, max_grid_cols, max_grid_rows)
    state_text = font.render(f"State: {['{:.1f}'.format(s) for s in state]}", True, (255, 255, 255))
    screen.blit(state_text, (10, WINDOW_HEIGHT - 30))

    # Обновление экрана
    pygame.display.flip()


# Начальное расстояние до цели
prev_distance = get_distance(player_tank.x, player_tank.y, target.x, target.y)

# Основной игровой цикл
while game_running:
    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # если закрыли окно
            game_running = False
        elif event.type == pygame.KEYDOWN:  # если нажата клавиша
            action = None
            # Определение действия по нажатой клавише
            if event.key == pygame.K_LEFT:
                action = ACTIONS[2]  # поворот влево
            elif event.key == pygame.K_RIGHT:
                action = ACTIONS[3]  # поворот вправо
            elif event.key == pygame.K_UP:
                action = ACTIONS[0]  # движение вперед
            elif event.key == pygame.K_DOWN:
                action = ACTIONS[1]  # движение назад
            elif event.key == pygame.K_SPACE:
                action = ACTIONS[4]  # выстрел

            if action:  # если действие определено
                collision = move_tank(player_tank, action, wall_positions, max_grid_cols, max_grid_rows)
                current_distance = get_distance(player_tank.x, player_tank.y, target.x, target.y)
                reached = check_goal_reached(player_tank, target)
                reward = calculate_reward(prev_distance, current_distance, collision, reached)
                prev_distance = current_distance
                # Вывод информации о действии
                print(f"Action: {action}, Reward: {reward:.2f}, Collision: {collision}, Reached: {reached}")
                if reached:  # если цель достигнута
                    print("Цель достигнута! Поздравляем!")
                    game_running = False

    # Отрисовка мира
    draw_environment()
    # Контроль частоты кадров
    clock.tick(FPS)

# Завершение работы Pygame
pygame.quit()
# Выход из программы
sys.exit()