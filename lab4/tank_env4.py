# tank_env.py
# -*- coding: utf-8 -*-

import math
import numpy as np
import gym
from gym import spaces

from env_common import in_collision  # функция проверки коллизии корпуса танка

class TankEnv(gym.Env):
    """
    Gym-совместимая среда «Танк и Цель» на двумерной сетке.

    State (наблюдение) — 9-мерный вектор:
       0) x:          позиция танка по горизонтали (float)
       1) y:          позиция танка по вертикали (float)
       2) angle:      угол танка в градусах (float, [0,360))
       3) dist:       евклидово расстояние до цели (float)
       4) rel_angle:  угол на цель относительно направления танка (float, [-180,180])
       5) d_up:       дистанция (в клетках) до ближайшей стены вверх
       6) d_right:    дистанция до стены вправо
       7) d_down:     дистанция до стены вниз
       8) d_left:     дистанция до стены влево

    Actions (Discrete(5)):
       0 — движение вперёд
       1 — движение назад
       2 — поворот влево (плюс turn_deg)
       3 — поворот вправо (минус turn_deg)
       4 — выстрел (попытка попасть в цель)

    Reward:
       +100 за попадание выстрелом или въездом в цель
       −10 за столкновение со стеной
       −1  за шаг времени
       +10 за приближение к цели (dist уменьшилось)
       −5  за удаление от цели  (dist увеличилось)
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        width: int,
        height: int,
        wall_coords: set[tuple[int,int]],
        goal: tuple[int,int],
        step_size: float = 1.0,
        turn_deg: float  = 15.0
    ) -> None:
        super().__init__()

        # Размеры карты (в клетках)
        self.width, self.height = width, height
        # Множество кортежей (x,y) клеток со стенами
        self.walls = wall_coords
        # Цель — координаты (x,y) в клетках
        self.goal  = goal

        # Дискретное пространство действий из 5 шагов
        self.action_space = spaces.Discrete(5)

        # Пространство наблюдений: вектор из 9 float-ов
        high = np.array([
            width,              # x ≤ width
            height,             # y ≤ height
            360.0,              # angle ≤ 360
            math.hypot(width, height),  # макс. расстояние до цели
            180.0,              # rel_angle ∈ [-180,180]
            max(width, height), # dist up
            max(width, height), # dist right
            max(width, height), # dist down
            max(width, height), # dist left
        ], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0, 0.0, -180.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Параметры движения
        self.step_size = step_size   # шаг вперед/назад
        self.turn_deg  = turn_deg    # угол поворота

        self.reset()


    def reset(self) -> np.ndarray:
        """
        Сброс среды. Танку задаётся начальная позиция (1,1) и угол 0°.
        Возвращает начальное наблюдение.
        """
        self.tank_pos   = np.array([1.0, 1.0], dtype=np.float32)
        self.tank_angle = 0.0
        # Для расчёта приближения/удаления запомним начальную дистанцию
        self.prev_dist  = self._dist_to_goal()
        self.done       = False
        return self._get_state()


    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Выполняет одно действие:
          - обновляет позицию/угол или совершает выстрел,
          - вычисляет награду,
          - возвращает (obs, reward, done, info).
        """
        assert self.action_space.contains(action), "Недопустимое действие"
        reward = -1.0  # штраф за время

        # 0) движение вперёд
        if action == 0:
            dx, dy = self._heading_vector()
            new_pos = self.tank_pos + self.step_size * np.array([dx, dy], dtype=np.float32)
            # столкновение или нет?
            if in_collision(new_pos[0], new_pos[1],
                            self.walls, self.width, self.height):
                reward -= 10.0
            else:
                self.tank_pos = new_pos

        # 1) движение назад
        elif action == 1:
            dx, dy = self._heading_vector()
            new_pos = self.tank_pos - self.step_size * np.array([dx, dy], dtype=np.float32)
            if in_collision(new_pos[0], new_pos[1],
                            self.walls, self.width, self.height):
                reward -= 10.0
            else:
                self.tank_pos = new_pos

        # 2) поворот влево
        elif action == 2:
            self.tank_angle = (self.tank_angle + self.turn_deg) % 360.0

        # 3) поворот вправо
        elif action == 3:
            self.tank_angle = (self.tank_angle - self.turn_deg) % 360.0

        # 4) выстрел
        elif action == 4:
            if self._hit_goal():
                reward += 100.0
                self.done = True

        # 5) приближение/удаление
        dist = self._dist_to_goal()
        if dist < self.prev_dist:
            reward += 10.0
        elif dist > self.prev_dist:
            reward -= 5.0
        self.prev_dist = dist

        # 6) въезд в цель «танком»
        if not self.done and self._in_goal():
            reward += 100.0
            self.done = True

        obs = self._get_state()
        return obs, reward, self.done, {}


    def _heading_vector(self) -> tuple[float,float]:
        """
        Вычисляет единичный вектор (dx,dy) в направлении текущего угла.
        dx = cos(angle), dy = −sin(angle).
        """
        rad = math.radians(self.tank_angle)
        return math.cos(rad), -math.sin(rad)


    def _dist_to_goal(self) -> float:
        """
        Евклидово расстояние от танка до цели.
        """
        return float(np.linalg.norm(self.tank_pos - np.array(self.goal, dtype=np.float32)))


    def _angle_to_goal(self) -> float:
        """
        Угол на цель относительно абсолютного направления танка.
        Результат в диапазоне [−180°, +180°].
        """
        vec = np.array(self.goal, dtype=np.float32) - self.tank_pos
        goal_ang = (math.degrees(math.atan2(-vec[1], vec[0])) + 360.0) % 360.0
        diff     = (goal_ang - self.tank_angle + 180.0) % 360.0 - 180.0
        return diff


    def _hit_goal(self) -> bool:
        """
        Попадание выстрелом: угол к цели мал (±turn_deg/2)
        и дистанция меньше 2×step_size.
        """
        return abs(self._angle_to_goal()) < self.turn_deg/2 and self._dist_to_goal() < 2*self.step_size


    def _in_goal(self) -> bool:
        """
        Въезд танка в цель (центральная клетка) на расстоянии <1 клетки.
        """
        return self._dist_to_goal() < 1.0


    def _ray_dist(self, dx: int, dy: int) -> float:
        """
        «Луч» от текущей позиции в направлении (dx,dy) —
        считаем число клеток до первой стены или границы.
        """
        steps = 0
        x, y  = self.tank_pos
        while True:
            x += dx; y += dy; steps += 1
            xi, yi = int(round(x)), int(round(y))
            if xi<0 or yi<0 or xi>=self.width or yi>=self.height or (xi, yi) in self.walls:
                return float(steps)


    def _get_state(self) -> np.ndarray:
        """
        Собираем полный вектор наблюдения:
        [x, y, angle, dist_to_goal, rel_angle, dist_up, dist_right, dist_down, dist_left]
        """
        dist    = self._dist_to_goal()
        rel_ang = self._angle_to_goal()
        up      = self._ray_dist(0, -1)
        right   = self._ray_dist(1,  0)
        down    = self._ray_dist(0,  1)
        left    = self._ray_dist(-1, 0)
        return np.array([
            self.tank_pos[0], self.tank_pos[1],
            self.tank_angle,
            dist, rel_ang,
            up, right, down, left
        ], dtype=np.float32)


    def render(self, mode='human') -> None:
        """
        Текстовый вывод состояния в консоль:
        позиция, угол и текущее расстояние.
        """
        print(f"Tank at ({self.tank_pos[0]:.1f},{self.tank_pos[1]:.1f}) "
              f"↻{self.tank_angle:.1f}° dist={self._dist_to_goal():.1f}")


    def close(self) -> None:
        """Заглушка (Gym требует)."""
        pass


# При запуске напрямую — демонстрация «рандомного» эпизода
if __name__ == "__main__":
    walls = [(3,5),(4,5),(5,5),(10,2),(10,3),(10,4),(15,8),(16,8),(17,8)]
    goal  = (18,10)
    env   = TankEnv(20, 15, walls, goal)

    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()   # рандом
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print("Episode finished, total reward:", total_reward)
