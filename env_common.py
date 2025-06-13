# -*- coding: utf-8 -*-
"""
env_common.py

Общий модуль для работы с 2D-картой, танком, стенами, целью и проверкой коллизий.
"""

import math
from math import floor, ceil
import pygame

# -----------------------
# Параметры сетки и окна
# -----------------------
GRID_SIZE     = 40
WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600

# -----------------------
# Классы для среды
# -----------------------
class Block:
    """Препятствие (стена) на карте."""
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def draw(self, surface: pygame.Surface) -> None:
        """Рисует стену-квадратик размером GRID_SIZE."""
        rect = pygame.Rect(self.x * GRID_SIZE,
                           self.y * GRID_SIZE,
                           GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, (80, 80, 80), rect)


class Goal:
    """Цель на карте."""
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def draw(self, surface: pygame.Surface) -> None:
        """Рисует цель в виде круга в центре ячейки."""
        cx = self.x * GRID_SIZE + GRID_SIZE // 2
        cy = self.y * GRID_SIZE + GRID_SIZE // 2
        radius = GRID_SIZE // 3
        pygame.draw.circle(surface, (0, 200, 0), (cx, cy), radius)


class Tank:
    """Танк с позицией (в клетках) и углом (в градусах)."""
    def __init__(self, x: int, y: int, angle: float = 0.0) -> None:
        self.x, self.y = x, y
        self.angle = angle

    def draw(self, surface: pygame.Surface) -> None:
        """Рисует танк: корпус и ствол по текущему углу."""
        cx = self.x * GRID_SIZE + GRID_SIZE // 2
        cy = self.y * GRID_SIZE + GRID_SIZE // 2

        # тело (rectangle)
        body = pygame.Surface((GRID_SIZE, GRID_SIZE // 2), pygame.SRCALPHA)
        body.fill((200, 200, 0))
        rotated = pygame.transform.rotate(body, -self.angle)
        rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rect.topleft)

        # ствол (линия)
        rad = math.radians(self.angle)
        dx = math.cos(rad) * (GRID_SIZE // 2)
        dy = -math.sin(rad) * (GRID_SIZE // 2)
        pygame.draw.line(surface, (200, 200, 0),
                         (cx, cy),
                         (cx + dx, cy + dy), 4)


# -----------------------
# Утилиты коллизий
# -----------------------
def build_wall_coords(blocks):
    """
    Из списка Block-объектов строит множество (x,y) для fast lookup.
    """
    return {(b.x, b.y) for b in blocks}


def in_collision(x: float, y: float,
                 walls: set,
                 width: int, height: int) -> bool:
    """
    Проверка столкновений «корпусом» танка:
     1) выход за границы [0,width)×[0,height)
     2) любые из 4 клеток (floor/ceil по x,y) заняты стеной
    """
    # 1) границы
    if x < 0 or y < 0 or x >= width or y >= height:
        return True

    # 2) проверка ближайших клеток
    xs = [floor(x), ceil(x)]
    ys = [floor(y), ceil(y)]
    for xc in xs:
        for yc in ys:
            if (xc, yc) in walls:
                return True
    return False
