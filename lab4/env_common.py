import math
from math import floor, ceil
import pygame

GRID_SIZE = 40
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class Block:
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def draw(self, surface: pygame.Surface) -> None:
        rect = pygame.Rect(self.x * GRID_SIZE,
                           self.y * GRID_SIZE,
                           GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, (80, 80, 80), rect)


class Goal:
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def draw(self, surface: pygame.Surface) -> None:
        cx = self.x * GRID_SIZE + GRID_SIZE // 2
        cy = self.y * GRID_SIZE + GRID_SIZE // 2
        radius = GRID_SIZE // 3
        pygame.draw.circle(surface, (0, 200, 0), (cx, cy), radius)


class Tank:
    def __init__(self, x: int, y: int, angle: float = 0.0) -> None:
        self.x, self.y = x, y
        self.angle = angle

    def draw(self, surface: pygame.Surface) -> None:
        cx = self.x * GRID_SIZE + GRID_SIZE // 2
        cy = self.y * GRID_SIZE + GRID_SIZE // 2

        body = pygame.Surface((GRID_SIZE, GRID_SIZE // 2), pygame.SRCALPHA)
        body.fill((200, 200, 0))
        rotated = pygame.transform.rotate(body, -self.angle)
        rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rect.topleft)

        rad = math.radians(self.angle)
        dx = math.cos(rad) * (GRID_SIZE // 2)
        dy = -math.sin(rad) * (GRID_SIZE // 2)
        pygame.draw.line(surface, (200, 200, 0),
                         (cx, cy),
                         (cx + dx, cy + dy), 4)


def build_wall_coords(blocks):
    return {(b.x, b.y) for b in blocks}


def in_collision(x: float, y: float,
                 walls: set,
                 width: int, height: int) -> bool:
    if x < 0 or y < 0 or x >= width or y >= height:
        return True

    xs = [floor(x), ceil(x)]
    ys = [floor(y), ceil(y)]
    for xc in xs:
        for yc in ys:
            if (xc, yc) in walls:
                return True
    return False
