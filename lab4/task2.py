import pygame
import sys
import math

from env_common import (
    GRID_SIZE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    Block,
    Goal,
    Tank,
    build_wall_coords,
    in_collision,
)

FPS = 30
MAX_STEPS = 1000

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

walls_list = [
    Block(3, 5), Block(4, 5), Block(5, 5),
    Block(10, 2), Block(10, 3), Block(10, 4),
    Block(15, 8), Block(16, 8), Block(17, 8)
]
wall_positions = build_wall_coords(walls_list)

max_grid_cols = WINDOW_WIDTH // GRID_SIZE
max_grid_rows = WINDOW_HEIGHT // GRID_SIZE

target = Goal(18, 10)
player_tank = Tank(2, 2, angle=45)

step_count = 0
game_running = True


def check_goal_reached(tank, goal):
    dist = math.hypot(tank.x - goal.x, tank.y - goal.y)
    return dist < 1


def draw_tank_with_rotation(tank, surface):
    tank_img = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(tank_img, (0, 0, 255), (0, 0, GRID_SIZE, GRID_SIZE))  
    rotated_img = pygame.transform.rotate(tank_img, -player_tank.angle)  
    rect = rotated_img.get_rect(center=(player_tank.x * GRID_SIZE + GRID_SIZE / 2,
                                        player_tank.y * GRID_SIZE + GRID_SIZE / 2))
    screen.blit(rotated_img, rect)


while game_running:
    step_count += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player_tank.angle = (player_tank.angle + 15) % 360
            elif event.key == pygame.K_RIGHT:
                player_tank.angle = (player_tank.angle - 15) % 360

            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                rad = math.radians(player_tank.angle)
                direction = 1 if event.key == pygame.K_UP else -1
                dx = int(round(math.cos(rad))) * direction
                dy = -int(round(math.sin(rad))) * direction
                new_x, new_y = player_tank.x + dx, player_tank.y + dy

                if not in_collision(new_x, new_y, wall_positions, max_grid_cols, max_grid_rows):
                    player_tank.x, player_tank.y = new_x, new_y

            elif event.key == pygame.K_r:
                player_tank.x, player_tank.y, player_tank.angle = 2, 2, 45
                step_count = 0
                print("Позиция танка сброшена")

    if check_goal_reached(player_tank, target):
        print("Цель достигнута! Поздравляем!")
        game_running = False

    if step_count >= MAX_STEPS:
        print("Достигнуто максимальное количество ходов. Игра окончена.")
        game_running = False

    screen.fill((30, 30, 30))

    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (0, y), (WINDOW_WIDTH, y))

    for block in walls_list:
        block.draw(screen)

    target.draw(screen)

    draw_tank_with_rotation(player_tank, screen)

    angle_text = font.render(f"Угол: {player_tank.angle}°", True, (255, 255, 255))
    pos_text = font.render(f"Позиция: ({player_tank.x}, {player_tank.y})", True, (255, 255, 255))
    steps_text = font.render(f"Шаги: {step_count}/{MAX_STEPS}", True, (255, 255, 255))

    screen.blit(angle_text, (10, 10))
    screen.blit(pos_text, (10, 30))
    screen.blit(steps_text, (10, 50))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
