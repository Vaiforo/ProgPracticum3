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

game_running = True

ACTIONS = {
    0: "move_forward",
    1: "move_backward",
    2: "turn_left",
    3: "turn_right",
    4: "shoot"
}


def get_distance(a_x, a_y, b_x, b_y):
    return math.hypot(a_x - b_x, a_y - b_y)


def get_relative_angle(tank_angle, tank_x, tank_y, goal_x, goal_y):
    dx = goal_x - tank_x
    dy = goal_y - tank_y
    angle_to_goal = math.degrees(math.atan2(-dy, dx)) % 360
    relative_angle = (angle_to_goal - tank_angle) % 360
    if relative_angle > 180:
        relative_angle -= 360
    return relative_angle


def check_goal_reached(tank, goal):
    dist = get_distance(tank.x, tank.y, goal.x, goal.y)
    return dist < 1


def distance_to_obstacle(tank_x, tank_y, dx, dy, wall_positions, max_cols, max_rows):
    dist = 0
    x, y = tank_x, tank_y
    while 0 <= x < max_cols and 0 <= y < max_rows:
        x += dx
        y += dy
        dist += 1
        if (x, y) in wall_positions:
            break
    return dist


def get_state(tank, goal, wall_positions, max_cols, max_rows):
    dist_goal = get_distance(tank.x, tank.y, goal.x, goal.y)
    rel_angle = get_relative_angle(tank.angle, tank.x, tank.y, goal.x, goal.y)
    obs_front = distance_to_obstacle(tank.x, tank.y,
                                     int(round(math.cos(math.radians(tank.angle)))),
                                     -int(round(math.sin(math.radians(tank.angle)))),
                                     wall_positions, max_cols, max_rows)
    obs_left = distance_to_obstacle(tank.x, tank.y,
                                    int(round(math.cos(math.radians(tank.angle + 90)))),
                                    -int(round(math.sin(math.radians(tank.angle + 90)))),
                                    wall_positions, max_cols, max_rows)
    obs_right = distance_to_obstacle(tank.x, tank.y,
                                     int(round(math.cos(math.radians(tank.angle - 90)))),
                                     -int(round(math.sin(math.radians(tank.angle - 90)))),
                                     wall_positions, max_cols, max_rows)
    obs_back = distance_to_obstacle(tank.x, tank.y,
                                    int(round(math.cos(math.radians(tank.angle + 180)))),
                                    -int(round(math.sin(math.radians(tank.angle + 180)))),
                                    wall_positions, max_cols, max_rows)

    return [tank.x, tank.y, tank.angle, dist_goal, rel_angle, obs_front, obs_left, obs_right, obs_back]


def calculate_reward(prev_dist, current_dist, collision, reached_goal):
    reward = -1
    if collision:
        reward -= 10
    if reached_goal:
        reward += 100
    elif current_dist < prev_dist:
        reward += 10
    else:
        reward -= 5
    return reward


def move_tank(tank, action, wall_positions, max_cols, max_rows):
    collision = False
    if action == "turn_left":
        tank.angle = (tank.angle + 15) % 360
    elif action == "turn_right":
        tank.angle = (tank.angle - 15) % 360
    elif action == "move_forward" or action == "move_backward":
        rad = math.radians(tank.angle)
        direction = 1 if action == "move_forward" else -1
        dx = int(round(math.cos(rad))) * direction
        dy = -int(round(math.sin(rad))) * direction
        nx, ny = tank.x + dx, tank.y + dy
        if 0 <= nx < max_cols and 0 <= ny < max_rows and not in_collision(nx, ny, wall_positions, max_cols, max_rows):
            tank.x, tank.y = nx, ny
        else:
            collision = True
    elif action == "shoot":
        pass


def draw_environment():
    screen.fill((30, 30, 30))

    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (0, y), (WINDOW_WIDTH, y))

    for block in walls_list:
        block.draw(screen)

    target.draw(screen)

    tank_img = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(tank_img, (0, 0, 255), (0, 0, GRID_SIZE, GRID_SIZE))
    rotated_img = pygame.transform.rotate(tank_img, -player_tank.angle)
    rect = rotated_img.get_rect(center=(player_tank.x * GRID_SIZE + GRID_SIZE / 2,
                                        player_tank.y * GRID_SIZE + GRID_SIZE / 2))
    screen.blit(rotated_img, rect)

    state = get_state(player_tank, target, wall_positions,
                      max_grid_cols, max_grid_rows)
    state_text = font.render(
        f"State: {['{:.1f}'.format(s) for s in state]}", True, (255, 255, 255))
    screen.blit(state_text, (10, WINDOW_HEIGHT - 30))

    pygame.display.flip()


prev_distance = get_distance(player_tank.x, player_tank.y, target.x, target.y)

while game_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False
        elif event.type == pygame.KEYDOWN:
            action = None
            if event.key == pygame.K_LEFT:
                action = ACTIONS[2]
            elif event.key == pygame.K_RIGHT:
                action = ACTIONS[3]
            elif event.key == pygame.K_UP:
                action = ACTIONS[0]
            elif event.key == pygame.K_DOWN:
                action = ACTIONS[1]
            elif event.key == pygame.K_SPACE:
                action = ACTIONS[4]

            if action:
                collision = move_tank(
                    player_tank, action, wall_positions, max_grid_cols, max_grid_rows)
                current_distance = get_distance(
                    player_tank.x, player_tank.y, target.x, target.y)
                reached = check_goal_reached(player_tank, target)
                reward = calculate_reward(
                    prev_distance, current_distance, collision, reached)
                prev_distance = current_distance
                print(
                    f"Action: {action}, Reward: {reward:.2f}, Collision: {collision}, Reached: {reached}")
                if reached:
                    print("Цель достигнута! Поздравляем!")
                    game_running = False

    draw_environment()
    clock.tick(FPS)

pygame.quit()
sys.exit()
