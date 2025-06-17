import math
import numpy as np
import gym
from gym import spaces

from env_common import in_collision


class TankEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        width: int,
        height: int,
        wall_coords: set[tuple[int, int]],
        goal: tuple[int, int],
        step_size: float = 1.0,
        turn_deg: float = 15.0
    ) -> None:
        super().__init__()

        self.width, self.height = width, height
        self.walls = wall_coords
        self.goal = goal

        self.action_space = spaces.Discrete(5)

        high = np.array([
            width,
            height,
            360.0,
            math.hypot(width, height),
            180.0,
            max(width, height),
            max(width, height),
            max(width, height),
            max(width, height),
        ], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0, 0.0, -180.0, 0.0,
                       0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.step_size = step_size
        self.turn_deg = turn_deg

        self.reset()

    def reset(self) -> np.ndarray:
        self.tank_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.tank_angle = 0.0
        self.prev_dist = self._dist_to_goal()
        self.done = False
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action), "Недопустимое действие"
        reward = -1.0

        if action == 0:
            dx, dy = self._heading_vector()
            new_pos = self.tank_pos + self.step_size * \
                np.array([dx, dy], dtype=np.float32)
            if in_collision(new_pos[0], new_pos[1],
                            self.walls, self.width, self.height):
                reward -= 10.0
            else:
                self.tank_pos = new_pos

        elif action == 1:
            dx, dy = self._heading_vector()
            new_pos = self.tank_pos - self.step_size * \
                np.array([dx, dy], dtype=np.float32)
            if in_collision(new_pos[0], new_pos[1],
                            self.walls, self.width, self.height):
                reward -= 10.0
            else:
                self.tank_pos = new_pos

        elif action == 2:
            self.tank_angle = (self.tank_angle + self.turn_deg) % 360.0

        elif action == 3:
            self.tank_angle = (self.tank_angle - self.turn_deg) % 360.0

        elif action == 4:
            if self._hit_goal():
                reward += 100.0
                self.done = True

        dist = self._dist_to_goal()
        if dist < self.prev_dist:
            reward += 10.0
        elif dist > self.prev_dist:
            reward -= 5.0
        self.prev_dist = dist

        if not self.done and self._in_goal():
            reward += 100.0
            self.done = True

        obs = self._get_state()
        return obs, reward, self.done, {}

    def _heading_vector(self) -> tuple[float, float]:
        rad = math.radians(self.tank_angle)
        return math.cos(rad), -math.sin(rad)

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm(self.tank_pos - np.array(self.goal, dtype=np.float32)))

    def _angle_to_goal(self) -> float:
        vec = np.array(self.goal, dtype=np.float32) - self.tank_pos
        goal_ang = (math.degrees(math.atan2(-vec[1], vec[0])) + 360.0) % 360.0
        diff = (goal_ang - self.tank_angle + 180.0) % 360.0 - 180.0
        return diff

    def _hit_goal(self) -> bool:
        return abs(self._angle_to_goal()) < self.turn_deg/2 and self._dist_to_goal() < 2*self.step_size

    def _in_goal(self) -> bool:
        return self._dist_to_goal() < 1.0

    def _ray_dist(self, dx: int, dy: int) -> float:
        steps = 0
        x, y = self.tank_pos
        while True:
            x += dx
            y += dy
            steps += 1
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or yi < 0 or xi >= self.width or yi >= self.height or (xi, yi) in self.walls:
                return float(steps)

    def _get_state(self) -> np.ndarray:
        dist = self._dist_to_goal()
        rel_ang = self._angle_to_goal()
        up = self._ray_dist(0, -1)
        right = self._ray_dist(1,  0)
        down = self._ray_dist(0,  1)
        left = self._ray_dist(-1, 0)
        return np.array([
            self.tank_pos[0], self.tank_pos[1],
            self.tank_angle,
            dist, rel_ang,
            up, right, down, left
        ], dtype=np.float32)

    def render(self, mode='human') -> None:
        print(f"Tank at ({self.tank_pos[0]:.1f},{self.tank_pos[1]:.1f}) "
              f"↻{self.tank_angle:.1f}° dist={self._dist_to_goal():.1f}")

    def close(self) -> None:
        pass


if __name__ == "__main__":
    walls = [(3, 5), (4, 5), (5, 5), (10, 2), (10, 3),
             (10, 4), (15, 8), (16, 8), (17, 8)]
    goal = (18, 10)
    env = TankEnv(20, 15, walls, goal)

    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print("Episode finished, total reward:", total_reward)
