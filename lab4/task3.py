import numpy as np
import pandas as pd
import random
import pickle
from collections import defaultdict
from tank_env import TankEnv

TOTAL_EPISODES = 500
MAX_STEPS_PER_EPISODE = 200
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_FACTOR = 0.995


def discretize_observation(observation: np.ndarray) -> tuple:
    pos_x, pos_y, ang, dist_to_goal, rel_angle, dist_up, dist_right, dist_down, dist_left = observation

    discretized_state = (
        int(round(pos_x)),
        int(round(pos_y)),
        int(ang // 15) * 15,
        int(round(dist_to_goal)),
        int(round(rel_angle)),
        int(round(dist_up)),
        int(round(dist_right)),
        int(round(dist_down)),
        int(round(dist_left)),
    )
    return discretized_state


def select_action(state: tuple, q_values: dict, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, 4)
    return int(np.argmax(q_values[state]))


def run_training():
    wall_cells = {(3, 5), (4, 5), (5, 5), (10, 2), (10, 3),
                  (10, 4), (15, 8), (16, 8), (17, 8)}
    goal_position = (18, 10)
    environment = TankEnv(width=20, height=15,
                          wall_coords=wall_cells, goal=goal_position)

    Q_table = defaultdict(lambda: np.zeros(
        environment.action_space.n, dtype=np.float32))

    epsilon = EPSILON_START
    rewards_log = []

    for episode in range(1, TOTAL_EPISODES + 1):
        observation = environment.reset()
        current_state = discretize_observation(observation)
        cumulative_reward = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = select_action(current_state, Q_table, epsilon)
            next_observation, reward, done, _ = environment.step(action)
            next_state = discretize_observation(next_observation)

            max_future_q = np.max(Q_table[next_state])
            current_q = Q_table[current_state][action]
            new_q = current_q + LEARNING_RATE * \
                (reward + DISCOUNT_FACTOR * max_future_q - current_q)
            Q_table[current_state][action] = new_q

            current_state = next_state
            cumulative_reward += reward

            if done:
                break

        rewards_log.append(cumulative_reward)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY_FACTOR)

        if episode % 50 == 0:
            print(
                f"Эпизод {episode:3d} | Награда: {cumulative_reward:6.1f} | ε={epsilon:.3f}")

    rewards_df = pd.DataFrame({
        "episode": np.arange(1, TOTAL_EPISODES + 1),
        "reward": rewards_log
    })
    rewards_df.to_csv("training_rewards.csv", index=False)

    with open("q_table.pkl", "wb") as file:
        pickle.dump(dict(Q_table), file)

    print("Обучение завершено, результаты сохранены.")


if __name__ == "__main__":
    run_training()
