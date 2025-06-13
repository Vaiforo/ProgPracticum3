# analyzer.py
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tank_env4 import TankEnv  # импорт вашей среды
from collections import defaultdict

def plot_rewards(csv_file: str) -> None:
    """
    Считывает файл CSV с наградами и строит график зависимости награды от номера эпизода.
    """
    df = pd.read_csv(csv_file)
    plt.style.use('ggplot')  # используем доступный стиль matplotlib
    plt.figure(figsize=(12, 6))

    plt.plot(df['episode'], df['reward'], color='#2a9d8f', linewidth=2.5, label='Награда за эпизод')
    plt.fill_between(df['episode'], df['reward'], color='#2a9d8f', alpha=0.2)

    plt.xlabel('Эпизод', fontsize=14)
    plt.ylabel('Суммарная награда', fontsize=14)
    plt.title('График обучения агента', fontsize=18, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def load_qtable(pickle_file: str) -> dict:
    """
    Загружает Q-таблицу из файла pickle.
    """
    with open(pickle_file, 'rb') as file:
        q_table = pickle.load(file)
    return q_table


def choose_greedy_action(current_state: tuple, qtable: dict) -> int:
    """
    Возвращает действие с максимальным значением Q для данного состояния.
    Если состояние отсутствует в таблице, возвращает случайное действие (0).
    """
    if current_state not in qtable:
        return 0
    return int(np.argmax(qtable[current_state]))


def evaluate_agent(
    qtable: dict,
    env_builder,
    test_cases: list[tuple[tuple[int,int], tuple[int,int]]],
    runs_per_case: int = 20,
    max_step_limit: int = 200
) -> None:
    """
    Для каждого тестового случая (начальная позиция танка и цели) выполняет
    заданное количество запусков агента с жадной стратегией,
    выводит среднюю награду и долю успешных попаданий.
    """
    results_summary = []
    for tank_pos, goal_pos in test_cases:
        rewards_collected = []
        success_counter = 0

        for _ in range(runs_per_case):
            env_instance = env_builder(tank_pos, goal_pos)
            observation = env_instance.reset()
            current_state = discretize_observation(observation)
            done_flag = False
            accumulated_reward = 0.0

            for _ in range(max_step_limit):
                action = choose_greedy_action(current_state, qtable)
                observation, reward, done_flag, _ = env_instance.step(action)
                current_state = discretize_observation(observation)
                accumulated_reward += reward
                if done_flag:
                    if reward >= 100:
                        success_counter += 1
                    break

            rewards_collected.append(accumulated_reward)
            env_instance.close()

        avg_reward = np.mean(rewards_collected)
        success_rate = success_counter / runs_per_case
        results_summary.append((tank_pos, goal_pos, avg_reward, success_rate))

    print(f"{'Танк->Цель':<25} {'Средняя награда':>15} {'Успех (%)':>12}")
    for tank_pos, goal_pos, avg_r, succ_rate in results_summary:
        print(f"{tank_pos} -> {goal_pos}    {avg_r:15.1f}    {succ_rate:12.2%}")


def discretize_observation(obs: np.ndarray) -> tuple:
    """
    Преобразует наблюдение среды в дискретное состояние,
    совместимое с Q-таблицей.
    """
    x, y, angle, distance, relative_angle, up_dist, right_dist, down_dist, left_dist = obs
    return (
        int(round(x)),
        int(round(y)),
        int(angle // 15) * 15,
        int(round(distance)),
        int(round(relative_angle)),
        int(round(up_dist)),
        int(round(right_dist)),
        int(round(down_dist)),
        int(round(left_dist))
    )


if __name__ == "__main__":
    # 1. Отрисовка графика обучения
    plot_rewards("training_rewards.csv")

    # 2. Загрузка сохранённой Q-таблицы
    q_table = load_qtable("q_table.pkl")

    # 3. Фабрика создания среды с разными начальными условиями
    def environment_factory(tank_start, goal_pos):
        wall_positions = {(3,5),(4,5),(5,5),(10,2),(10,3),(10,4),(15,8),(16,8),(17,8)}
        return TankEnv(width=20, height=15,
                       wall_coords=wall_positions,
                       goal=goal_pos)

    # 4. Тестовые ситуации (позиция танка и цели)
    test_scenarios = [
        ((1,1), (18,10)),
        ((5,5), (15,12)),
        ((10,1),(2,13)),
        ((2,13),(17,3)),
    ]

    # 5. Запуск тестирования агента
    evaluate_agent(q_table, environment_factory, test_scenarios,
                   runs_per_case=30,
                   max_step_limit=200)
