import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import os

# Импорт из ES-HyperNEAT (предполагается, что установлен)
from ES_HyperNEAT.hyperneat import HyperNEAT
from ES_HyperNEAT.substrate import Substrate

# Функция генерации датасета (как раньше)
def generate_feature(feature_type, length):
    if feature_type == "binary":
        return np.random.choice([0, 1], size=length)
    elif feature_type == "nominal":
        return np.random.choice(['A', 'B', 'C', 'D'], size=length)
    elif feature_type == "ordinal":
        return np.random.choice([1, 2, 3, 4, 5], size=length)
    elif feature_type == "quantitative":
        return np.random.uniform(0, 100, size=length)
    else:
        raise ValueError("Unknown feature type")

def create_dataset(num_features, num_samples):
    feature_types = ["binary", "nominal", "ordinal", "quantitative"]
    chosen_types = random.sample(feature_types, k=min(4, num_features))
    while len(chosen_types) < num_features:
        chosen_types.append(random.choice(feature_types))
    random.shuffle(chosen_types)

    data = {}
    for i in range(num_features):
        data[f"Obj1_Feat{i+1}"] = generate_feature(chosen_types[i], num_samples)
    for i in range(num_features):
        data[f"Obj2_Feat{i+1}"] = generate_feature(chosen_types[i], num_samples)

    labels = []
    for idx in range(num_samples):
        matches = sum(
            data[f"Obj1_Feat{f}"][idx] == data[f"Obj2_Feat{f}"][idx]
            for f in range(1, num_features+1)
        )
        labels.append(1 if matches >= num_features // 2 else 0)  # 1=Yes, 0=No
    return pd.DataFrame(data), np.array(labels)

# Настройка substrate — пример для ES-HyperNEAT
def create_substrate(input_dims, output_dims=1):
    # Входной слой — например, 2D решётка из признаков (зависит от задачи)
    # Для простоты используем одномерную проекцию: (input_dims, 1, 1)
    input_coordinates = [(i, 0, 0) for i in range(input_dims)]

    # Выходной слой — 1 нейрон
    output_coordinates = [(0, 0, 0)]

    return Substrate(input_coordinates, output_coordinates)

# Функция для оценки модели (F1)
def evaluate_hyperneat_model(hyperneat, X_val, y_val):
    preds = []
    for x in X_val:
        output = hyperneat.feed_forward(x)
        preds.append(1 if output[0] > 0.5 else 0)
    return f1_score(y_val, preds)

# Основная функция запуска
def run_es_hyperneat(X_train, y_train, X_val, y_val, input_dim):
    # Создаём substrate с размером входа input_dim
    substrate = create_substrate(input_dim)

    # Настройки HyperNEAT (здесь базовые, можно тонко настраивать)
    hyperneat = HyperNEAT(
        substrate=substrate,
        population_size=50,
        max_generations=50,
        verbosity=2,
    )

    # Запуск эволюции
    winner = hyperneat.run(X_train, y_train)

    # Оценка модели
    f1 = evaluate_hyperneat_model(hyperneat, X_val, y_val)
    print(f"F1-score модели ES-HyperNEAT: {f1:.4f}")

    return winner, hyperneat

# Путь для сохранения моделей
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Генерируем данные для малого датасета
df_small, labels_small = create_dataset(num_features=4, num_samples=30)
X_small = df_small.values
y_small = labels_small
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_small, y_small, test_size=0.2, random_state=42)

# Генерируем данные для большого датасета
df_big, labels_big = create_dataset(num_features=15, num_samples=1500)
X_big = df_big.values
y_big = labels_big
X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

# Запускаем ES-HyperNEAT на малом датасете
winner_s, model_s = run_es_hyperneat(X_train_s, y_train_s, X_val_s, y_val_s, input_dim=X_train_s.shape[1])
pickle.dump((winner_s, model_s), open(f"{save_dir}/es_hyperneat_small.pkl", "wb"))

# Запускаем ES-HyperNEAT на большом датасете
winner_b, model_b = run_es_hyperneat(X_train_b, y_train_b, X_val_b, y_val_b, input_dim=X_train_b.shape[1])
pickle.dump((winner_b, model_b), open(f"{save_dir}/es_hyperneat_big.pkl", "wb"))
