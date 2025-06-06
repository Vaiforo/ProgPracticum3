import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from joblib import dump

# Определение информации о признаках с типами и параметрами генерации
feature_info = {
    # Количественные признаки (координаты, радиус, скорость)
    "x1": {"type": "quant", "low": 0, "high": 50},  # x-координата первого объекта (уменьшен диапазон до [0, 50])
    "y1": {"type": "quant", "low": 0, "high": 50},  # y-координата первого объекта (уменьшен диапазон до [0, 50])
    "r1": {"type": "quant", "low": 2, "high": 12},  # радиус первого объекта (увеличен диапазон до [2, 12])
    "x2": {"type": "quant", "low": 0, "high": 50},  # x-координата второго объекта
    "y2": {"type": "quant", "low": 0, "high": 50},  # y-координата второго объекта
    "r2": {"type": "quant", "low": 2, "high": 12},  # радиус второго объекта
    "vx1": {"type": "quant", "low": -10, "high": 10},  # x-компонента скорости первого объекта
    "vy1": {"type": "quant", "low": -10, "high": 10},  # y-компонента скорости первого объекта
    "vx2": {"type": "quant", "low": -10, "high": 10},  # x-компонента скорости второго объекта
    "vy2": {"type": "quant", "low": -10, "high": 10},  # y-компонента скорости второго объекта
    # Номинальные признаки (форма, цвет)
    "shape1": {"type": "nom", "options": ["circle", "square", "triangle"]},  # форма первого объекта
    "color1": {"type": "nom", "options": ["red", "blue", "green"]},  # цвет первого объекта
    "shape2": {"type": "nom", "options": ["circle", "square", "triangle"]},  # форма второго объекта
    "color2": {"type": "nom", "options": ["red", "blue", "green"]},  # цвет второго объекта
    # Порядковые признаки (категория размера)
    "size_category1": {"type": "ord", "options": ["small", "medium", "large"]},  # размер первого объекта
    "size_category2": {"type": "ord", "options": ["small", "medium", "large"]},  # размер второго объекта
    # Бинарные признаки
    "is_moving1": {"type": "bin"},  # движется ли первый объект (0 или 1)
    "is_moving2": {"type": "bin"}  # движется ли второй объект (0 или 1)
}

# Определение наборов признаков для разных диапазонов количества признаков
set_A = ["x1", "y1", "r1", "x2", "y2", "r2", "shape1"]  # 7 признаков (диапазон 4–7)
set_B = ["x1", "y1", "r1", "x2", "y2", "r2", "shape1", "color1", "size_category1", "is_moving1"]  # 10 признаков
set_C = set_B + ["vx1", "vy1"]  # 12 признаков (диапазон 10+)


# Функция для генерации датасета
def generate_dataset(sample_size: int, feature_set: list, max_attempts: int = 10) -> pd.DataFrame:
    """
    Генерирует датасет с заданным размером выборки и набором признаков, гарантируя наличие обоих классов.

    Параметры:
    sample_size (int): Количество образцов для генерации.
    feature_set (list): Список имен признаков для включения.
    max_attempts (int): Максимальное количество попыток генерации датасета.

    Возвращает:
    pd.DataFrame: Датасет с указанными признаками и меткой Collision.

    Raises:
    ValueError: Если не удалось сгенерировать датасет с обоими классами.
    """
    for attempt in range(max_attempts):
        data = {}
        for feature in feature_set:
            info = feature_info[feature]
            if info["type"] == "quant":
                data[feature] = np.random.uniform(info["low"], info["high"], sample_size)
            elif info["type"] in ["nom", "ord"]:
                data[feature] = np.random.choice(info["options"], sample_size)
            elif info["type"] == "bin":
                data[feature] = np.random.randint(0, 2, sample_size)

        # Вычисление метки Collision
        x1 = data["x1"]
        y1 = data["y1"]
        r1 = data["r1"]
        x2 = data["x2"]
        y2 = data["y2"]
        r2 = data["r2"]
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        data["Collision"] = (distance < r1 + r2).astype(int)

        # Проверка на наличие обоих классов
        if len(np.unique(data["Collision"])) == 2:
            return pd.DataFrame(data)

        print(f"Попытка {attempt + 1}: Датасет содержит только один класс, повторная генерация...")

    raise ValueError(f"Не удалось сгенерировать датасет с обоими классами после {max_attempts} попыток")


# Функция для определения числовых и категориальных столбцов
def get_feature_types(df: pd.DataFrame) -> tuple:
    """
    Возвращает списки числовых и категориальных столбцов.

    Параметры:
    df (pd.DataFrame): DataFrame с данными.

    Возвращает:
    tuple: (num_cols, cat_cols) - списки числовых и категориальных столбцов.
    """
    num_cols = []
    cat_cols = []
    for col in df.columns:
        if col in feature_info:
            f_type = feature_info[col]["type"]
            if f_type in ["quant", "bin"]:
                num_cols.append(col)
            elif f_type in ["nom", "ord"]:
                cat_cols.append(col)
    return num_cols, cat_cols


# Основной скрипт
for sample_size in [50, 200, 700, 1500]:
    for feature_set in [set_A, set_B, set_C]:
        print(f"Датасет для размера выборки {sample_size} и {len(feature_set)} признаков:")

        # Генерация датасета с проверкой на оба класса
        try:
            df = generate_dataset(sample_size=sample_size, feature_set=feature_set)
        except ValueError as e:
            print(e)
            exit(1)

        # Разделение на признаки (X) и целевую переменную (y)
        X = df.drop("Collision", axis=1)
        y = df["Collision"]

        # Стратифицированное разделение на обучающую и тестовую выборки
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"Ошибка при разделении данных: {e}")
            exit(1)

        # Получение числовых и категориальных столбцов
        num_cols, cat_cols = get_feature_types(X_train)

        # Препроцессинг: масштабирование числовых и кодирование категориальных признаков
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(), cat_cols)
            ]
        )

        # Обучение препроцессора и преобразование данных
        preprocessor.fit(X_train)
        X_train_preprocessed = preprocessor.transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Определение моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),  # Линейная модель
            "Decision Tree": DecisionTreeClassifier(),  # Нелинейная модель
            "Random Forest": RandomForestClassifier(n_estimators=100),  # Ансамблевая модель
            "SVM": SVC()  # Модель с разделяющей гиперплоскостью
        }

        # Обучение и оценка моделей
        results = {}
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train_preprocessed, y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test_preprocessed)
            accuracy = accuracy_score(y_test, y_pred)

            results[name] = {"accuracy": accuracy, "train_time": train_time}
            print(f"{name}: Точность = {accuracy:.4f}, Время обучения = {train_time:.4f} с")

        # Выбор лучшей модели
        best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
        best_model = models[best_model_name]
        print(f"Лучшая модель: {best_model_name} с точностью {results[best_model_name]['accuracy']:.4f}")

        # Сохранение лучшей модели
        dump(best_model, f"best_model_s{sample_size}_{len(feature_set)}.joblib")

        # Вывод распределения классов для проверки
        print(f"Распределение классов в обучающей выборке: {np.bincount(y_train)}")
        print(f"Распределение классов в тестовой выборке: {np.bincount(y_test)}\n")

# Примечания:
# - Генерация датасета включает как релевантные (например, позиции и радиусы), так и нерелевантные (например, цвет) признаки.
# - Метка Collision определяется проверкой, меньше ли расстояние между центрами объектов суммы их радиусов.
# - Обучаются четыре классические модели машинного обучения: логистическая регрессия, дерево решений, случайный лес и SVM.
# - Лучшая модель выбирается по точности на тестовой выборке и сохраняется с помощью joblib.
# - Время обучения фиксируется для определения более быстрых алгоритмов.
# - Для определения алгоритмов, использующих наименьший объём данных:
#   - Простые модели, такие как логистическая регрессия и дерево решений, обычно лучше работают с меньшими наборами данных.
#   - Случайный лес и SVM могут требовать больше данных для предотвращения переобучения.
# - Код можно расширить для генерации и обработки других комбинаций датасетов, изменяя sample_size и feature_set.
