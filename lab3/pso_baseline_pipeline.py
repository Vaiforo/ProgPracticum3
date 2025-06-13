# Импорт необходимых библиотек
import numpy as np  # Для работы с массивами и математическими операциями
import pandas as pd  # Для работы с табличными данными
import random  # Для генерации случайных чисел
from pathlib import Path  # Для удобной работы с путями файловой системы

# Импорт функций из scikit-learn
from sklearn.model_selection import train_test_split  # Для разделения данных
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Для кодирования и масштабирования
from sklearn.linear_model import LogisticRegression  # Модель логистической регрессии
from sklearn.tree import DecisionTreeClassifier  # Модель дерева решений
from sklearn.metrics import f1_score  # Метрика оценки качества

# Импорт библиотеки для оптимизации роевым методом (PSO)
import pyswarms as ps  # Алгоритм оптимизации Particle Swarm Optimization
import pickle  # Для сохранения и загрузки моделей


# Функция генерации признаков заданного типа
def generate_feature(feature_type: str, length: int) -> np.ndarray:
    """Генерирует массив значений признака заданного типа"""
    if feature_type == "binary":
        return np.random.choice([0, 1], size=length)  # Бинарные значения (0/1)
    elif feature_type == "nominal":
        return np.random.choice(['A', 'B', 'C', 'D'], size=length)  # Категориальные
    elif feature_type == "ordinal":
        return np.random.choice([1, 2, 3, 4, 5], size=length)  # Порядковые
    elif feature_type == "quantitative":
        return np.random.uniform(0, 100, size=length)  # Количественные (0-100)
    else:
        raise ValueError("Неизвестный тип признака")  # Ошибка при неверном типе


# Функция создания датасета
def create_dataset(num_features: int, num_samples: int) -> pd.DataFrame:
    """Создает датасет с заданным числом признаков и образцов"""
    feature_types = ["binary", "nominal", "ordinal", "quantitative"]  # Доступные типы
    chosen_types = random.sample(feature_types, k=min(4, num_features))  # Выбираем случайные типы

    # Дополняем список типов, если нужно больше признаков
    while len(chosen_types) < num_features:
        chosen_types.append(random.choice(feature_types))
    random.shuffle(chosen_types)  # Перемешиваем типы

    data = {}  # Словарь для хранения данных
    # Генерируем признаки для первого объекта
    for i in range(num_features):
        data[f"Obj1_Feat{i + 1}"] = generate_feature(chosen_types[i], num_samples)
    # Генерируем признаки для второго объекта
    for i in range(num_features):
        data[f"Obj2_Feat{i + 1}"] = generate_feature(chosen_types[i], num_samples)

    # Создаем метки (Yes если >= половины признаков совпадают, иначе No)
    labels = []
    for idx in range(num_samples):
        matches = sum(
            data[f"Obj1_Feat{f}"][idx] == data[f"Obj2_Feat{f}"][idx]
            for f in range(1, num_features + 1)
        )
        labels.append("Yes" if matches >= num_features // 2 else "No")
    data["Collision"] = labels  # Добавляем метки в датасет
    return pd.DataFrame(data)  # Возвращаем DataFrame


# Функция оценки качества для PSO
def pso_fitness(params, model_class, X_train, y_train, X_val, y_val):
    """Вычисляет качество модели для оптимизации PSO"""
    n_particles = params.shape[0]  # Число частиц в рое
    fitness = np.zeros(n_particles)  # Массив для хранения оценок

    for i in range(n_particles):
        p = params[i]  # Параметры текущей частицы
        if model_class == LogisticRegression:
            C = p[0]  # Параметр регуляризации
            max_iter = int(p[1])  # Максимальное число итераций
            max_iter = max(max_iter, 1000)  # Гарантируем минимум 1000 итераций
            model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
        elif model_class == DecisionTreeClassifier:
            max_depth = int(p[0])  # Максимальная глубина дерева
            min_samples_split = int(p[1])  # Минимальное число образцов для разделения
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        else:
            fitness[i] = 1.0  # Если модель неизвестна, возвращаем худший результат
            continue

        # Проверка, что есть оба класса
        if len(np.unique(y_train)) < 2:
            fitness[i] = 1.0
            continue

        try:
            model.fit(X_train, y_train)  # Обучаем модель
            preds = model.predict(X_val)  # Прогнозируем на валидации
            score = f1_score(y_val, preds)  # Вычисляем F1-score
            fitness[i] = -score  # Минимизируем -F1 (PSO минимизирует функцию)
        except Exception:
            fitness[i] = 1.0  # При ошибке возвращаем худший результат

    return fitness  # Возвращаем массив оценок


# Функция оптимизации гиперпараметров через PSO
def optimize_pso(model_class, X_train, y_train, X_val, y_val, bounds, options, iters=50):
    """Оптимизирует гиперпараметры модели с помощью PSO"""
    # Инициализируем оптимизатор PSO
    optimizer = ps.single.GlobalBestPSO(
        n_particles=20,  # Число частиц в рое
        dimensions=len(bounds[0]),  # Размерность пространства параметров
        options=options,  # Параметры алгоритма PSO
        bounds=bounds  # Границы пространства поиска
    )

    # Запускаем оптимизацию
    best_cost, best_pos = optimizer.optimize(
        pso_fitness, iters,  # Функция оценки и число итераций
        model_class=model_class, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val  # Дополнительные аргументы для pso_fitness
    )

    print(f"Лучшие параметры (PSO): {best_pos}, F1: {-best_cost:.4f}")
    return best_pos, -best_cost  # Возвращаем лучшие параметры и F1-score


# Основной блок выполнения
if __name__ == "__main__":
    # Параметры генерации данных
    num_features = 6  # Число признаков у каждого объекта
    num_samples = 200  # Число образцов в датасете

    # Генерируем датасет
    data = create_dataset(num_features, num_samples)
    print(data.head())  # Выводим первые строки датасета

    # Разделяем на признаки (X) и целевую переменную (y)
    X = data.drop(columns=["Collision"])
    y = data["Collision"].map({"Yes": 1, "No": 0})  # Преобразуем метки в числа

    # Кодируем категориальные признаки
    for col in X.columns:
        if X[col].dtype == object:  # Если признак категориальный
            X[col] = LabelEncoder().fit_transform(X[col])  # Кодируем числами

    # Масштабируем количественные признаки
    quant_cols = [col for col in X.columns if X[col].dtype in [np.float64, np.float32]]
    scaler = StandardScaler()  # Инициализируем стандартизатор
    if quant_cols:
        X[quant_cols] = scaler.fit_transform(X[quant_cols])  # Масштабируем

    # Разбиваем данные на обучающую, валидационную и тестовую выборки
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    # Выводим распределение классов
    print("Распределение классов в полном тренировочном наборе:", np.unique(y_train_full, return_counts=True))
    print("Распределение классов в тренировочном наборе:", np.unique(y_train, return_counts=True))

    # Задаем диапазоны для оптимизации гиперпараметров
    logreg_bounds = (np.array([0.001, 100]), np.array([10.0, 1500]))  # Для логистической регрессии
    dtree_bounds = (np.array([1, 2]), np.array([20, 20]))  # Для дерева решений

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Параметры PSO

    # Оптимизируем логистическую регрессию
    best_logreg_params, best_logreg_score = optimize_pso(
        LogisticRegression, X_train, y_train, X_val, y_val,
        bounds=logreg_bounds, options=options, iters=50
    )

    # Оптимизируем дерево решений
    best_dtree_params, best_dtree_score = optimize_pso(
        DecisionTreeClassifier, X_train, y_train, X_val, y_val,
        bounds=dtree_bounds, options=options, iters=50
    )

    # Проверяем распределение классов перед финальным обучением
    print("Распределение классов перед финальным обучением:", np.unique(y_train_full, return_counts=True))
    if len(np.unique(y_train_full)) < 2:
        raise ValueError("Ошибка: в полном тренировочном наборе только один класс!")

    # Обучаем финальные модели на всех обучающих данных
    final_logreg = LogisticRegression(
        C=best_logreg_params[0],  # Оптимальный C
        max_iter=int(max(best_logreg_params[1], 1000)),  # Оптимальное число итераций
        solver='lbfgs'  # Алгоритм оптимизации
    )
    final_logreg.fit(X_train_full, y_train_full)  # Обучаем на всех данных

    final_dtree = DecisionTreeClassifier(
        max_depth=int(best_dtree_params[0]),  # Оптимальная глубина
        min_samples_split=int(best_dtree_params[1])  # Оптимальный параметр разделения
    )
    final_dtree.fit(X_train_full, y_train_full)  # Обучаем на всех данных

    # Сохраняем обученные модели
    save_dir = Path("saved_models")  # Путь к директории
    save_dir.mkdir(exist_ok=True)  # Создаем директорию, если не существует

    # Сохраняем модель логистической регрессии
    with open(save_dir / "best_logreg_pso_model.pkl", "wb") as f:
        pickle.dump(final_logreg, f)

    # Сохраняем модель дерева решений
    with open(save_dir / "best_dtree_pso_model.pkl", "wb") as f:
        pickle.dump(final_dtree, f)

    print("Модели успешно сохранены в 'saved_models'.")