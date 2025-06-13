# Импортируем необходимые модули
import numpy as np  # Для работы с массивами и генерации случайных данных
import pandas as pd  # Для работы с табличными данными (DataFrame)
import random  # Для генерации случайных чисел
from itertools import product  # Для создания комбинаций параметров
from pathlib import Path  # Для удобной работы с путями файловой системы

# --- Определяем текущую директорию ---
current_folder = Path(__file__).resolve().parent  # Получаем папку, где находится скрипт
save_folder = current_folder / "datasets_storage"  # Создаем путь к папке для сохранения

# --- Настройки генерации ---
# Диапазоны для количества строк в датасетах
rows_ranges = [
    (30, 100),  # Маленькие датасеты
    (100, 500),  # Средние датасеты
    (500, 1000),  # Большие датасеты
    (1000, 1500),  # Очень большие датасеты
]

# Диапазоны для количества признаков
features_ranges = [
    (4, 7),  # Малое количество признаков
    (8, 10),  # Среднее количество признаков
    (11, 15),  # Большое количество признаков
]

# Возможные типы признаков
types_of_features = ["binary", "nominal", "ordinal", "quantitative"]


def create_feature(ftype: str, length: int):
    """
    Генерирует массив значений признака заданного типа.

    Параметры:
        ftype: тип признака (binary, nominal, ordinal, quantitative)
        length: количество значений в массиве

    Возвращает:
        np.ndarray с сгенерированными значениями
    """
    if ftype == "binary":
        return np.random.choice([0, 1], size=length)  # Случайные 0 и 1
    elif ftype == "nominal":
        return np.random.choice(['A', 'B', 'C', 'D'], size=length)  # Категории без порядка
    elif ftype == "ordinal":
        return np.random.choice([1, 2, 3, 4, 5], size=length)  # Категории с порядком
    elif ftype == "quantitative":
        return np.random.uniform(0, 100, size=length)  # Числа от 0 до 100
    else:
        raise ValueError("Неподдерживаемый тип признака")


def create_sample_dataset(feature_num: int, sample_num: int) -> pd.DataFrame:
    """
    Создает датасет для сравнения двух объектов.

    Параметры:
        feature_num: количество признаков у каждого объекта
        sample_num: количество строк в датасете

    Возвращает:
        pd.DataFrame с данными и меткой коллизии
    """
    # Выбираем случайные типы признаков
    chosen_types = random.sample(types_of_features, k=min(4, feature_num))
    # Дополняем до нужного количества признаков
    while len(chosen_types) < feature_num:
        chosen_types.append(random.choice(types_of_features))
    random.shuffle(chosen_types)  # Перемешиваем типы

    dataset = {}  # Словарь для хранения данных

    # Генерируем признаки для первого объекта
    for idx in range(feature_num):
        dataset[f"Obj1_Feat{idx + 1}"] = create_feature(chosen_types[idx], sample_num)

    # Генерируем признаки для второго объекта
    for idx in range(feature_num):
        dataset[f"Obj2_Feat{idx + 1}"] = create_feature(chosen_types[idx], sample_num)

    # Определяем коллизии (совпадения признаков)
    collision_labels = []
    for i in range(sample_num):
        # Считаем количество совпавших признаков
        equal_feats = sum(
            dataset[f"Obj1_Feat{j + 1}"][i] == dataset[f"Obj2_Feat{j + 1}"][i]
            for j in range(feature_num)
        )
        # Если совпало больше половины - считаем коллизией
        collision_labels.append("Да" if equal_feats >= feature_num // 2 else "Нет")

    dataset["Collision"] = collision_labels  # Добавляем метки

    return pd.DataFrame(dataset)  # Преобразуем в DataFrame


# --- Основной процесс генерации ---
all_datasets = []  # Список для хранения всех датасетов
set_id = 1  # Счетчик датасетов

# Создаем все комбинации параметров
for row_rng, feat_rng in product(rows_ranges, features_ranges):
    # Генерируем случайные параметры из диапазонов
    samples_count = random.randint(*row_rng)  # Количество строк
    features_count = random.randint(*feat_rng)  # Количество признаков

    # Создаем датасет
    dataset_df = create_sample_dataset(features_count, samples_count)
    all_datasets.append((set_id, samples_count, features_count, dataset_df))

    # Выводим информацию о датасете
    print(f"\n--- Набор данных {set_id} | Строк: {samples_count} | Признаков: {features_count} ---")
    print(dataset_df.head(10))  # Показываем первые 10 строк
    set_id += 1

# --- Сохранение результатов ---
save_answer = input("\nСохранить все датасеты в CSV? (да/нет): ").strip().lower()

if save_answer in ['да', 'yes', 'y']:
    save_folder.mkdir(exist_ok=True)  # Создаем папку, если не существует
    for ds_id, rows, feats, df in all_datasets:
        # Формируем имя файла
        path_to_save = save_folder / f"dataset_{ds_id}_rows{rows}_feats{feats}.csv"
        df.to_csv(path_to_save, index=False)  # Сохраняем без индексов
        print(f"Сохранено: {path_to_save}")
else:
    print("Сохранение отменено.")