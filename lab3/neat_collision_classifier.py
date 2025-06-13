# Импорт необходимых библиотек
import neat  # Библиотека NEAT (нейроэволюция)
import numpy as np  # Для работы с массивами
import pandas as pd  # Для работы с таблицами
import random  # Для генерации случайных чисел
import pickle  # Для сохранения моделей
from sklearn.preprocessing import LabelEncoder  # Кодирование категориальных признаков
from sklearn.metrics import f1_score  # Метрика оценки
from sklearn.model_selection import train_test_split  # Разделение данных
from pathlib import Path  # Работа с путями файловой системы


# Функция генерации признаков (дублируется для полноты)
def generate_feature(feature_type: str, length: int):
    """Генерирует массив значений признака заданного типа"""
    if feature_type == "binary":
        return np.random.choice([0, 1], size=length)  # Бинарные значения
    elif feature_type == "nominal":
        return np.random.choice(['A', 'B', 'C', 'D'], size=length)  # Категории
    elif feature_type == "ordinal":
        return np.random.choice([1, 2, 3, 4, 5], size=length)  # Порядковые
    elif feature_type == "quantitative":
        return np.random.uniform(0, 100, size=length)  # Числовые
    else:
        raise ValueError("Unknown feature type")  # Ошибка при неизвестном типе


# Функция создания датасета (дублируется для полноты)
def create_dataset(num_features: int, num_samples: int) -> pd.DataFrame:
    """Создает датасет с заданным числом признаков и образцов"""
    feature_types = ["binary", "nominal", "ordinal", "quantitative"]
    chosen_types = random.sample(feature_types, k=min(4, num_features))
    while len(chosen_types) < num_features:
        chosen_types.append(random.choice(feature_types))
    random.shuffle(chosen_types)

    data = {}
    # Генерация признаков для первого объекта
    for i in range(num_features):
        data[f"Obj1_Feat{i + 1}"] = generate_feature(chosen_types[i], num_samples)
    # Генерация признаков для второго объекта
    for i in range(num_features):
        data[f"Obj2_Feat{i + 1}"] = generate_feature(chosen_types[i], num_samples)

    # Создание меток (Yes/No) на основе совпадений признаков
    labels = []
    for idx in range(num_samples):
        matches = sum(
            data[f"Obj1_Feat{f}"][idx] == data[f"Obj2_Feat{f}"][idx]
            for f in range(1, num_features + 1)
        )
        labels.append("Yes" if matches >= num_features // 2 else "No")
    data["Collision"] = labels
    return pd.DataFrame(data)


# Функция подготовки данных для обучения
def prepare_dataset(num_features, num_samples):
    """Подготавливает данные для нейросети"""
    df = create_dataset(num_features, num_samples)  # Создаем датасет
    X = df.drop(columns=["Collision"])  # Признаки
    y = df["Collision"].map({"Yes": 1, "No": 0})  # Преобразуем метки в числа

    # Кодируем категориальные признаки
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col])
    return X.values, y.values  # Возвращаем numpy массивы


# Функция оценки генома (индивидуального решения)
def eval_genome(genome, config, X_train, y_train, X_val, y_val):
    """Оценивает качество генома на валидационных данных"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)  # Создаем сеть
    predictions = []
    for xi in X_val:
        output = net.activate(xi)  # Получаем предсказание
        predictions.append(1 if output[0] > 0.5 else 0)  # Бинарная классификация
    return f1_score(y_val, predictions)  # Возвращаем F1-score


# Основная функция запуска NEAT
def run_neat(X_train, y_train, X_val, y_val, config_file, generations=50):
    """Запускает алгоритм NEAT для обучения"""
    print(f"Размер данных: {X_train.shape[1]} признаков")

    # Загружаем конфигурацию NEAT из файла
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Создаем популяцию
    population = neat.Population(config)

    # Добавляем отчеты для вывода прогресса
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Функция оценки всех геномов в популяции
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = eval_genome(genome, config, X_train, y_train, X_val, y_val)

    # Запускаем эволюцию
    winner = population.run(eval_genomes, generations)
    return winner, config


# Функция сохранения обученной модели
def save_neat_model(winner, config, filename):
    """Сохраняет лучший геном и конфигурацию в файл"""
    with open(filename, "wb") as f:
        pickle.dump((winner, config), f)


# Основной блок выполнения
if __name__ == "__main__":
    # Подготовка малого датасета (4 признака на объект, всего 8)
    X_small, y_small = prepare_dataset(num_features=4, num_samples=30)
    X_small = np.hstack([X_small[:, :4], X_small[:, 4:8]])  # Объединяем признаки

    # Подготовка большого датасета (15 признаков на объект, всего 30)
    X_big, y_big = prepare_dataset(num_features=15, num_samples=1500)

    # Разделение на обучающую и валидационную выборки
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_small, y_small, test_size=0.2, random_state=42)
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

    # Пути к файлам конфигурации NEAT
    config_small_path = "config_small"  # Для 8 входных признаков
    config_big_path = "config_big"  # Для 30 входных признаков

    # Запуск NEAT для малого датасета
    winner_small, config_small = run_neat(X_train_s, y_train_s, X_val_s, y_val_s, config_small_path)
    print("NEAT для малого датасета завершён.")

    # Запуск NEAT для большого датасета
    winner_big, config_big = run_neat(X_train_b, y_train_b, X_val_b, y_val_b, config_big_path)
    print("NEAT для большого датасета завершён.")

    # Сохранение обученных моделей
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)  # Создаем директорию, если не существует

    # Сохраняем модель для малого датасета
    save_neat_model(winner_small, config_small, save_dir / "neat_small.pkl")
    # Сохраняем модель для большого датасета
    save_neat_model(winner_big, config_big, save_dir / "neat_big.pkl")

    print("Модели NEAT сохранены.")