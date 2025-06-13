# --- Импорт необходимых библиотек ---
import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# --- Пути к папкам ---
datasets_dir = Path("datasets_storage")          # папка с 12 датасетами
output_dir = Path("learn_models")                 # папка для сохранения обученных моделей
output_dir.mkdir(parents=True, exist_ok=True)     # создать папку, если не существует

# --- Вопрос: нужно ли сохранять обученные модели ---
save_models = input("\nСохранить модели в файлы? (да/нет): ").strip().lower() in ['да', 'yes', 'y']

# --- Поиск всех CSV-датасетов в папке ---
dataset_files = list(datasets_dir.glob("*.csv"))

# --- Обработка каждого датасета ---
for dataset_file in dataset_files:
    print(f"\n--- Работаем с датасетом: {dataset_file.name} ---")

    # --- Загрузка датасета ---
    df = pd.read_csv(dataset_file)
    X = df.drop(columns=["Collision"])
    y = df["Collision"]

    # --- Кодирование категориальных признаков ---
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # --- Кодируем y: "Да" → 1, "Нет" → 0 ---
    y = y.map({"Да": 1, "Нет": 0})

    # --- Разделение данных ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Проверка: есть ли два класса для обучения ---
    if len(np.unique(y_train)) < 2:
        print(f" Пропускаем {dataset_file.name}: только один класс.")
        continue

    # --- Инициализация моделей ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC()
    }

    training_times = {}

    print("Обучение моделей:")
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        training_times[name] = elapsed_time
        print(f"    {name} обучена за {elapsed_time:.4f} секунд.")

    # --- Выбор 3 самых быстрых моделей ---
    fastest_models = sorted(training_times.items(), key=lambda x: x[1])[:3]
    fastest_model_names = [name for name, _ in fastest_models]

    print("\nТри самые быстрые модели:")
    for name in fastest_model_names:
        print(f"  {name}")

    # --- Сохранение только 3-х самых быстрых моделей ---
    if save_models:
        for name in fastest_model_names:
            model = models[name]
            model_filename = output_dir / f"{dataset_file.stem}_{name.replace(' ', '_').lower()}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Модель '{name}' сохранена как: {model_filename}")
    else:
        print("Сохранение моделей отключено.")
