# Импорт необходимых библиотек
import pandas as pd  # Библиотека для работы с данными (DataFrame)
import os  # Модуль для взаимодействия с операционной системой
from pathlib import Path  # Удобный способ работы с путями в разных ОС
from sklearn.pipeline import Pipeline  # Инструмент для построения ML-конвейера
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Для нормализации и кодирования признаков
from sklearn.linear_model import LogisticRegression  # Модель логистической регрессии


def run_pipeline(raw_csv: str, out_csv: str) -> dict:
    """
    Функция загружает данные из CSV-файла, обучает модель и сохраняет предсказания.

    raw_csv — путь к входному CSV-файлу
    out_csv — путь к выходному файлу с предсказаниями
    Возвращает словарь с результатами
    """

    # Загрузка данных из CSV файла в DataFrame
    df = pd.read_csv(raw_csv)

    # Разделение на признаки (X) и целевую переменную (y)
    X = df.drop(columns=['Collision'])  # Все столбцы, кроме 'Collision' — это признаки
    y = df['Collision'].map({'Да': 1, 'Нет': 0})  # Целевая переменная, преобразованная в числа (1/0)

    # Кодирование категориальных признаков (преобразование строковых значений в числовые)
    for col in X.columns:
        if X[col].dtype == object:  # Если тип данных — строка (категориальный признак)
            le = LabelEncoder()  # Создаем объект LabelEncoder
            X[col] = le.fit_transform(X[col])  # Обучаем и применяем кодировку к столбцу

    # Создаем ML-конвейер (Pipeline), состоящий из двух этапов:
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Нормализация данных
        ('lr', LogisticRegression(max_iter=1000))  # Логистическая регрессия с увеличенным количеством итераций
    ])

    # Обучаем конвейер на обучающих данных
    pipe.fit(X, y)

    # Делаем предсказание на тех же данных и добавляем его как новый столбец в DataFrame
    df['pred'] = pipe.predict(X)

    # Сохраняем измененный DataFrame с предсказаниями в новый CSV-файл
    df.to_csv(out_csv, index=False)

    # Вычисляем точность модели: совпадают ли предсказанные значения с реальными
    acc = (df['pred'] == y).mean()

    # Возвращаем информацию о выполнении в виде словаря
    return {
        'script': 'script1',
        'accuracy': acc,
        'n_rows': len(df)
    }


def main():
    """
    Основная функция. Обрабатывает все CSV-файлы из указанной директории.
    """

    # Путь к папке с входными CSV-файлами
    input_folder = Path(r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\datasets_storage")

    # Путь к папке, куда будут сохраняться выходные файлы
    output_folder = Path(r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\outputs")

    # Создаем папку, если она не существует
    output_folder.mkdir(exist_ok=True)

    # Список результатов обработки каждого файла
    results = []

    # Перебираем все CSV-файлы в папке input_folder
    for file_path in input_folder.glob("*.csv"):
        # Формируем имя выходного файла
        out_file = output_folder / f"pred_{file_path.name}"

        # Сообщаем пользователю, что файл обрабатывается
        print(f"Обрабатываем {file_path.name} ...")

        # Выполняем основной ML-процесс
        res = run_pipeline(str(file_path), str(out_file))

        # Сообщаем, что файл сохранён
        print(f"Сохранено в {out_file}\n")

        # Добавляем результаты в список
        results.append(res)

    # Выводим сводку результатов обработки всех файлов
    print("Итоги:")
    for r in results:
        print(r)


# Точка входа в программу
if __name__ == "__main__":
    main()