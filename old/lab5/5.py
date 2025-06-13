import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_pipeline(raw_csv: str, out_csv: str) -> dict:
    # Читаем CSV с твоей папки datasets_storage
    df = pd.read_csv(raw_csv)

    # Признаки — все кроме 'Collision', целевая — 'Collision', преобразуем "Да"/"Нет"
    X = df.drop('Collision', axis=1)
    y = df['Collision'].map({'Да':1, 'Нет':0})

    # Кодируем категориальные признаки (строки)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col])

    # Загружаем модель из saved_models с правильным путём
    with open(r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\saved_models\optimized_lr_ga.pkl", 'rb') as f:
        best_lr = pickle.load(f)

    # Создаём pipeline с масштабированием и моделью
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', best_lr)
    ])

    # Обучаем модель
    pipe.fit(X, y)

    # Добавляем предсказания
    df['pred'] = pipe.predict(X)

    # Сохраняем CSV с результатами в папку outputs
    df.to_csv(out_csv, index=False)

    # Считаем точность
    accuracy = (df['pred'] == y).mean()

    return {'script': 'script5', 'accuracy': accuracy, 'n_rows': len(df)}

if __name__ == '__main__':
    import sys
    # Путь к входному файлу в твоей папке datasets_storage
    input_file = r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\datasets_storage\dataset_1_rows61_feats6.csv"
    # Путь к выходному файлу в папке outputs
    output_file = r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\outputs\script5_output.csv"

    result = run_pipeline(input_file, output_file)
    print(result)
