import sys
import json
import time
from typing import Counter
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def main(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        data = pd.DataFrame(data_dict)
    except Exception as e:
        print("Ошибка при чтении файла:", str(e))
        return

    X = data.drop('collision', axis=1)
    y = data['collision']

    models = [
        joblib.load(r'models\logistic_regression_model.joblib'),
        joblib.load(r'models\random_forest_model.joblib'),
        joblib.load(r'models\svm_lin_model.joblib'),
        joblib.load(r'models\svm_poly_model.joblib'),
        joblib.load(r'models\decision_tree_model.joblib'),
        joblib.load(r'models\svm_rbf_model.joblib')
    ]

    pipelines = []

    for model in models:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        start_time = time.time()
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        elapsed_time = time.time() - start_time

        f1 = f1_score(y, y_pred)

        pipelines.append(
            (model.__class__.__name__, dict(Counter(y_pred.tolist())), elapsed_time, f1))

    result = f""
    for pipeline in pipelines:
        result += f"Модель: {pipeline[0]}\n"
        result += f"Количество аварийных столкновений: {pipeline[1][1]}\n"
        result += f"Количество небезопасных столкновений: {pipeline[1][0]}\n"
        result += f"Время работы модели: {pipeline[2]:.6f} секунд\n"
        result += f"Коэффициент F1: {pipeline[3]:.6f}\n"
        result += "-" * 30 + "\n"
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python run_model.py '<data_in_json>'")
        sys.exit(1)

    data_json = sys.argv[1]
    result = main(data_json)
