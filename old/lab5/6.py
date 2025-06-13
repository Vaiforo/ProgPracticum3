import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

def train_and_save_model(input_path: str, save_path: str) -> dict:
    df = pd.read_csv(input_path)
    X = df.drop(columns=['Collision'])
    y = df['Collision'].map({'Да': 1, 'Нет': 0})

    # Кодируем категориальные признаки
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])

    pipe.fit(X, y)

    # Сохраняем только обученную модель (SVC) в pickle
    with open(save_path, 'wb') as f:
        pickle.dump(pipe.named_steps['svc'], f)

    # Вычисляем точность на обучающих данных
    preds = pipe.predict(X)
    accuracy = (preds == y).mean()

    print(f"Модель сохранена в {save_path}")

    # Возвращаем словарь с результатами в нужном формате
    return {'script': 'script6', 'accuracy': accuracy, 'rows': len(df)}

if __name__ == "__main__":
    input_file = r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\datasets_storage\dataset_1_rows61_feats6.csv"
    save_file = r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\saved_models\optimized_svc_pso.pkl"
    results = train_and_save_model(input_file, save_file)
    print(results)
