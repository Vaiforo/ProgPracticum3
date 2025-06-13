import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def run_pipeline(raw_csv: str, out_csv: str) -> dict:
    df = pd.read_csv(raw_csv)

    X = df.drop(columns=['Collision'])
    y = df['Collision'].map({'Да': 1, 'Нет': 0})

    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    pipe.fit(X, y)

    df['pred'] = pipe.predict(X)
    df.to_csv(out_csv, index=False)

    accuracy = (df['pred'] == y).mean()

    return {'script': 'script3', 'accuracy': accuracy, 'n_rows': len(df)}

def main():
    input_dir = Path(r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\datasets_storage")
    output_dir = Path(r"C:\Users\Arlecchino\Desktop\work plesk labs\venv\.venv\Labs\outputs")
    output_dir.mkdir(exist_ok=True)

    results = []
    for file_path in input_dir.glob("*.csv"):
        out_file = output_dir / f"pred_{file_path.name}"
        print(f"Processing {file_path.name} ...")
        res = run_pipeline(str(file_path), str(out_file))
        print(f"Saved predictions to {out_file}\n")
        results.append(res)

    print("Summary:")
    for r in results:
        print(r)

if __name__ == '__main__':
    main()
