
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_DIR
    / "02_Data"
    / "data_banknote_authentication.txt"
)

# The dataset has four numeric features and one target column.
columns = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "class",
]

data = pd.read_csv(DATA_FILE, header=None, names=columns)

X = data.drop(columns="class")
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(random_state=42),
    ),
}

missing_levels = [0, 5, 10, 20, 30, 40]
seeds = [10, 20, 30, 40, 42]
results = []

for level in missing_levels:
    for seed in seeds:
        rng = np.random.default_rng(seed)

        X_train_missing = X_train.to_numpy(dtype=float, copy=True)
        if level > 0:
            mask = rng.random(X_train_missing.shape) < level / 100
            X_train_missing[mask] = np.nan

        X_train_missing = pd.DataFrame(
            X_train_missing,
            columns=X.columns,
            index=X_train.index,
        )

        for model_name, model in models.items():
            model.fit(X_train_missing, y_train)
            predictions = model.predict(X_test)

            results.append({
                "missing_percent": level,
                "seed": seed,
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
            })


output_dir = PROJECT_DIR / "05_Results"
output_dir.mkdir(exist_ok=True)

results_df = pd.DataFrame(results)

# Save all 90 individual experiment runs
runs_file = output_dir / "experiment_runs_new.csv"
results_df.to_csv(runs_file, index=False)

# Create a summary from these exact same runs
summary_df = (
    results_df
    .groupby(["missing_percent", "model"])
    .agg(
        Mean_Accuracy=("accuracy", "mean"),
        Std_Accuracy=("accuracy", "std"),
        Mean_F1=("f1_score", "mean"),
        Std_F1=("f1_score", "std"),
    )
    .reset_index()
)

# Save to a new file; preserve all previous results
summary_file = output_dir / "experiment_runs_new_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"Dataset rows: {len(data)}")
print(f"Individual runs: {len(results_df)}")
print(f"Runs saved to: {runs_file}")
print(f"Summary saved to: {summary_file}")
print(summary_df.round(4).to_string(index=False))