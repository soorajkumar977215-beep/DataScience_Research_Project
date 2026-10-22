
from pathlib import Path
import shutil

import pandas as pd
import matplotlib.pyplot as plt


# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "05_Results"
FIGURES_DIR = PROJECT_DIR / "06_Figures"

CSV_FILE = RESULTS_DIR / "experiment_runs_new.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Check that the latest experiment results exist
if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Results file not found: {CSV_FILE}\n"
        "Run 01_missing_data_experiment.py first."
    )

# Load individual experiment runs
df = pd.read_csv(CSV_FILE)

required_columns = {
    "missing_percent",
    "seed",
    "model",
    "accuracy",
    "f1_score",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"CSV is missing required columns: {missing_columns}"
    )

# Create summaries from the exact runs used in the experiment
summary = (
    df.groupby(["missing_percent", "model"])
    .agg(
        Mean_Accuracy=("accuracy", "mean"),
        Mean_F1=("f1_score", "mean"),
    )
    .reset_index()
)

# Expected figure names
figure_names = [
    "all_models_accuracy.png",
    "all_models_f1.png",
    "missing_data_accuracy.png",
]

# Back up existing figures before replacing them
backup_dir = FIGURES_DIR / "backup_before_update"
backup_dir.mkdir(exist_ok=True)

for name in figure_names:
    old_file = FIGURES_DIR / name

    if old_file.exists():
        backup_file = backup_dir / name

        if not backup_file.exists():
            shutil.copy2(old_file, backup_file)
            print(f"Backup created: {backup_file}")

# Use a consistent order for models
model_order = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]

# Figure 1: Accuracy of all models
plt.figure(figsize=(9, 6))

for model_name in model_order:
    model_data = summary[summary["model"] == model_name]
    plt.plot(
        model_data["missing_percent"],
        model_data["Mean_Accuracy"] * 100,
        marker="o",
        label=model_name,
    )

plt.xlabel("Missing Data (%)")
plt.ylabel("Mean Accuracy (%)")
plt.title("Effect of Missing Data on Model Accuracy")
plt.xticks([0, 5, 10, 20, 30, 40])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "all_models_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Figure 2: F1-score of all models
plt.figure(figsize=(9, 6))

for model_name in model_order:
    model_data = summary[summary["model"] == model_name]
    plt.plot(
        model_data["missing_percent"],
        model_data["Mean_F1"] * 100,
        marker="o",
        label=model_name,
    )

plt.xlabel("Missing Data (%)")
plt.ylabel("Mean F1-score (%)")
plt.title("Effect of Missing Data on Model F1-score")
plt.xticks([0, 5, 10, 20, 30, 40])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "all_models_f1.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Figure 3: Focused accuracy trend for Logistic Regression
plt.figure(figsize=(9, 6))

model_name = "Logistic Regression"
model_data = summary[summary["model"] == model_name]

plt.plot(
    model_data["missing_percent"],
    model_data["Mean_Accuracy"] * 100,
    marker="o",
    label=model_name,
)

plt.xlabel("Missing Data (%)")
plt.ylabel("Mean Accuracy (%)")
plt.title("Effect of Missing Data on Logistic Regression Accuracy")
plt.xticks([0, 5, 10, 20, 30, 40])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "missing_data_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Updated figures saved successfully.")
print("Runs used:", len(df))
print("Figures folder:", FIGURES_DIR)