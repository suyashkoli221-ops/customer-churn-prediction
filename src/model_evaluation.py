
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

results_path = BASE_DIR / "models" / "model_comparison.csv"
output_path = BASE_DIR / "visualizations" / "model_comparison.png"

results = pd.read_csv(results_path)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    width=0.8
)

plt.title("Machine Learning Model Comparison", fontsize=16)
plt.xlabel("Machine Learning Models")
plt.ylabel("Evaluation Score")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(output_path, dpi=300)
plt.show()

print("Evaluation chart saved successfully!")
print(output_path)