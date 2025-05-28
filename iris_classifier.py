from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Linear Regression": LinearRegression()
}

# Store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if name == "Linear Regression":
        # Round and clamp to [0, 2] for classification
        preds = np.clip(np.round(preds), 0, 2).astype(int)

    # Calculate metrics safely
    try:
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    except Exception as e:
        print(f"[!] Metric error for {name}: {e}")
        acc = prec = rec = f1 = 0.0

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    # Save model
    joblib.dump(model, f"iris_model_{name.lower().replace(' ', '_')}.pkl")

    # Show metrics
    print(f"\n {name}")
    print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")
    print(confusion_matrix(y_test, preds))

# Plot comparison for each metric
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(7, 5))
    values = [results[m][metric] for m in models]
    sns.barplot(x=list(models.keys()), y=values)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"metrics_plot_{metric.lower().replace(' ', '_')}.png")
    plt.close()
    plt.show()
