# ============================================================
# EXPERIMENT 5
# Monitoring Model Explainability and Data Drift
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from scipy.stats import ks_2samp


# ============================================================
# TASK 1: SETUP ENVIRONMENT AND LOAD DATASET
# ============================================================

print("\n==========================================")
print("TASK 1: SETUP ENVIRONMENT AND LOAD DATASET")
print("==========================================")

data = {
    "age": [
        25, 35, 45, 30, 50,
        28, 40, 60, 32, 48,
        26, 37, 52, 29, 44,
        55, 31, 42, 58, 34
    ],

    "tenure": [
        2, 5, 8, 3, 10,
        1, 7, 12, 4, 9,
        2, 6, 11, 3, 8,
        13, 5, 7, 10, 4
    ],

    "monthly_charges": [
        50, 70, 90, 55, 100,
        45, 80, 110, 60, 95,
        52, 75, 105, 58, 85,
        120, 65, 82, 115, 68
    ],

    "support_calls": [
        1, 2, 4, 1, 5,
        0, 3, 6, 2, 4,
        1, 2, 5, 1, 3,
        7, 2, 3, 6, 2
    ],

    "churn": [
        "No", "No", "Yes", "No", "Yes",
        "No", "Yes", "Yes", "No", "Yes",
        "No", "No", "Yes", "No", "Yes",
        "Yes", "No", "Yes", "Yes", "No"
    ]
}

df = pd.DataFrame(data)

print("\nDataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# TASK 2: GENERATE DATA DRIFT REPORT
# ============================================================

print("\n==========================================")
print("TASK 2: GENERATE DATA DRIFT REPORT")
print("==========================================")

reference_data = df.copy()
current_data = df.copy()

np.random.seed(42)

current_data["age"] = (
    current_data["age"]
    + np.random.randint(2, 10, size=len(current_data))
)

current_data["monthly_charges"] = (
    current_data["monthly_charges"]
    + np.random.randint(5, 30, size=len(current_data))
)

current_data["support_calls"] = (
    current_data["support_calls"]
    + np.random.randint(0, 3, size=len(current_data))
)

features = [
    "age",
    "tenure",
    "monthly_charges",
    "support_calls"
]

print("\nData Drift Results:")
print("------------------------------------------")

drift_results = []

for feature in features:

    statistic, p_value = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    if p_value < 0.05:
        drift_status = "Drift Detected"
    else:
        drift_status = "No Significant Drift"

    drift_results.append([
        feature,
        statistic,
        p_value,
        drift_status
    ])

    print(
        feature,
        " | KS Statistic:",
        round(statistic, 4),
        " | P-value:",
        round(p_value, 4),
        " |",
        drift_status
    )

drift_df = pd.DataFrame(
    drift_results,
    columns=[
        "Feature",
        "KS_Statistic",
        "P_Value",
        "Drift_Status"
    ]
)

# Save drift results
drift_df.to_csv(
    "data_drift_results.csv",
    index=False
)

print("\nData drift results saved as:")
print("data_drift_results.csv")


# ============================================================
# TASK 3: VISUALIZE DRIFT RESULTS
# ============================================================

print("\n==========================================")
print("TASK 3: VISUALIZE DRIFT RESULTS")
print("==========================================")

reference_mean = reference_data[features].mean()
current_mean = current_data[features].mean()

x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    reference_mean,
    width,
    label="Reference Data"
)

plt.bar(
    x + width / 2,
    current_mean,
    width,
    label="Current Data"
)

plt.xlabel("Features")
plt.ylabel("Mean Value")
plt.title("Data Drift Visualization")

plt.xticks(x, features)

plt.legend()

plt.tight_layout()

plt.savefig(
    "drift_visualization.png"
)

plt.show()

print("\nDrift visualization saved as:")
print("drift_visualization.png")


# ============================================================
# TASK 4: MODEL TRAINING AND EVALUATION
# ============================================================

print("\n==========================================")
print("TASK 4: MODEL TRAINING AND EVALUATION")
print("==========================================")

model_data = df.copy()

model_data["churn"] = model_data["churn"].map({
    "No": 0,
    "Yes": 1
})

X = model_data[
    [
        "age",
        "tenure",
        "monthly_charges",
        "support_calls"
    ]
]

y = model_data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# TASK 4 - MODEL EXPLAINABILITY
# ============================================================

print("\n==========================================")
print("MODEL EXPLAINABILITY")
print("==========================================")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)


plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.show()

print("\nFeature importance graph saved as:")
print("feature_importance.png")


# ============================================================
# TASK 5: MODEL PERFORMANCE REPORT
# ============================================================

print("\n==========================================")
print("TASK 5: MODEL PERFORMANCE REPORT")
print("==========================================")

performance_data = pd.DataFrame({
    "Actual": y_test.values,
    "Prediction": y_pred
})

performance_data.to_csv(
    "model_predictions.csv",
    index=False
)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nModel predictions saved as:")
print("model_predictions.csv")


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.yticks(
    [0, 1],
    ["No Churn", "Churn"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nGenerated Files:")

print("1. data_drift_results.csv")
print("2. drift_visualization.png")
print("3. feature_importance.png")
print("4. confusion_matrix.png")
print("5. model_predictions.csv")

print("\nAll 5 tasks completed successfully!")