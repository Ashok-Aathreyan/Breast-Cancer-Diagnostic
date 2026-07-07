import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from preprocessing import load_and_preprocess_data


def train_models():
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = load_and_preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    best_model = None
    best_name = ""
    best_accuracy = 0

    print("=" * 60)
    print("BREAST CANCER DIAGNOSTIC SYSTEM")
    print("MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy : {accuracy * 100:.2f}%")

        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy * 100, 2)
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names.tolist(), "models/features.pkl")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Accuracy (%)",
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv("models/model_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"Model    : {best_name}")
    print(f"Accuracy : {best_accuracy * 100:.2f}%")

    print("\nSaved Files:")
    print("✓ models/best_model.pkl")
    print("✓ models/scaler.pkl")
    print("✓ models/features.pkl")
    print("✓ models/model_comparison.csv")

    print("\nTraining Completed Successfully.")


if __name__ == "__main__":
    train_models()