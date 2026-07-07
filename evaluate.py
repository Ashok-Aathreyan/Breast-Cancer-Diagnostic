import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from preprocessing import load_and_preprocess_data


def evaluate_model():
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = load_and_preprocess_data()

    model = joblib.load("models/best_model.pkl")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=" * 60)
    print("BREAST CANCER DIAGNOSTIC SYSTEM")
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"]
        )
    )

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if hasattr(model, "predict_proba"):
        plt.figure(figsize=(6, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve")
        plt.grid(True)
        plt.show()

    print("\nEvaluation Completed Successfully.")


if __name__ == "__main__":
    evaluate_model()