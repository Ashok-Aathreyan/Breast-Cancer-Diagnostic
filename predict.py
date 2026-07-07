import joblib
import numpy as np


def predict_patient():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/features.pkl")

    print("=" * 60)
    print("BREAST CANCER DIAGNOSTIC SYSTEM")
    print("PATIENT DIAGNOSIS")
    print("=" * 60)

    print("\nEnter the following patient details:\n")

    values = []

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                values.append(value)
                break
            except ValueError:
                print("Please enter a valid numeric value.")

    sample = np.array(values).reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    if prediction == 1:
        print("Diagnosis          : MALIGNANT")
        print(f"Confidence         : {probability[1] * 100:.2f}%")
    else:
        print("Diagnosis          : BENIGN")
        print(f"Confidence         : {probability[0] * 100:.2f}%")

    print("\nProbability Scores")
    print("------------------------------")
    print(f"Benign     : {probability[0] * 100:.2f}%")
    print(f"Malignant  : {probability[1] * 100:.2f}%")

    print("\nPrediction Completed Successfully.")


if __name__ == "__main__":
    predict_patient()