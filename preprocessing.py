import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path="dataset/breast_cancer.csv"):
    df = pd.read_csv("C:\Breast Cancer Diagonosis\dataset\data.csv")

    if "Unnamed: 32" in df.columns:
        df.drop(columns=["Unnamed: 32"], inplace=True)

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X.columns
    )


if __name__ == "__main__":
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = load_and_preprocess_data()

    print("=" * 50)
    print("BREAST CANCER DATA PREPROCESSING")
    print("=" * 50)

    print(f"\nTraining Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")
    print(f"Number of Features : {X_train.shape[1]}")

    print("\nFeature Names:")
    for feature in feature_names:
        print(feature)

    print("\nPreprocessing Completed Successfully.")