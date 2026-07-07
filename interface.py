from train import train_models
from evaluate import evaluate_model
from predict import predict_patient


def display_menu():
    print("\n" + "=" * 60)
    print("        BREAST CANCER DIAGNOSTIC SYSTEM")
    print("=" * 60)
    print("1. Train Machine Learning Models")
    print("2. Evaluate Best Model")
    print("3. Predict Breast Cancer")
    print("4. Exit")
    print("=" * 60)


def main():
    while True:
        display_menu()

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            print("\nTraining Models...\n")
            train_models()

        elif choice == "2":
            print("\nEvaluating Best Model...\n")
            try:
                evaluate_model()
            except FileNotFoundError:
                print("\nModel not found!")
                print("Please train the model first by selecting Option 1.")

        elif choice == "3":
            print("\nPredicting...\n")
            try:
                predict_patient()
            except FileNotFoundError:
                print("\nModel not found!")
                print("Please train the model first by selecting Option 1.")

        elif choice == "4":
            print("\nThank you for using the Breast Cancer Diagnostic System.")
            print("Goodbye!")
            break

        else:
            print("\nInvalid choice! Please enter a number between 1 and 4.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()