# predict.py
# Handles string labels and avoids indexing errors

import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load saved model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

print("\nEnter Iris Flower Measurements")
print("-----------------------------------------")
print("Use these reference ranges from the dataset:")
print("• Sepal Length : 4.3 cm  to 7.9 cm")
print("• Sepal Width  : 2.0 cm  to 4.4 cm")
print("• Petal Length : 1.0 cm  to 6.9 cm")
print("• Petal Width  : 0.1 cm  to 2.5 cm")
print("-----------------------------------------\n")

# Safe numeric input
def get_number(prompt):
    while True:
        value = input(prompt)
        try:
            return float(value)
        except ValueError:
            print("Invalid input. Enter a numeric value.\n")

# Take user inputs
sl = get_number("Sepal Length (cm) [Example: 5.1]: ")
sw = get_number("Sepal Width (cm)  [Example: 3.5]: ")
pl = get_number("Petal Length (cm) [Example: 1.4]: ")
pw = get_number("Petal Width (cm)  [Example: 0.2]: ")

# Prepare input for prediction
input_data = np.array([[sl, sw, pl, pw]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

# Print result directly (prediction is already string label)
print("\n-----------------------------------------")
print(" Predicted Species:", prediction)
print("-----------------------------------------")
