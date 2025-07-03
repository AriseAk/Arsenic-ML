import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser(description="Predict arsenic safety using trained model")
parser.add_argument('--model', type=str, default='model.pkl', help='Path to saved model (.pkl)')
args = parser.parse_args()

try:
    model = joblib.load(args.model)
    print(f"✅ Loaded model from {args.model}")
except FileNotFoundError:
    print(f"❌ Model file not found: {args.model}")
    exit()

sample_data = {
    'pH': [7.2],
    'Iron': [0.9],
    'Manganese': [0.25],
    'Conductivity': [320],
    'TDS': [180],
    'Well_Depth': [30]
}

X_new = pd.DataFrame(sample_data)

prediction = model.predict(X_new)[0]
probability = model.predict_proba(X_new)[0][1] 

print("\n Input Sample:")
print(X_new)

print("\n Prediction Result:")
if prediction == 1:
    print(f"Water is likely UNSAFE (Arsenic > 10 µg/L) | Risk: {round(probability * 100, 2)}%")
else:
    print(f"Water is likely SAFE (Arsenic ≤ 10 µg/L) | Risk: {round(probability * 100, 2)}%")
