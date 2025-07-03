import pandas as pd
import argparse
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_auc_score, 
                             roc_curve, RocCurveDisplay)
import sys

parser = argparse.ArgumentParser(description="Train Random Forest to predict arsenic contamination")
parser.add_argument('--file', type=str, default='data.csv', help='Path to dataset CSV file')
parser.add_argument('--save_model', type=str, default='model.pkl', help='Filename to save trained model')
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError(f"Dataset not found: {args.file}")

df = pd.read_csv(args.file)

df['As_Unsafe'] = df['Arsenic'].apply(lambda x: 1 if x > 0.01 else 0)

features = ['pH', 'Iron', 'Manganese', 'Conductivity', 'TDS', 'Well_Depth']
X = df[features]
y = df['As_Unsafe']

class_counts = y.value_counts()
print("Class distribution:\n", class_counts)
if len(class_counts) < 2:
    print("❌ ERROR: Your dataset does not contain both classes (safe and unsafe).")
    print("Please check your data or adjust the threshold for 'unsafe' arsenic.")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train class distribution:\n", pd.Series(y_train).value_counts())
print("Test class distribution:\n", pd.Series(y_test).value_counts())

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', verbose=1)
grid.fit(X_train, y_train)

print("Best Hyperparameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_proba)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title("ROC Curve")
plt.show()

scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("\nCross-validated Accuracy:", scores.mean())

joblib.dump(best_model, args.save_model)
print(f"\n✅ Model saved to {args.save_model}")
