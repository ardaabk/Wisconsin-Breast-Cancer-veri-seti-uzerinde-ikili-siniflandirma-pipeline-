import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png", dpi=300) # Kayıt işlemi
plt.show()

# Cleaning
columns_to_drop = ['smoothness error', 'mean fractal dimension', 'texture error', 'symmetry error']
X = df.drop(['diagnosis'] + columns_to_drop, axis=1)
y = df['diagnosis']

# ==========================================
# 2. MODEL PIPELINE
# ==========================================
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 3. CROSS VALIDATION
# ==========================================
scores = cross_val_score(model, X, y, cv=5)
cv_mean = scores.mean()

# ==========================================
# 4. EVALUATION & REPORTING
# ==========================================
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=['Malignant', 'Benign'])

# --- 1. Confusion Matrix'i Kaydet ---
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'], 
            yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("2_confusion_matrix.png", dpi=300) # Kayıt işlemi
plt.show()

# --- 2. Sonuç Raporunu (results_report.txt) Oluştur ---
with open("results_report.txt", "w") as f:
    f.write("=== MACHINE LEARNING PROJECT RESULTS REPORT ===\n\n")
    f.write(f"5-Fold Cross-Validation Average Accuracy: %{cv_mean * 100:.2f}\n")
    f.write(f"Test Set Accuracy: %{acc * 100:.2f}\n\n")
    f.write("--- Detailed Classification Report ---\n")
    f.write(report)
    f.write("\n\nNote: This report was automatically generated.")

print("İşlem tamamlandı!")
print("- Grafikler .png formatında kaydedildi.")
print("- Sonuçlar 'results_report.txt' dosyasına yazıldı.")