# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. LOAD DATASET
df = pd.read_csv("heart_disease_uci.csv")  

print(df.head())
print(df.info())
print(df.describe())

# 3. TARGET CREATION
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
# Convert multi-class (0–4) → binary
# 0 = No disease, 1 = Disease
df = df[['age','sex','cp','trestbps','chol','fbs',
         'restecg','thalch','exang','oldpeak','target']]
print(df.isnull().sum())

# 4. DATA OVERVIEW BEFORE CLEANING
print("\nMissing values before cleaning:\n", df.isnull().sum())

# 5. DATA PREPROCESSING
# Convert boolean → numeric
df['fbs'] = df['fbs'].map({True: 1, False: 0})
df['exang'] = df['exang'].map({True: 1, False: 0})
# Encode categorical columns
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
# Convert string categories → numeric labels
df['cp'], _ = pd.factorize(df['cp'])
df['restecg'], _ = pd.factorize(df['restecg'])
# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# 6. SPLIT DATA
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. APPLY NAIVE BAYES
model = GaussianNB()
model.fit(X_train, y_train)

# 8. PREDICTION & PROBABILITY
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\nSample Probabilities:\n", y_prob[:5])


# 9. EVALUATION
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.2f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 11. VISUALIZATION

# Disease vs No Disease
sns.countplot(x='target', data=df)
plt.title("Disease vs No Disease")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# 12. FINAL PROBABILITY OUTPUT
sample = X_test.iloc[[0]]
prob = model.predict_proba(sample)

print("\nFinal Prediction Example:")
print(f"Probability of No Disease: {prob[0][0]:.2f}")
print(f"Probability of Disease: {prob[0][1]:.2f}")

# 13. COMBINED OUTPUT
# prediction + probability together 
print("\nSample Predictions with Probabilities:")
for i in range(5):
    print(f"Predicted: {y_pred[i]} | "
          f"P(No Disease): {y_prob[i][0]:.2f} | "
          f"P(Disease): {y_prob[i][1]:.2f}")
