# gender_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load data
df = pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/gender-classification/voice.csv")
print("Dataset loaded successfully!")

# Step 2: Encode labels (male/female → 0/1)
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Step 3: Prepare features & labels
X = df.drop("label", axis=1)
y = df["label"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Optional - Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
