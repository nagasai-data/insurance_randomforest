# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("train.csv")  # Make sure this CSV is in the same folder as your script

# Converting categorical values to numerical
data["Gender"] = data["Gender"].map({'Male': 1, 'Female': 0})
data["Vehicle_Damage"] = data["Vehicle_Damage"].map({'Yes': 1, 'No': 0})
data["Vehicle_Age"] = data["Vehicle_Age"].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})

# Dropping the 'id' column as it's not useful for prediction
data.drop(columns=["id"], inplace=True)

# Splitting into features (X) and target (y)
X = data.drop(columns=["Response"])
y = data["Response"]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()
