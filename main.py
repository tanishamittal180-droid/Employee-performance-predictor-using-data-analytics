# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. CREATE SYNTHETIC DATASET
# -------------------------------
np.random.seed(42)

n = 500

data = pd.DataFrame({
    'age': np.random.randint(22, 60, n),
    'experience': np.random.randint(1, 20, n),
    'department': np.random.choice(['HR', 'IT', 'Sales'], n),
    'salary': np.random.randint(20000, 100000, n),
    'projects': np.random.randint(1, 10, n),
    'training_hours': np.random.randint(0, 50, n),
    'attendance': np.random.uniform(0.7, 1.0, n),
    'feedback_score': np.random.uniform(1, 5, n)
})

# Create target variable
conditions = [
    (data['projects'] > 6) & (data['feedback_score'] > 4),
    (data['projects'] > 3),
]

choices = ['High', 'Medium']

data['performance'] = np.select(conditions, choices, default='Low')

print(data.head())

# -------------------------------
# 2. ENCODE CATEGORICAL DATA
# -------------------------------
le = LabelEncoder()
data['department'] = le.fit_transform(data['department'])
data['performance'] = le.fit_transform(data['performance'])

# -------------------------------
# 3. SPLIT DATA
# -------------------------------
X = data.drop('performance', axis=1)
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# 4. TRAIN MODEL
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# 5. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. EVALUATION
# -------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. VISUALIZATION
# -------------------------------
sns.countplot(x='performance', data=data)
plt.title("Performance Distribution")
plt.show()