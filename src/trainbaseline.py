import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# STEP 1: Load dataset
data = pd.read_csv("data/processed/cleaned_dataset.csv")

# STEP 2: Features & target
X = data[['LOC', 'CBO', 'RFC', 'WMC']]
y = data['defect']

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Feature scaling (IMPORTANT)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 5: Train Random Forest (BASELINE)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# STEP 6: Predictions
y_pred = model.predict(X_test)

# STEP 7: Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))