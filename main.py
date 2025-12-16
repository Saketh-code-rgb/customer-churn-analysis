import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. DATA GENERATION (Simulating a dataset so you don't need a CSV file)
def generate_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 70, num_samples),
        'monthly_bill': np.random.uniform(20, 120, num_samples),
        'customer_service_calls': np.random.randint(0, 10, num_samples),
        'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], num_samples),
        'contract_length_months': np.random.randint(1, 36, num_samples)
    }
    df = pd.DataFrame(data)
    # Simulate 'Churn' (Target Variable) based on logic + noise
    # Older people with high bills and many service calls are likely to churn
    churn_prob = (df['age'] / 100) + (df['monthly_bill'] / 200) + (df['customer_service_calls'] / 10)
    df['churn'] = (churn_prob + np.random.normal(0, 0.2, num_samples)) > 1.0
    df['churn'] = df['churn'].astype(int)
    
    return df

# 2. DATA PREPROCESSING
print("Generating Data...")
df = generate_data()

# Convert categorical variables to numeric (One-Hot Encoding)
df = pd.get_dummies(df, columns=['subscription_type'], drop_first=True)

# Define Features (X) and Target (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Split into Train and Test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL TRAINING
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. EVALUATION
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Project Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 5. FEATURE IMPORTANCE (What drove the decision?)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Factors influencing Churn:")
print(importances.head(3))
