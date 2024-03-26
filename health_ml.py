import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(0)
n_samples=1000
n_features = 5

data={
    'Age': np.random.randint(18,80,n_samples),
    'BMI': np.random.uniform(15,50, n_samples),
    'Glucose_level': np.random.uniform(70,200,n_samples),
    'BloodPressure': np.random.randint(80,180,n_samples),
    'FamilyHistory': np.random.choice([0,1], n_samples),
    'Diabetes': np.random.choice([0,1],n_samples)
}


df = pd.DataFrame(data)
X = df[['Age','BMI','Glucose_level','BloodPressure','FamilyHistory']]
y = df['Diabetes']

x_train, x_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#standarize feature
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#create machine learning model(Rnadom forest classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

#train the model on the training data
clf.fit(x_train,y_train)

#note prediction on test
y_pred = clf.predict(x_test)

#evaluate the model
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)