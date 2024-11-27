import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the Iris dataset
file_path = r"C:\Users\Dharnish\Downloads\IRIS.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Explore the dataset
print("Dataset shape:", data.shape)
print("\nFirst 10 rows of the dataset:")
print(data.head(10))
print("\nDataset description:")
print(data.describe())
print("\nDataset info:")
print(data.info())
print("\nNumber of duplicated rows:", data.duplicated().sum())

# Drop duplicates if any
data.drop_duplicates(inplace=True)
print("Number of duplicated rows after dropping:", data.duplicated().sum())
print("\nNull values in the dataset:")
print(data.isnull().sum())

# Unique values in each column
for column in data.columns:
    print(f'{column}: {data[column].nunique()} unique values')

# Visualize species distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='species', palette='RdBu')
plt.title('Distribution of Iris Species')
plt.show()

# Encode the species labels
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
print("\nClasses after encoding:", le.classes_)  # Print original class names

# Visualize correlation matrix
plt.figure(figsize=(10, 5))
sns.heatmap(data.corr(), annot=True, linewidths=2)
plt.title('Correlation Matrix')
plt.show()

# Prepare data for training
X = data.drop('species', axis=1)  # Features
y = data['species']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate the model
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print('Model:', model)
    print('Training Accuracy:', train_accuracy)
    print('Testing Accuracy:', test_accuracy)
    print('\nClassification Report:\n', classification_report(y_test, y_test_pred, target_names=le.classes_))

# Train and evaluate Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
print("\nRandom Forest Classifier:")
evaluate_model(rf_model)

# Train and evaluate Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
print("\nDecision Tree Classifier:")
evaluate_model(dt_model)
