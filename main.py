import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB

# Step 1: Import the Dataset
data = pd.read_excel('Data.xlsx')

# Step 2: Dataset Size
rows, columns = data.shape
print("Dataset Size - Rows:", rows, "Columns:", columns)

# Step 3: Missing Values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Step 4: Data Distribution
data.hist()
plt.show()

# Step 5: Outliers
sns.boxplot(data=data)
plt.show()

# Step 6: Correlations
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Step 7: Feature Selection
selected_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

# DataFrame called 'data'
X_labeled = data.loc[:99, selected_features]  # Features of labeled data
y_labeled = data.loc[:99, 'Y']   # Target variable of labeled data
X_unlabeled = data.loc[100:119, selected_features]  # Features of unlabeled data

# Create a Gradient Boosting Regression model
model = GradientBoostingRegressor()

# Train the model on the labeled data
model.fit(X_labeled, y_labeled)

# Predict the target variable for the unlabeled data
y_unlabeled_pred = model.predict(X_unlabeled)

# Print the predicted Y values for the unlabeled data
for i, pred in enumerate(y_unlabeled_pred):
    print("Predicted Y for row", i + 101, ":", pred)

    # Calculate the R-squared value
    y_labeled_pred = model.predict(X_labeled)
    r2 = r2_score(y_labeled, y_labeled_pred)
    print("R-squared:", r2)

    # Define the cross-validation scheme
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, X_labeled, y_labeled, cv=kfold, scoring='r2')

    # Print the cross-validation scores
    print("Cross-Validation Scores:", scores)
    print("Mean R^2 Score:", scores.mean())


