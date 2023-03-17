# From https://data36.com/random-forest-in-python/
import pandas as pd
from IPython.core.display_functions import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Below imports are from: https://www.datacamp.com/tutorial/random-forests-classifier-python
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


# Import dataframe with values
df = pd.read_csv("possum.csv") #TODO: add actual data

# Ignore below if data is already cleaned for null
df = df.dropna()

# Seperate into X (features) and Y (labels)
X = df.drop(["case", "site", "Pop", "sex"], axis=1)
y = df["sex"]

# Split into training and test set (consider other methods)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# Create model
# n_estimators determines the number of decision trees that make up our random forest.
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)

# Predict values using model
predictions = rf_model.predict(X_test)
# If you want to see the probabilities
rf_model.predict_proba(X_test.head())

# Calculate accuracy (one of the ways to evaluate the model)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Export the first three decision trees from the forest
for i in range(3):
    tree = rf_model.estimators_[i]
    dot_data = export_graphviz(tree, out_file=f'tree{i}.dot',
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    from subprocess import call
    call(['dot', '-Tpng', f'tree{i}.dot', '-o', 'tree.png', '-Gdpi=600'])

# Calculate feature importance
importances = rf_model.feature_importances_
columns = X.columns
i = 0

while i < len(columns):
    print(f"The importance of feature '{columns[i]}' is {round(importances[i] * 100, 2)}%\n.")
    i += 1