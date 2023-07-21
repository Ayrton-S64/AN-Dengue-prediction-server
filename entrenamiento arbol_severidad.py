import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset into a pandas DataFrame
data = pd.read_csv('./Data/diagnosticos/denguesevero_procesado(NO ID).csv')


# Perform one-hot encoding for categorical variables
categorical_columns = ['EDAD', 'GENERO', 'CEFALEA', 'MIALGIA', 'DOLOR ABDOMINAL', 'ERUPCIONES', 'VOMITOS', 'DIFICULTAD AL RESPIRAR',
                       'HEMORRAGIA', 'ORGANOMEGALIA']

# Perform one-hot encoding for the categorical columns
# data_encoded = pd.concat([pd.get_dummies(data, columns=categorical_columns, drop_first=True), data], axis=1)
data_encoded = pd.get_dummies(data, columns=categorical_columns, dtype=int, drop_first=True)

print(data_encoded.head())

target = data_encoded['RESULTADO']
features = data_encoded.drop('RESULTADO', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=50)

print(X_train.columns)

# Create a decision tree classifier and train it
tree_classifier = DecisionTreeClassifier(max_depth=4, random_state=50)
tree_classifier.fit(X_train, y_train)

# Get the feature importances
feature_importances = tree_classifier.feature_importances_

# Create a DataFrame to display the feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print(feature_importance_df)

# Make predictions on the testing set and evaluate the model's performance
y_pred = tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Export the trained model
joblib.dump(tree_classifier, './modelos/severity_diagnosis-decision_tree_model.pkl')