from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the datasets
try:
    train_df = pd.read_csv('paxel_freshness_train.csv')
    test_df = pd.read_csv('paxel_freshness_test.csv')
except FileNotFoundError:
    print("Please ensure 'paxel_freshness_train.csv' and 'paxel_freshness_test.csv' are uploaded.")
    # Handle the error appropriately, maybe exit or prompt the user again.
    exit()

# Drop irrelevant columns
train_df = train_df.drop(['package_id', 'origin_city', 'destination_city'], axis=1)
test_df = test_df.drop(['package_id', 'origin_city', 'destination_city'], axis=1)


# Data preprocessing
def preprocess_data(df):
    # Fill missing values if any
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Convert categorical features to numeric
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

train_df, train_encoders = preprocess_data(train_df.copy())
# Since test_df doesn't have 'freshness_status', we just process it
# without expecting 'freshness_status' to be present.
test_df, test_encoders = preprocess_data(test_df.copy())

# Separate features and target
X_train = train_df.drop('freshness_status', axis=1)
y_train = train_df['freshness_status']
# Since test_df doesn't have 'freshness_status', all data in test_df are features
X_test = test_df

# Align columns between training and testing sets
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    # Drop columns in X_train that are not in X_test, except for 'freshness_status'
    if c != 'freshness_status':
        X_train = X_train.drop(c, axis=1)


X_test = X_test[X_train.columns]

# Create and train the Decision Tree model
# Use hyperparameters to avoid overfitting
dt_classifier = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = dt_classifier.predict(X_test)

# Display the predictions
# Since we don't have y_test, we can only display the predictions
print("Predictions for the test data:")
print(y_pred)

# Add the prediction column to test_df
test_df_with_predictions = test_df.copy()
test_df_with_predictions['freshness_status'] = y_pred

# Revert all encoded columns to their original form
for col, le in test_encoders.items():
    if col in test_df_with_predictions.columns:
        # Check if the column is already in its original form to avoid errors
        if pd.api.types.is_numeric_dtype(test_df_with_predictions[col]):
             test_df_with_predictions[col] = le.inverse_transform(test_df_with_predictions[col])

# Revert the freshness_status column to its original form
le_freshness = train_encoders['freshness_status']
test_df_with_predictions['freshness_status'] = le_freshness.inverse_transform(test_df_with_predictions['freshness_status'])

# Save the dataframe to a CSV file
output_filename = 'paxel_freshness_test_with_predictions.csv'
test_df_with_predictions.to_csv(output_filename, index=False)

# Download the file
files.download(output_filename)

print(f"File '{output_filename}' is ready for download.")
display(test_df_with_predictions.head())

from sklearn.tree import export_graphviz
import graphviz
from PIL import Image
import os

# Export the model to DOT format completely (no depth limit)
dot_data_full = export_graphviz(dt_classifier, out_file=None,
                                feature_names=X_train.columns,
                                class_names=train_encoders['freshness_status'].classes_,
                                filled=True, rounded=True,
                                special_characters=True)

# Create a graph from dot_data
graph_full = graphviz.Source(dot_data_full)

# Save the graph as a PNG file
# graph.render will automatically add the .png extension
png_filename_full = 'decision_tree_full'
graph_full.render(png_filename_full, format='png', cleanup=True)

# Convert the PNG file to JPEG
jpeg_filename_full = 'decision_tree_full.jpeg'
# Open the correct PNG file, which is named 'decision_tree_full.png'
img_full = Image.open(png_filename_full + '.png')
img_full.convert('RGB').save(jpeg_filename_full, 'jpeg')

# Remove the temporary PNG file
os.remove(png_filename_full + '.png')

# Download the JPEG file
files.download(jpeg_filename_full)

print(f"The complete Decision Tree visualization has been saved as '{jpeg_filename_full}' and is ready for download.")

print("Legend for Decision Tree Interpretation:\n")

# Iterate through all saved encoders
for feature, encoder in train_encoders.items():
    # Create a mapping from numeric values to original labels
    mapping = {i: label for i, label in enumerate(encoder.classes_)}
    print(f"Feature: '{feature}'")
    for num_val, str_val in mapping.items():
        print(f"  {num_val} = '{str_val}'")
    print("-" * 30)

import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance scores from the model
importances = dt_classifier.feature_importances_

# Create a dataframe for easy visualization
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
})

# Sort the dataframe by importance score
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Features Ordered by Importance Level (Feature Importance):\n")
print(feature_importance_df)

# Visualize Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance from Decision Tree Model', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()