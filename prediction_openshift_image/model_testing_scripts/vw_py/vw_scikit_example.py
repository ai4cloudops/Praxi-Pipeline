from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Prepare the data
data = [
    ('1:0.1', {'5': 0.2, '15': 1.5, '30': 0.3}),  # Label format 'label:cost'
    ('2:0.1', {'10': 0.5, '20': 1.2, '25': 0.4}),
    ('3:0.1', {'5': 1.0, '15': 0.5, '35': 2.0})
]

# Separate labels and features
labels = [label for label, features in data]
feature_dicts = [features for label, features in data]

# Convert feature dictionaries to a sparse matrix format using DictVectorizer
vectorizer = DictVectorizer(sparse=True)
# Fitting the vectorizer to the feature dictionaries
X = vectorizer.fit_transform(feature_dicts)  # Using fit_transform to both fit and transform the data

# Initialize VWClassifier
# Pass custom VW args if necessary. Here we assume CSOAA isn't directly supported, so we demonstrate a possible approach:
# `--csoaa_ldf multiline` is used to specify csoaa with multiline examples.
model = VWClassifier(loss_function='logistic', quiet=True, csoaa=3)

# Train the model
model.fit(X, labels)

# Make a prediction
test_features = vectorizer.transform([{'5': 1.0, '15': 0.5, '35': 2.0}])  # Using transform since the vectorizer is already fitted
prediction = model.predict_proba(test_features)

print("Predicted label:", prediction)
