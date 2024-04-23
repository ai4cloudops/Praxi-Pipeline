from sklearn.feature_extraction.text import CountVectorizer

# Example token to index mapping
# token_to_index = {'apple': 0, 'banana': 1, 'orange': 2}
token_to_index = {
    'the': 0,
    'quick': 1,
    'brown': 2,
    'fox': 3,
    'jumps': 4,
    'over': 5,
    'lazy': 6,
    'dog': 7
}

# Example observations: a dictionary with token as key and count as value
# observations = [{'apple': 2, 'banana': 1}, {'orange': 3, 'apple': 1}]
observations = [
    {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1},
    {'the': 1, 'lazy': 1, 'dog': 1}
]

# Prepare observations by converting each to a string with tokens repeated 'count' times
observations_prepared = []
for observation in observations:
    observation_str = ' '.join([' '.join([token] * count) for token, count in observation.items()])
    observations_prepared.append(observation_str)

print(observations_prepared)

# Initialize CountVectorizer with the given vocabulary
vectorizer = CountVectorizer(vocabulary=token_to_index)

# Vectorize the prepared observations
X = vectorizer.transform(observations_prepared)

print(X)

# The result is a sparse matrix. To view it as an array:
X_array = X.toarray()

# Print the result
print(X_array)
