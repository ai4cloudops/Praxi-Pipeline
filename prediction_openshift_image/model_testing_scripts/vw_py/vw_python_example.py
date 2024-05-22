import vowpalwabbit

def train_vw_csoaa(data):
    # Initialize the Vowpal Wabbit model with csoaa reduction
    # Use the `--csoaa` followed by the number of labels as a string to specify the reduction
    # Example with 3 labels: csoaa_3
    model = vowpalwabbit.Workspace("--csoaa 3 --quiet --passes 10 --random_seed 4 -b 26 --learning_rate 1.5 --cache")

    # Train the model with the provided data
    for label, features in data:
        example = "|"
        # Construct the Vowpal Wabbit example format
        for class_idx in range(3,0,-1):
            if class_idx == label:
                example = str(class_idx) + ":0 " + example  # Labels are used directly for csoaa
            else:
                example = str(class_idx) + ":1 " + example  # Labels are used directly for csoaa
        for idx, value in features.items():
            example += f" {idx}:{value}"
        
        # Learn from the example
        model.learn(example)
    
    return model

# Example data: list of tuples with label and feature dictionary
sparse_data = [
    (1, {'5': 0.2, '15': 1.5, '30': 0.3}),
    (2, {'10': 0.5, '20': 1.2, '25': 0.4}),
    (3, {'5': 1.0, '15': 0.5, '35': 2.0})
]

model = train_vw_csoaa(sparse_data)

# Make a prediction with the trained model
# Construct a test example in a similar format
# test_example = "1 2 3 |features 5:1.0 15:0.5 35:2.0"
test_example = "1 2 3 | 10:0.5 20:1.2 25:0.4"
prediction = model.predict(test_example, prediction_type=vowpalwabbit.PredictionType.MULTICLASS)
print("Predicted label:", prediction)
print(model.get_prediction_type())
ex = vowpalwabbit.Example(model, test_example)
print(ex.get_prediction())
print(ex.get_prediction(vowpalwabbit.PredictionType.MULTICLASS))
