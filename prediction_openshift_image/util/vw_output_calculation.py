import yaml

def read_yaml_file(filepath):
    """Reads a YAML file and returns the data."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

def sum_weights_for_classifier_tags(file_path, classifier_index, tags, tag_values):
    total_weight = 0.0
    weights_l, pos_weights_l = [], []
    lines_l = []
    classifier_str = f"[{classifier_index}]"
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if classifier_str in line:
                parts = line.split(':')
                feature = parts[0].strip()
                # if "asset" in feature:
                #     print()
                # Multiply weight by tag value if tag is in the feature name
                for tag in tags:
                    if tag in feature and len(parts) > 2:
                        weight = float(parts[-1])
                        total_weight += weight
                        weights_l.append(weight)
                        lines_l.append(line)
                        if weight > 0:
                            pos_weights_l.append(weight)
                        break  # Ensures that only the first matching tag is used
    return total_weight, weights_l, lines_l, pos_weights_l

# Path to your YAML file
yaml_filepath = '/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL/python3_9_18-bookworm.spacy_v3_7_2.0.tag'
# Path to your readable VW model file
model_filepath = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_2timesdata_batchdatareplay0_batchbybatch30_SL_conf_oaa90 copy/inverted_model.txt'
# Classifier index to sum weights for
classifier_index = 1

# Read tags from the YAML file
yaml_data = read_yaml_file(yaml_filepath)
tags = list(yaml_data.get('tags', {}).keys())  # Extract tags from YAML
tag_values = yaml_data.get('tags', {})  # Dictionary of tags and their values

# Calculate the total weight for classifier 0
total_weight_classifier_0, weights_l, lines_l, pos_weights_l = sum_weights_for_classifier_tags(model_filepath, classifier_index, tags, tag_values)
# print("==============================")
# print("Weights:")
# print(weights_l)
# print("==============================")
# print("Lines:")
# print(lines_l)
# print("==============================")
# print("Positive Weights:")
# print(pos_weights_l)
# print("==============================")
print(f"Total weight for classifier {classifier_index} using specified tags: {total_weight_classifier_0}")
