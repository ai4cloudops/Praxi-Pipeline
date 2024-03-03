import os, sys
import yaml
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(1, '/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/')
from columbus.columbus import columbus

def load_changesets_from_yaml(directory):
    """Load all changesets from YAML files in the specified directory."""
    changesets_l = []
    for yaml_file in Path(directory).glob('*.yaml'):
        with open(yaml_file, 'r') as file:
            changeset = yaml.safe_load(file)
            changesets_l.append(changeset)
    return changesets_l

def process_changeset(changeset):
    """Process a single changeset with the columbus function."""
    tag_dict = columbus(changeset['changes'], freq_threshold=2)
    tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
    return {'labels': changeset['labels'], 'tags': tags}

def process_changesets_in_parallel(changesets_l):
    """Process each changeset in parallel and return the list of tagsets."""
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        tagsets_l = list(executor.map(process_changeset, changesets_l))
    return tagsets_l

if __name__ == "__main__":
    changesets_directory = "/home/ubuntu/Praxi-Pipeline/get_layer_changes/cwd/changesets"
    changesets_l = load_changesets_from_yaml(changesets_directory)

    # Process changesets in parallel
    tagsets_l = process_changesets_in_parallel(changesets_l)

    # Save tagsets to file
    with open("/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets_l", 'wb') as writer:
        pickle.dump(tagsets_l, writer)

    # For debugging: Write each tagset to a separate file
    for ind, tag_dict in enumerate(tagsets_l):
        with open(f"/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets_logging{ind}.tag", 'w') as writer:
            yaml.dump(tag_dict, writer, default_flow_style=False)
