import os
# import sys
import yaml
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
# sys.path.insert(1, '/home/cc/Praxi-Pipeline/taggen_openshift_image/')
# from columbus.columbus import columbus

def process_yaml_file(filepath):
    """Load a changeset from a YAML file, process it, and write the output dict to tagsets_directory with '.tag' postfix."""
    import sys
    sys.path.insert(1, '/home/cc/Praxi-Pipeline/taggen_openshift_image/')
    from columbus.columbus import columbus

    # print(f"Processing {filepath}")
    # Construct the output filename (.yaml to .tag)
    output_filename = filepath.name.rsplit('.', 1)[0] + '.tag'
    output_path = Path(tagsets_directory) / output_filename
    if Path(output_path).exists():
        return f"Skipped {output_filename}"

    with open(filepath, 'r') as file:
        changeset = yaml.safe_load(file)

    # Process the changeset
    tag_dict = columbus(changeset['changes'], freq_threshold=1)
    # tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
    output_dict = {'labels': changeset['labels'], 'tags': tag_dict}
    
    # Write the output
    with open(output_path, 'w') as output_file:
        yaml.dump(output_dict, output_file, default_flow_style=False)
    
    return f"Done {output_filename}"

def process_all_yaml_files_in_parallel(directory):
    """Process each YAML file in the given directory in parallel."""
    yaml_files = list(Path(directory).glob('*.yaml'))
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_yaml_file = {executor.submit(process_yaml_file, df): df for idx, df in enumerate(yaml_files)}
        for idx, future in enumerate(as_completed(future_to_yaml_file)):
            yaml_file = future_to_yaml_file[future]
            try:
                result = future.result()
                print(f"{idx} {result}")
            except Exception as exc:
                print(f'{idx} {yaml_file} generated an exception: {exc}')
        # restuls = executor.map(process_yaml_file, yaml_files)
        # for idx, result in enumerate(restuls):
        #     try:
        #         print(f"{idx} {result}")
        #     except Exception as exc:
        #         print(f'{idx} {result} generated an exception: {exc}')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    changesets_directory = "/home/cc/Praxi-Pipeline/data/data_4/changesets_ML_3/"

    tagsets_directory = "/home/cc/Praxi-Pipeline/taggen_openshift_image/cwd/"
    os.makedirs(tagsets_directory, exist_ok=True)

    # Process all YAML files in parallel
    process_all_yaml_files_in_parallel(changesets_directory)
