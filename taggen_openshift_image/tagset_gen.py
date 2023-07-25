#!/usr/bin/python3
""" Script function:
     - given a directory of changesets, create a directory containing the
       corresponding tagsets
COMMAND LINE INPUTS:
     - one input: changeset directory
     - two inputs: changset directory, tagset directory (IN THAT ORDER)
         * changeset directory must exist, if tagset directory does not exist it
           will be created
OUTPUTS:
     - a directory containing tagsets (.yaml files w/ .tag extension) for each
       of the changesets in the given changeset directory
"""

# Imports
#from collections import Counter

import logging
import logging.config

import pickle, json
import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, '../')

from pathlib import Path
import yaml

from tqdm import tqdm

from columbus.columbus import columbus
import argparse


# def parse_cs(changeset_names, cs_dir, multilabel=False): # SHOULD PROBABLY GET RID OF ITERATIVE OPTION
#     """ Function for parsing a list of changesets.
#     input: list of changeset names (strings), name of the directory in which
#             they are located
#     output: a list of labels and a corresponding list of features for each
#             changeset in the directory
#             (list of filepaths of changed/added files)
#     """
#     features = []
#     labels = []
#     for cs_name in tqdm(changeset_names):
#             changeset = get_changeset(cs_name, cs_dir)
#             if multilabel:
#                 """ running a trial in which there may be more than one label for
#                     a given changeset """
#                 if 'labels' in changeset:
#                     labels.append(changeset['labels'])
#                 else:
#                     labels.append(changeset['label'])
#             else: # each changeset will have just one label
#                 labels.append(changeset['label'])
#             features.append(changeset['changes'])
#     return features, labels

# def get_changeset(cs_fname, cs_dir):
#     """ Function that takes a changeset and returns the dictionary stored in it
#     input: file name of a *single* changeset
#     output: dictionary containing changed/added filepaths and label(s)
#     """
#     cs_dir_obj = Path(cs_dir).expanduser()
#     changeset = None
#     for csfile in cs_dir_obj.glob(cs_fname):
#         if changeset is not None:
#             raise IOError("Too many changesets match the file name")
#         with csfile.open('r') as f:
#             changeset = yaml.full_load(f)
#     if changeset is None:
#         logging.error("No changesets match the name %s", str(csfile))
#         raise IOError("No changesets match the name")
#     if 'changes' not in changeset or ('label' not in changeset and 'labels' not in changeset):
#         logging.error("Couldn't read changeset")
#         raise IOError("Couldn't read changeset")
#     return changeset

def get_columbus_tags(X, disable_tqdm=False, return_freq=True,
                       freq_threshold=2):
    """ Function that gets the columbus tags for a given list of filesystem
        changes
    input: a list of filesystem changes
    output: a list of tags and their frequency (as strings separated by a colon)
    """
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        if "PIL" in changeset:
            print(0)
        tag_dict = columbus(changeset, freq_threshold=freq_threshold)
        if return_freq:
            tags.append(['{}:{}'.format(tag, freq) for tag, freq
                         in tag_dict.items()])
        else:
            tags.append([tag for tag, freq in tag_dict.items()])
    return tags

# def create_tagset_names(changeset_names):
#     """ Create names for the new tagset files
#         (same as changeset names but with a .tag extension)
#     input: list of changeset names
#     output: list of names for tagsets created for these changesets
#     """
#     tagset_names = []
#     for name in changeset_names:
#         new_tagname = name[:-4] + "tag"
#         tagset_names.append(new_tagname)
#     return tagset_names

# def get_changeset_names(cs_dir):
#     """ Get the names of all the changesets in a given directory
#     # input: a directory name
#     # output: names of all changeset files within the directory
#     """
#     all_files = [f for f in listdir(cs_dir) if isfile(join(cs_dir, f))]
#     changeset_names = [f for f in all_files if ".yaml" in f and ".tag" not in f]
#     return changeset_names

# def create_dict(labels, tags):
#     """ Creates the tagset files and puts them in the specified directory
#     input: names of tagsets, name of target directory, labels, ids, and tags
#     output: returns nothing, creates tagset files (.yaml format) in given directory
#     """
#     for i, label in enumerate(labels):
#         if(isinstance(label, list)):
#             cur_dict = {'labels': label, 'tags': tags[i]}
#         else:
#             cur_dict = {'label': label, 'tags': tags[i]}
#         yield cur_dict

# def run(changesets, labels):

#     # logging.info("Generating tagsets:")
#     tags = get_columbus_tags(changesets)
    
#     # logging.info("Writing tagset files to %s", ts_dir)
#     return create_dict(labels, tags)

#     # logging.info("Tagset generation time: %s", str(time.time() - prog_start))

if __name__ == "__main__":
    # Load data from previous component
    cs_dump_path = "/home/ubuntu/Praxi-Pipeline/get_layer_changes/cwd/changesets_l_dump"
    with open(cs_dump_path, 'rb') as in_changesets_l:
        changesets_l = pickle.load(in_changesets_l)
                              
    # Tagset Generator
    tagsets_l = []
    for changeset in changesets_l:
        # tags = get_columbus_tags(changeset['changes'])
        tag_dict = columbus(changeset['changes'], freq_threshold=2)
        tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
        cur_dict = {'labels': changeset['labels'], 'tags': tags}
        tagsets_l.append(cur_dict)

    # Debug
    changesets_path = "/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/changesets.txt"
    with open(changesets_path, 'w') as writer:
        for change_dict in changesets_l:
            writer.write(json.dumps(change_dict) + '\n')
    tagsets_path = "/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets.txt"
    with open(tagsets_path, 'w') as writer:
        for tag_dict in tagsets_l:
            writer.write(json.dumps(tag_dict) + '\n')
    # time.sleep(5000)

    # Pass data to next component
    ts_dump_path = "/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets_l_dump"
    with open(ts_dump_path, 'wb') as writer:
        pickle.dump(tagsets_l, writer)