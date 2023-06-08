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

import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, '../')

from pathlib import Path
import yaml

from tqdm import tqdm

from columbus.columbus import columbus


def parse_cs(changeset_names, cs_dir, multilabel=False): # SHOULD PROBABLY GET RID OF ITERATIVE OPTION
    """ Function for parsing a list of changesets.
    input: list of changeset names (strings), name of the directory in which
            they are located
    output: a list of labels and a corresponding list of features for each
            changeset in the directory
            (list of filepaths of changed/added files)
    """
    features = []
    labels = []
    for cs_name in tqdm(changeset_names):
            changeset = get_changeset(cs_name, cs_dir)
            if multilabel:
                """ running a trial in which there may be more than one label for
                    a given changeset """
                if 'labels' in changeset:
                    labels.append(changeset['labels'])
                else:
                    labels.append(changeset['label'])
            else: # each changeset will have just one label
                labels.append(changeset['label'])
            features.append(changeset['changes'])
    return features, labels

def get_changeset(cs_fname, cs_dir):
    """ Function that takes a changeset and returns the dictionary stored in it
    input: file name of a *single* changeset
    output: dictionary containing changed/added filepaths and label(s)
    """
    cs_dir_obj = Path(cs_dir).expanduser()
    changeset = None
    for csfile in cs_dir_obj.glob(cs_fname):
        if changeset is not None:
            raise IOError("Too many changesets match the file name")
        with csfile.open('r') as f:
            changeset = yaml.full_load(f)
    if changeset is None:
        logging.error("No changesets match the name %s", str(csfile))
        raise IOError("No changesets match the name")
    if 'changes' not in changeset or ('label' not in changeset and 'labels' not in changeset):
        logging.error("Couldn't read changeset")
        raise IOError("Couldn't read changeset")
    return changeset

def get_columbus_tags(X, disable_tqdm=False, return_freq=True,
                       freq_threshold=2):
    """ Function that gets the columbus tags for a given list of filesystem
        changes
    input: a list of filesystem changes
    output: a list of tags and their frequency (as strings separated by a colon)
    """
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        tag_dict = columbus(changeset, freq_threshold=freq_threshold)
        if return_freq:
            tags.append(['{}:{}'.format(tag, freq) for tag, freq
                         in tag_dict.items()])
        else:
            tags.append([tag for tag, freq in tag_dict.items()])
    return tags

def create_tagset_names(changeset_names):
    """ Create names for the new tagset files
        (same as changeset names but with a .tag extension)
    input: list of changeset names
    output: list of names for tagsets created for these changesets
    """
    tagset_names = []
    for name in changeset_names:
        new_tagname = name[:-4] + "tag"
        tagset_names.append(new_tagname)
    return tagset_names

def get_changeset_names(cs_dir):
    """ Get the names of all the changesets in a given directory
    # input: a directory name
    # output: names of all changeset files within the directory
    """
    all_files = [f for f in listdir(cs_dir) if isfile(join(cs_dir, f))]
    changeset_names = [f for f in all_files if ".yaml" in f and ".tag" not in f]
    return changeset_names

def create_files(tagset_names, ts_dir, labels, ids, tags):
    """ Creates the tagset files and puts them in the specified directory
    input: names of tagsets, name of target directory, labels, ids, and tags
    output: returns nothing, creates tagset files (.yaml format) in given directory
    """
    for i, tagset_name in enumerate(tagset_names):
        if(isinstance(labels[i], list)):
            cur_dict = {'labels': labels[i], 'id': ids, 'tags': tags[i]}
        else:
            cur_dict = {'label': labels[i], 'id': ids, 'tags': tags[i]}
        cur_fname = ts_dir + '/' + tagset_name
        with open(cur_fname, 'w') as outfile:
            print("gen_tagset", os.path.dirname(outfile.name))
            print("gen_tagset", cur_fname)
            yaml.dump(cur_dict, outfile, default_flow_style=False)
        yield cur_dict

def run():
    # prog_start = time.time()

    cs_dir = "/fake-snapshot/changesets"
    ts_dir = "/tagset"
    if not os.path.isdir(ts_dir):
        os.mkdir(ts_dir)


    changeset_names = get_changeset_names(cs_dir)
    if len(changeset_names)==0:
        logging.error("No changesets in selected directory. Make sure to chose an input directory containing changesets")
        raise ValueError("No changesets in selected directory")

    # logging.info("Creating names for new tagset files:")
    tagset_names = create_tagset_names(changeset_names)
    # ids = get_ids(changeset_names)
    ids = []

    changesets = []
    labels = []
    changesets, labels = parse_cs(changeset_names, cs_dir, multilabel = True)

    # logging.info("Generating tagsets:")
    tags = get_columbus_tags(changesets)

    # logging.info("Writing tagset files to %s", ts_dir)
    return create_files(tagset_names, ts_dir, labels, ids, tags)

    # logging.info("Tagset generation time: %s", str(time.time() - prog_start))
