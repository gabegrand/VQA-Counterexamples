import argparse
import h5py
import numpy as np
import os
import pickle
import shutil
import yaml
import json
import click
from IPython.display import Image, display
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import vqa.lib.engine as engine
import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.lib.criterions as criterions
import vqa.datasets as datasets
import vqa.models as models

with open('options/vqa2/mutan_att_trainval.yaml', 'r') as handle:
    options = yaml.load(handle)
options['vgenome'] = None

trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                options['vqa'],
                                options['coco'],
                                options['vgenome'])

train_loader = trainset.data_loader(batch_size=options['optim']['batch_size'],
                                    num_workers=1,
                                    shuffle=True)

print("Loaded trainset")

train_examples_list = pickle.load(open('data/vqa2/processed/nans,2000_maxlength,26_minwcount,0_nlp,mcb_pad,right_trainsplit,train/trainset.pickle', 'rb'))
qid_to_example = {ex['question_id']: ex for ex in train_examples_list}

comp_pairs = json.load(open("data/vqa2/raw/annotations/v2_mscoco_train2014_complementary_pairs.json", "r"))
comp_q = {}
for q1, q2 in comp_pairs:
    comp_q[q1] = q2
    comp_q[q2] = q1

knns = np.load("data/coco/extract/arch,fbresnet152_size,448/knn/knn_results_trainset.npy").reshape(1)[0]
knn_idx = knns["indices"]

f = h5py.File('data/coco/extract/arch,fbresnet152_size,448/trainset.hdf5', 'r')
features = f.get('noatt')

print("Loaded KNN features")

for i, sample in tqdm(enumerate(train_loader)):

    # Get KNN features for original images
    iids_orig = [trainset.dataset_img.name_to_index[image_name] for image_name in sample['image_name']]
    knns_batch = [list(knn_idx[i]) for i in iids_orig]
    # knn_features = [[features[i] for i in knns_batch[j]] for j in range(len(knns_batch))]

    # Get complementary questions
    no_compliment = 0
    missing_example = 0

    qids_comp = []
    for qid in sample['question_id']:
        if qid in comp_q:
            qids_comp.append(comp_q[qid])
        else:
            qids_comp.append(None)
            no_compliment += 1

    image_names_comp = []
    for qid in qids_comp:
        if not qid:
            image_names_comp.append(None)
            continue
        if qid not in qid_to_example:
            image_names_comp.append(None)
            missing_example += 1
            continue

        image_names_comp.append(qid_to_example[qid]['image_name'])

    iids_comp = [trainset.dataset_img.name_to_index[name] if name is not None else None for name in image_names_comp]

    good_example_idxs = []
    for i, iid in enumerate(iids_comp):
        if iid is not None:
            if iid in knns_batch[i]:
                good_example_idxs.append(i)

    print(len(image_names_comp), no_compliment, missing_example)
    print(len(good_example_idxs))

    knn_features = {j : [features[i] for i in knns_batch[j]] for j in good_example_idxs]}

    good_examples = []
    for i in good_example_idxs:
        good_examples.append({
            'v_orig': sample['visual'][good_example_idxs]
            'q_orig': sample['question'][good_example_idxs]
        })
