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


parser = argparse.ArgumentParser()

parser.add_argument('--path_opt', default='options/vqa2/counterexamples_default.yaml',
                    type=str, help='path to a yaml options file')

parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')

def main():

    args = parser.parse_args()

    #########################################################################################
    # Create options
    #########################################################################################

    options = {
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }

    with open(args.path_opt, 'r') as handle:
        options_yaml = yaml.load(handle)
    options = utils.update_values(options, options_yaml)
    options['vgenome'] = None

    #########################################################################################
    # Create datasets
    #########################################################################################

    print('=> Loading VQA dataset...')
    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'],
                                    options['coco'],
                                    options['vgenome'])

    train_loader = trainset.data_loader(batch_size=options['optim']['batch_size'],
                                        num_workers=1,
                                        shuffle=True)

    train_examples_list = pickle.load(open(options['vqa']['path_trainset'], 'rb'))
    q_id_to_example = {ex['question_id']: ex for ex in train_examples_list}

    comp_pairs = json.load(open(options['vqa']['path_comp_pairs'], 'r'))
    q_to_comp = {}
    for q1, q2 in comp_pairs:
        q_to_comp[q1] = q2
        q_to_comp[q2] = q1

    print('=> Loading KNN data...')
    knns = np.load(os.path.join(options['coco']['path_knn'], 'knn_results_trainset.npy')).reshape(1)[0]

    print('=> Loading COCO image features...')
    f = h5py.File(os.path.join(options['coco']['path_raw'], 'trainset.hdf5'), 'r')
    features = f.get('noatt')

    for s_idx, sample in tqdm(enumerate(train_loader)):

        orig, comp, neighbors = buildTrainExample(sample, trainset, features, knns, q_id_to_example, q_to_comp)


def buildTrainExample(sample, dataset, features, knns, q_id_to_example, q_to_comp):

    # Get KNNs for original image
    v_ids_orig = [dataset.dataset_img.name_to_index[image_name] for image_name in sample['image_name']]
    knns_batch = [list(knns['indices'][i]) for i in v_ids_orig]

    # Get complementary questions
    q_ids_comp = []
    err_no_comp = 0
    for q_id in sample['question_id']:
        if q_id in q_to_comp:
            q_ids_comp.append(q_to_comp[q_id])
        else:
            q_ids_comp.append(None)
            err_no_comp += 1

    # Get complementary images
    image_names_comp = []
    err_no_ex = 0
    for q_id in q_ids_comp:
        if not q_id:
            image_names_comp.append(None)
            continue
        if q_id not in q_id_to_example:
            image_names_comp.append(None)
            err_no_ex += 1
            continue
        image_names_comp.append(q_id_to_example[q_id]['image_name'])

    v_ids_comp = [dataset.dataset_img.name_to_index[name] if name is not None else None for name in image_names_comp]

    good_idxs = []
    comp_idxs = []
    err_no_knn = 0
    for i, v_id in enumerate(v_ids_comp):
        if v_id is not None:
            if v_id in knns_batch[i]:
                good_idxs.append(i)
                comp_idxs.append(knns_batch[i].index(v_id))
            else:
                err_no_knn += 1

    if len(good_idxs) == 0:
        continue

    # Get KNN features for good examples
    knn_features = [np.array([features[i] for i in knns_batch[j]]) for j in good_idxs]

    orig = {
        # Index of original image within knns is always zero
        'idxs': np.zeros(len(good_idxs)),
        'q': sample['question'][good_idxs],
        'q_id': sample['question_id'][good_idxs],
        'a': sample['question'][good_idxs],
    }
    comp = {
        # Index (0 - 24) of comp within knns of original image
        'idxs': comp_idxs,
        'q_id': [q_ids_comp[q_id] for q_id in good_idxs]
    }
    neighbors = {
        'v': knn_features,
    }

    # print('Missing q_comp: {}'.format(err_no_comp))
    # print('Missing ex: {}'.format(err_no_ex))
    # print('Comp not in KNNs: {}'.format(err_no_knn))
    # print('Total: {} / {}'.format(len(good_idxs), len(sample['image_name'])))

    return orig, comp, neighbors


if __name__ == '__main__':
    main()
