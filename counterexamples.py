import argparse
import click
import h5py
import json
import numpy as np
import os
import pickle
import re
import shutil
import yaml

from IPython.display import Image, display
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.autograd import Variable

import vqa.lib.engine as engine
import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.lib.criterions as criterions
import vqa.datasets as datasets
import vqa.models as models
from vqa.models.cx import RandomBaseline, DistanceBaseline, CXModel, MutanNoAttCX

from train import load_checkpoint


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
    trainset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'trainset_augmented.pickle'), 'rb'))
    # trainset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'trainset_augmented_small.pickle'), 'rb'))

    print('=> Loading KNN data...')
    knns = json.load(open(options['coco']['path_knn'], 'r'))
    knns = {int(k):v for k,v in knns.items()}

    print('=> Loading COCO image features...')
    f = h5py.File(os.path.join(options['coco']['path_raw'], 'trainset.hdf5'), 'r')
    features = f.get('noatt')

    #########################################################################################
    # Create model
    #########################################################################################
    print('=> Building model...')

    # cx_model = RandomBaseline(knn_size=24)
    vqa_model = models.factory(options['model'],
                       trainset['vocab_words'], trainset['vocab_answers'],
                       cuda=True, data_parallel=True)

    start_epoch, best_acc1, exp_logger = load_checkpoint(vqa_model.module, None, os.path.join(options['logs']['dir_logs'], 'best'))

    cx_model = CXModel(vqa_model, knn_size=24)

    #########################################################################################
    # Train loop
    #########################################################################################

    for epoch in range(0, options['optim']['epochs']):

        # TRAIN
        total_examples = total_correct = 0
        s_idx = 0

        for ex in tqdm(trainset['examples_list']):

            batch_size = 1

            # Get image features
            # TODO: if these load slowly, put features on CPU
            image_features = torch.from_numpy(features[trainset['name_to_index'][ex['image_name']]]).view(batch_size, -1).cuda()
            knn_idx = sorted([trainset['name_to_index'][name] for name in ex['knns']])
            knn_features = torch.from_numpy(features[knn_idx]).view(batch_size, 24, -1).cuda()

            # Put other tensors on GPU
            # TODO: unclear if necessary
            question_wids = torch.LongTensor(ex['question_wids']).view(batch_size, -1).cuda()
            answer_aid = [ex['answer_aid']]

            scores = cx_model(image_features, knn_features, question_wids, answer_aid)

            correct = recallAtK(scores, np.array(ex['comp']['knn_index']), k=5)

            total_examples += batch_size
            total_correct += correct.sum()

            if s_idx % 100 == 0:
                print("Epoch {} ({}/{}): Recall@5: {:.4f}".format(
                       epoch, s_idx, len(trainset['examples_list']), total_correct / total_examples))

            s_idx += 1

        # # VAL
        # for sample in tqdm(val_loader):
        #
        #     example = buildTrainExample(sample, valset, features, knns, q_id_to_example, q_to_comp)
        #     if example is None:
        #         continue


def recallAtK(scores, ground_truth, k=5):
    _, top_idxs = scores.topk(k)
    return (Tensor(ground_truth.reshape((-1, 1))).expand_as(top_idxs).numpy() == \
            top_idxs.cpu().data.numpy()).sum(axis=1)


def coco_name_to_num(name):
    assert(name[-4:] == '.jpg')
    assert(name[-17] == '_')
    return int(name[-16:-4])


def coco_num_to_name(num, split='train'):
    if len(str(num)) > 12:
        raise ValueError
    if split == 'train':
        return 'COCO_train2014_{}.jpg'.format(str(num).zfill(12))
    elif split == 'val':
        return 'COCO_val2014_{}.jpg'.format(str(num).zfill(12))
    else:
        raise ValueError('split must be train or val; got {}'.format(split))


if __name__ == '__main__':
    main()
