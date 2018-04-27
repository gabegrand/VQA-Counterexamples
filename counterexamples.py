import argparse
import click
import h5py
import json
import numpy as np
import os
import pickle
import random
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
from tensorboard import SummaryWriter

import vqa.lib.engine as engine
import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.lib.criterions as criterions
import vqa.datasets as datasets
import vqa.models as models
from vqa.models.cx import RandomBaseline, DistanceBaseline, BlackBox, LinearContext

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

parser.add_argument('-c', '--comment', type=str)

parser.add_argument('-dev', '--dev_mode', action='store_true')

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
    if args.dev_mode:
        trainset_fname = 'trainset_augmented_small.pickle'
        valset_fname = 'valset_augmented_small.pickle'
    else:
        trainset_fname = 'trainset_augmented.pickle'
        valset_fname = 'valset_augmented.pickle'
    trainset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], trainset_fname), 'rb'))

    if not args.dev_mode:
        valset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], valset_fname), 'rb'))

    print('=> Loading KNN data...')
    knns = json.load(open(options['coco']['path_knn'], 'r'))
    knns = {int(k):v for k,v in knns.items()}

    print('=> Loading COCO image features...')
    features_train = h5py.File(os.path.join(options['coco']['path_raw'], 'trainset.hdf5'), 'r').get('noatt')
    features_train = np.array(features_train)

    if not args.dev_mode:
        features_val = h5py.File(os.path.join(options['coco']['path_raw'], 'valset.hdf5'), 'r').get('noatt')
        features_val = np.array(features_val)

    #########################################################################################
    # Create model
    #########################################################################################
    print('=> Building model...')

    # cx_model = DistanceBaseline(knn_size=24)
    vqa_model = models.factory(options['model'],
                               trainset['vocab_words'], trainset['vocab_answers'],
                               cuda=True, data_parallel=True)

    if not args.dev_mode:
        start_epoch, best_acc1, exp_logger = load_checkpoint(vqa_model.module, None, os.path.join(options['logs']['dir_logs'], 'best'))

    # cx_model = BlackBox(vqa_model, knn_size=24)
    cx_model = LinearContext(vqa_model, knn_size=24)
    cx_model.cuda()

    #########################################################################################
    # Train loop
    #########################################################################################

    optimizer = torch.optim.Adam(cx_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    for epoch in range(0, options['optim']['epochs']):

        # TRAIN
        total_examples = total_correct = 0
        cx_model.train()

        for batch in tqdm(batchify(trainset['examples_list'], batch_size=options['optim']['batch_size'])):
            batch_size = len(batch)
            image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(batch, features_train, trainset['name_to_index'])

            scores = cx_model(image_features, question_wids, answer_aids)

            loss = criterion(scores, comp_idxs)

            correct = recallAtK(scores, comp_idxs, k=5)

            total_examples += batch_size
            total_correct += correct.sum()

            loss.backward()
            optimizer.step()

            # for name, p in cx_model.named_parameters():
            #     print(name, p.grad)

            if total_examples % 100 == 0 or total_examples == len(trainset['examples_list']):
                recall = total_correct / total_examples
                print("Epoch {} ({}/{}): Loss: {:.2f}, Recall@5: {:.4f}".format(epoch, total_examples, len(trainset['examples_list']), float(loss), recall))

                ex_num = (epoch * len(trainset['examples_list'])) + total_examples
                writer.add_scalar('recall@5', recall, ex_num)
                writer.add_scalar('loss_train', loss, ex_num)


       #TODO: Val


def batchify(example_list, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(example_list)
    batched_dataset = []
    for i in range(0, len(example_list), batch_size):
        batched_dataset.append(example_list[i:min(i + batch_size, len(example_list))])
    return batched_dataset


def getDataFromBatch(batch, features, name_to_index):
    image_idxs = []
    question_wids = []
    answer_aids = []
    comp_idxs = []

    for ex in batch:
        image_idx = [name_to_index[ex['image_name']]]
        knn_idxs = [name_to_index[name] for name in ex['knns']]
        image_idxs.append(image_idx + knn_idxs)
        question_wids.append(ex['question_wids'])
        answer_aids.append(ex['answer_aid'])
        comp_idxs.append(ex['comp']['knn_index'])

    image_features = torch.from_numpy(np.array([features[idxs] for idxs in image_idxs])).cuda()
    question_wids = torch.LongTensor(question_wids).cuda()
    answer_aids = torch.LongTensor(answer_aids).cuda()
    comp_idxs = Variable(torch.LongTensor(comp_idxs), requires_grad=False)

    return image_features, question_wids, answer_aids, comp_idxs


def recallAtK(scores, ground_truth, k=5):
    assert(scores.shape[0] == ground_truth.shape[0])
    _, top_idxs = scores.topk(k)
    ground_truth = ground_truth.cpu().data.numpy()
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
