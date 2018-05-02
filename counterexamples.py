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

from datetime import datetime
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
from vqa.models.cx import RandomBaseline, DistanceBaseline, BlackBox, LinearContext, CXModelBase

from train import load_checkpoint as load_vqa_checkpoint


parser = argparse.ArgumentParser()

parser.add_argument('--path_opt', default='options/vqa2/counterexamples_default.yaml',
                    type=str, help='path to a yaml options file')

parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--best', action='store_true',
                    help='whether to resume best checkpoint')

parser.add_argument('-c', '--comment', type=str, default='')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    help='print frequency')
parser.add_argument('-v', '--eval_freq', default=-1, type=int,
                    help='eval frequency')

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
    # Bookkeeping
    #########################################################################################

    if args.resume:
        run_name = args.resume
        save_dir = os.path.join('logs', 'cx', run_name)
        assert(os.path.isdir(save_dir))

        i = 1
        log_dir = os.path.join('runs', run_name, 'resume_{}'.format(i))
        while(os.path.isdir(log_dir)):
            i += 1
            log_dir = os.path.join('runs', run_name, 'resume_{}'.format(i))
    else:
        run_name = datetime.now().strftime('%b%d-%H-%M-%S')
        if args.comment:
            run_name += '_' + args.comment
        save_dir = os.path.join('logs', 'cx', run_name)
        if os.path.isdir(save_dir):
            if click.confirm('Save directory already exists in {}. Erase?'.format(save_dir)):
                os.system('rm -r ' + save_dir)
            else:
                return
        os.makedirs(os.path.join(save_dir, 'ckpt'))
        os.makedirs(os.path.join(save_dir, 'best'))
        # Tensorboard log directory
        log_dir = os.path.join('runs', run_name)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    print('Saving model to {}'.format(save_dir))
    print('Logging results to {}'.format(log_dir))

    #########################################################################################
    # Create datasets
    #########################################################################################

    print('=> Loading VQA dataset...')
    trainset_fname = 'trainset_augmented.pickle'
    trainset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'pickle_old', trainset_fname), 'rb'))

    # if not args.dev_mode:
    valset_fname = 'valset_augmented.pickle'
    valset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'pickle_old', valset_fname), 'rb'))

    print('=> Loading KNN data...')
    knns = json.load(open(options['coco']['path_knn'], 'r'))
    knns = {int(k):v for k,v in knns.items()}

    print('=> Loading COCO image features...')
    features_train = h5py.File(os.path.join(options['coco']['path_raw'], 'trainset.hdf5'), 'r').get('noatt')
    features_train = np.array(features_train)

    # if not args.dev_mode:
    features_val = h5py.File(os.path.join(options['coco']['path_raw'], 'valset.hdf5'), 'r').get('noatt')
    features_val = np.array(features_val)

    #########################################################################################
    # Create model
    #########################################################################################
    print('=> Building model...')

    vqa_model = models.factory(options['model'],
                               trainset['vocab_words'], trainset['vocab_answers'],
                               cuda=True, data_parallel=True)
    vqa_model = vqa_model.module

    load_vqa_checkpoint(vqa_model, None, os.path.join(options['logs']['dir_logs'], 'best'))

    # cx_model = LinearContext(vqa_model, knn_size=24)
    cx_model = CXModelBase(vqa_model, knn_size=24)

    if args.resume:
        info, start_epoch, best_recall = load_cx_checkpoint(cx_model, save_dir, resume_best=args.best)
    else:
        info = []
        start_epoch = 1
        best_recall = 0

    cx_model.cuda()

    #########################################################################################
    # Train loop
    #########################################################################################
    with h5py.File("vqa_trainset_cached.hdf5", "w") as f:
        a_dset = f.create_dataset("answers", (len(trainset['examples_list']), 25, 2000), 'f')
        y_dset = f.create_dataset("context", (len(trainset['examples_list']), 25, 360), 'f')
        q_dset = f.create_dataset("q_ids", (len(trainset['examples_list']),), 'i')

        print('=> Starting training...')
        for epoch in range(start_epoch, options['optim']['epochs'] + 1):

            train_b = 0

            trainset_batched = batchify(trainset['examples_list'], batch_size=options['optim']['batch_size'], shuffle=False)
            for batch in tqdm(trainset_batched):
                # TRAIN
                cx_model.eval()
                vqa_model.eval()
                for p in vqa_model.parameters():
                    p.requires_grad = False

                image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(batch, features_train, trainset['name_to_index'])

                bsz = image_features.size(0)

                a_orig, y_orig, a_knns, y_knns = cx_model.vqa_forward(image_features, question_wids)

                a = torch.cat((a_orig.view(-1, 1, 2000), a_knns), dim=1).data.cpu()
                y = torch.cat((y_orig.view(-1, 1, 360), y_knns), dim=1).data.cpu()

                a_dset[64 * train_b:(64 * train_b)+bsz] = a
                y_dset[64 * train_b:(64 * train_b)+bsz] = y
                q_dset[64 * train_b:(64 * train_b)+bsz] = np.array([ex['question_id'] for ex in batch])

                train_b += 1

    with h5py.File("vqa_valset_cached.hdf5", "w") as f:
        a_dset = f.create_dataset("answers", (len(valset['examples_list']), 25, 2000), 'f')
        y_dset = f.create_dataset("context", (len(valset['examples_list']), 25, 360), 'f')
        q_dset = f.create_dataset("q_ids", (len(valset['examples_list']),), 'i')

        print('=> Starting training...')
        for epoch in range(start_epoch, options['optim']['epochs'] + 1):

            val_b = 0

            valset_batched = batchify(valset['examples_list'], batch_size=options['optim']['batch_size'], shuffle=False)
            for batch in tqdm(valset_batched):
                # TRAIN
                cx_model.eval()
                vqa_model.eval()
                for p in vqa_model.parameters():
                    p.requires_grad = False

                image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(batch, features_val, valset['name_to_index'])

                bsz = image_features.size(0)

                a_orig, y_orig, a_knns, y_knns = cx_model.vqa_forward(image_features, question_wids)

                a = torch.cat((a_orig.view(-1, 1, 2000), a_knns), dim=1).data.cpu()
                y = torch.cat((y_orig.view(-1, 1, 360), y_knns), dim=1).data.cpu()

                a_dset[64 * val_b:(64 * val_b)+bsz] = a
                y_dset[64 * val_b:(64 * val_b)+bsz] = y
                q_dset[64 * val_b:(64 * val_b)+bsz] = np.array([ex['question_id'] for ex in batch])

                val_b += 1


def eval_model(cx_model, valset, features_val, batch_size):
    cx_model.eval()

    val_i = val_correct = val_loss = 0

    criterion = nn.CrossEntropyLoss(size_average=False)

    valset_batched = batchify(valset['examples_list'], batch_size=batch_size)
    for batch in tqdm(valset_batched):
        image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(batch, features_val, valset['name_to_index'])
        scores = cx_model(image_features, question_wids, answer_aids)

        val_loss += float(criterion(scores, comp_idxs))
        correct = recallAtK(scores, comp_idxs, k=5)
        val_correct += correct.sum()

        val_i += len(batch)

    results = {
        'loss': (val_loss / val_i),
        'recall_5': (val_correct / val_i)
    }

    return results


def log_results(writer, mode, epoch, i, loss, recall):
    print("Epoch {} {}: Loss: {:.2f}, Recall@5: {:.4f}".format(epoch, mode, loss, recall))
    writer.add_scalar('loss', loss, i)
    writer.add_scalar('recall_5', recall, i)


def recallAtK(scores, ground_truth, k=5):
    assert(scores.shape[0] == ground_truth.shape[0])
    _, top_idxs = scores.topk(k)
    ground_truth = ground_truth.cpu().data.numpy()
    return (Tensor(ground_truth.reshape((-1, 1))).expand_as(top_idxs).numpy() == \
            top_idxs.cpu().data.numpy()).sum(axis=1)


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

    # TODO: Make call to .cuda() conditional on model type
    image_features = torch.from_numpy(np.array([features[idxs] for idxs in image_idxs])).cuda()
    question_wids = torch.LongTensor(question_wids).cuda()
    answer_aids = torch.LongTensor(answer_aids).cuda()
    comp_idxs = Variable(torch.LongTensor(comp_idxs), requires_grad=False).cuda()

    return image_features, question_wids, answer_aids, comp_idxs


def save_cx_checkpoint(cx_model, info, save_dir, is_best=True):
    path_ckpt_model = os.path.join(save_dir, 'ckpt', 'model.ckpt')
    path_ckpt_info = os.path.join(save_dir, 'ckpt', 'info.ckpt')
    path_best_model = os.path.join(save_dir, 'best', 'model.ckpt')
    path_best_info = os.path.join(save_dir, 'best', 'info.ckpt')
    torch.save(cx_model.state_dict(), path_ckpt_model)
    torch.save(info, path_ckpt_info)
    if is_best:
        shutil.copyfile(path_ckpt_model, path_best_model)
        shutil.copyfile(path_ckpt_info, path_best_info)
    print('{}Saved checkpoint to {}'.format('* ' if is_best else '', save_dir))


def load_cx_checkpoint(cx_model, save_dir, resume_best=True):
    if resume_best:
        path_ckpt_model = os.path.join(save_dir, 'best', 'model.ckpt')
        path_ckpt_info = os.path.join(save_dir, 'best', 'info.ckpt')
    else:
        path_ckpt_model = os.path.join(save_dir, 'ckpt', 'model.ckpt')
        path_ckpt_info = os.path.join(save_dir, 'ckpt', 'info.ckpt')

    model_state = torch.load(path_ckpt_model)
    cx_model.load_state_dict(model_state)
    print('Loaded model from {}'.format(path_ckpt_model))

    info = torch.load(path_ckpt_info)
    assert(len(info) > 0)
    last_epoch = len(info)
    print('Epoch {}: {}'.format(last_epoch, info[-1]))

    return info, last_epoch + 1, info[-1]['recall_5']


if __name__ == '__main__':
    main()
