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
from vqa.models.cx import RandomBaseline, DistanceBaseline, BlackBox, LinearContext, PairwiseModel

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

parser.add_argument('--pairwise', action='store_true', help='Pairwise training')
parser.add_argument('--trainable_vqa', action='store_true', help='If true, backprop through VQA model')

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
    if args.dev_mode:
        trainset_fname = 'trainset_augmented_small.pickle'
    else:
        trainset_fname = 'trainset_augmented.pickle'
    trainset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'pickle_old', trainset_fname), 'rb'))

    # if not args.dev_mode:
    valset_fname = 'valset_augmented_small.pickle'
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
    cx_model = PairwiseModel(vqa_model, knn_size=2)

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
    print('=> Starting training...')

    if args.pairwise:
        print('==> Pairwise training')

    optimizer = torch.optim.Adam(cx_model.parameters(), lr=options['optim']['lr'])

    for epoch in range(start_epoch, options['optim']['epochs'] + 1):

        cx_model.train()
        if args.trainable_vqa:
            vqa_model.eval()
        else:
            vqa_model.train()

        train_b = 0

        criterion = nn.CrossEntropyLoss(size_average=False)

        trainset_batched = batchify(trainset['examples_list'], batch_size=options['optim']['batch_size'])
        for batch in tqdm(trainset_batched):
            assert(cx_model.training)

            image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(batch, features_train, trainset['name_to_index'], pairwise=args.pairwise)

            scores = cx_model(image_features, question_wids, answer_aids)

            loss = criterion(scores, comp_idxs) / len(batch)

            if args.pairwise:
                correct = recallAtK(scores, comp_idxs, k=1)
            else:
                correct = recallAtK(scores, comp_idxs, k=5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_b += 1

            if train_b % args.print_freq == 0:
                log_results(train_writer, mode='train', epoch=epoch, i=((epoch - 1) * len(trainset_batched)) + train_b, loss=float(loss), recall=(correct.sum() / len(batch)))

            if (args.eval_freq > 0 and train_b % args.eval_freq == 0) or train_b == len(trainset_batched):
                eval_results = eval_model(cx_model, valset, features_val, options['optim']['batch_size'], pairwise=args.pairwise)
                log_results(val_writer, mode='val', epoch=epoch, i=((epoch - 1) * len(trainset_batched)) + train_b, loss=eval_results['loss'], recall=eval_results['recall'])

        info.append(eval_results)

        if info[-1]['recall'] > best_recall:
            is_best = True
            best_recall = info[-1]['recall']
        else:
            is_best = False

        save_cx_checkpoint(cx_model, info, save_dir, is_best=is_best)


def eval_model(cx_model, valset, features_val, batch_size, pairwise=False):
    cx_model.eval()

    val_i = val_correct = val_loss = 0

    criterion = nn.CrossEntropyLoss(size_average=False)

    valset_batched = batchify(valset['examples_list'], batch_size=batch_size)
    for batch in tqdm(valset_batched):
        image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(batch, features_val, valset['name_to_index'], pairwise)
        scores = cx_model(image_features, question_wids, answer_aids)

        val_loss += float(criterion(scores, comp_idxs))

        if pairwise:
            correct = recallAtK(scores, comp_idxs, k=1)
        else:
            correct = recallAtK(scores, comp_idxs, k=5)
        val_correct += correct.sum()

        val_i += len(batch)

    results = {
        'loss': (val_loss / val_i),
        'recall': (val_correct / val_i)
    }

    cx_model.train()

    return results


def log_results(writer, mode, epoch, i, loss, recall):
    print("Epoch {} {}: Loss: {:.2f}, Recall: {:.4f}".format(epoch, mode, loss, recall))
    writer.add_scalar('loss', loss, i)
    writer.add_scalar('recall', recall, i)


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


def getDataFromBatch(batch, features, name_to_index, pairwise=False):
    image_idxs = []
    question_wids = []
    answer_aids = []
    comp_idxs = []

    for ex in batch:
        image_idx = [name_to_index[ex['image_name']]]
        knn_idxs = [name_to_index[name] for name in ex['knns']]
        if pairwise:
            comp_idx = knn_idxs[ex['comp']['knn_index']]
            knn_idxs.remove(comp_idx)
            # TODO: Weight this sample
            other_idx = random.choice(knn_idxs)
            knn_idxs = [comp_idx, other_idx]
        image_idxs.append(image_idx + knn_idxs)
        question_wids.append(ex['question_wids'])
        answer_aids.append(ex['answer_aid'])
        comp_idxs.append(ex['comp']['knn_index'])

    if pairwise:
        comp_idxs = [0 for _ in range(len(batch))]

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

    return info, last_epoch + 1, info[-1]['recall']


def check_grad(cx_model):
    for name, p in cx_model.named_parameters():
        if p.grad is not None:
            print(name, p.grad.norm())
        else:
            print(name, "None")


if __name__ == '__main__':
    main()
