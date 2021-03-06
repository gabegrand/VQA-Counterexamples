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
from vqa.models.cx import (RandomBaseline, DistanceBaseline, BlackBox,
    LinearContext, PairwiseModel, PairwiseLinearModel, SemanticBaseline,
    SimilarityModel, NeuralModel)

from train import load_checkpoint as load_vqa_checkpoint

from cx_visu import viz_knns, viz_qa


parser = argparse.ArgumentParser()

parser.add_argument('--path_opt', default='options/cx/counterexamples_default.yaml',
                    type=str, help='path to a yaml options file')

parser.add_argument('-cx', '--cx_model', required=True,
                    type=str, help='Counterexample model type')

parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-lb', '--sb_lambda', type=float,
                    help='semantic baseline lambda')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')

parser.add_argument('--project_dir', default='/datadrive/vqa.pytorch/', type=str,
                    help='path to project root whose data to use')

parser.add_argument('--resume', default='', type=str,
                    help='run name to resume')
parser.add_argument('--best', action='store_true',
                    help='whether to resume best checkpoint')

parser.add_argument('-c', '--comment', type=str, default='')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    help='print frequency')
parser.add_argument('-v', '--eval_freq', default=-1, type=int,
                    help='eval frequency')
parser.add_argument('-t', '--test', action='store_true', help='Run eval on full testset after training')
parser.add_argument('--viz', action='store_true', help='Run viz on valset after training')


parser.add_argument('--pairwise', action='store_true',
                    help='Pairwise training')

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--pretrained_vqa',
                   dest='pretrained_vqa', action='store_true')
group.add_argument('--untrained_vqa', dest='pretrained_vqa',
                   action='store_false')

parser.set_defaults(pretrained_vqa=True)

parser.add_argument('--trainable_vqa', action='store_true',
                    help='If true, backprop through VQA model')

parser.add_argument('-dev', '--dev_mode', action='store_true')


def main():

    args = parser.parse_args()

    ##########################################################################
    # Create options
    ##########################################################################


    # Parse options
    options = {
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        },
        'cx_model': {
            'pretrained_vqa': args.pretrained_vqa,
            'trainable_vqa': args.trainable_vqa,
        }
    }

    with open(args.path_opt, 'r') as handle:
        options_yaml = yaml.load(handle)
    options = utils.update_values(options, options_yaml)
    options['vgenome'] = None


    # Ensure determinism
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Bookkeeping
    ##########################################################################

    if args.cx_model == 'NeuralModel' and not args.comment:
        args.comment = options['cx_model']['name']

    if args.resume:
        print('hello')
        run_name = args.resume
        save_dir = os.path.join(args.project_dir, 'logs', 'cx', run_name)
        assert(os.path.isdir(save_dir))

        i = 1
        log_dir = os.path.join(args.project_dir, 'runs', run_name, 'resume_{}'.format(i))
        while(os.path.isdir(log_dir)):
            i += 1
            log_dir = os.path.join(args.project_dir, 'runs', run_name, 'resume_{}'.format(i))
    else:
        print('why am i here')
        run_name = datetime.now().strftime('%b%d-%H-%M-%S')
        if args.comment:
            run_name += '_' + args.comment

        save_dir = os.path.join(args.project_dir, 'logs', 'cx', run_name)

        if os.path.isdir(save_dir):
            if click.confirm('Save directory already exists in {}. Erase?'.format(save_dir)):
                os.system('rm -r ' + save_dir)
            else:
                return
        os.makedirs(os.path.join(save_dir, 'ckpt'))
        os.makedirs(os.path.join(save_dir, 'best'))
        # Tensorboard log directory
        log_dir = os.path.join(args.project_dir, 'runs', run_name)

    if args.viz:
        viz_dir = os.path.join(args.project_dir, 'viz', 'cx', run_name)
        print('viz_dir', viz_dir, run_name)
        if os.path.isdir(viz_dir):
            if click.confirm('Viz directory already exists in {}. Erase?'.format(viz_dir)):
                os.system('rm -r ' + viz_dir)
            else:
                return
        os.makedirs(viz_dir)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    print('Saving model to {}'.format(save_dir))
    print('Logging results to {}'.format(log_dir))

    ##########################################################################
    # Create datasets
    ##########################################################################

    print('=> Loading VQA dataset...')


    if args.dev_mode:
        trainset_fname = 'trainset_augmented_small.pickle'
    else:
        trainset_fname = 'trainset_augmented.pickle'
    trainset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'pickle_old', trainset_fname), 'rb'))

    valset_fname = 'valset_augmented_small.pickle'
    valset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'pickle_old', valset_fname), 'rb'))

    if args.test:
        testset_fname = 'valset_augmented.pickle'
        testset = pickle.load(open(os.path.join(options['vqa']['path_trainset'], 'pickle_old', testset_fname), 'rb'))

    print('=> Loading KNN data...')
    knns = json.load(open(options['coco']['path_knn'], 'r'))
    knns = {int(k): v for k, v in knns.items()}

    print('=> Loading COCO image features...')
    features_train = h5py.File(os.path.join(
        options['coco']['path_features'], 'trainset.hdf5'), 'r').get('noatt')
    features_train = np.array(features_train)

    # if not args.dev_mode:
    features_val = h5py.File(os.path.join(
        options['coco']['path_features'], 'valset.hdf5'), 'r').get('noatt')

    features_val = np.array(features_val)

    ##########################################################################
    # Create model
    ##########################################################################
    print('=> Building model...')

    vqa_model = None
    optimizer = None
    if args.cx_model == "RandomBaseline":
        cx_model = RandomBaseline(knn_size=24)
    elif args.cx_model == "DistanceBaseline":
        cx_model = DistanceBaseline(knn_size=24)
    else:
        vqa_model = models.factory(options['model'],
                                   trainset['vocab_words'], trainset[
                                       'vocab_answers'],
                                   cuda=True, data_parallel=True)
        vqa_model = vqa_model.module
        if args.pretrained_vqa:
            load_vqa_checkpoint(vqa_model, None, os.path.join(
                options['logs']['dir_logs'], 'best'))

        if args.cx_model == "BlackBox":
            cx_model = BlackBox(vqa_model, knn_size=24,
                                trainable_vqa=args.trainable_vqa)
        elif args.cx_model == "LinearContext":
            cx_model = LinearContext(
                vqa_model, knn_size=24, trainable_vqa=args.trainable_vqa)
        elif args.cx_model == "SemanticBaseline":
            if args.sb_lambda is None:
                raise ValueError(
                    "If semantic baseline is selected then --sb_lambda must also be provided.")
            cx_model = SemanticBaseline(
                vqa_model, knn_size=24, trainable_vqa=args.trainable_vqa)

            cx_model.set_lambda(args.sb_lambda)
            emb = pickle.load(open(os.path.join(
                options['vqa']['path_trainset'], "answer_embedding.pickle"), 'rb'))
            cx_model.set_answer_embedding(emb)
        elif args.cx_model == "NeuralModel":
            model_spec = options['cx_model']

            if model_spec['pretrained_emb']:
                emb = pickle.load(open(os.path.join(options['vqa']['path_trainset'], "answer_embedding.pickle"), 'rb'))
            else:
                emb = None

            cx_model = NeuralModel(model_spec=model_spec,
                                   dim_h=model_spec['dim_h'],
                                   n_layers=model_spec['n_layers'],
                                   emb=emb,
                                   drop_p=model_spec['drop_p'],
                                   vqa_model=vqa_model,
                                   knn_size=24,
                                   trainable_vqa=model_spec['trainable_vqa'])

        elif args.cx_model == "PairwiseModel":
            assert(args.pairwise)
            cx_model = PairwiseModel(
                vqa_model, knn_size=2, trainable_vqa=args.trainable_vqa)
        elif args.cx_model == "PairwiseLinearModel":
            cx_model = PairwiseLinearModel(
                vqa_model, knn_size=24, trainable_vqa=args.trainable_vqa)

        else:
            raise ValueError("Unrecognized cx_model {}".format(args.cx_model))

        optimizer = torch.optim.Adam(
            cx_model.parameters(), lr=options['optim']['lr'])

    print("Built {}".format(args.cx_model))

    if args.resume:
        info, start_epoch, best_recall = load_cx_checkpoint(
            cx_model, save_dir, resume_best=args.best)
    else:
        info = []
        start_epoch = 1
        best_recall = 0

    cx_model.cuda()

    ##########################################################################
    # Train loop
    ##########################################################################
    print('=> Starting training...')

    if args.pairwise:
        print('==> Pairwise training')

    epoch = None
    for epoch in range(start_epoch, options['optim']['epochs'] + 1):

        cx_model.train()
        if vqa_model is not None:
            if options['cx_model']['trainable_vqa']:
                vqa_model.train()
            else:
                vqa_model.eval()

        train_b = 0

        criterion = nn.CrossEntropyLoss(size_average=False)

        trainset_batched = batchify(trainset['examples_list'], batch_size=options[
                                    'optim']['batch_size'])
        for batch in tqdm(trainset_batched):
            assert(cx_model.training)
            if vqa_model is not None:
                if options['cx_model']['trainable_vqa']:
                    assert(vqa_model.training)
                else:
                    assert(not vqa_model.training)

            image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(
                batch, features_train, trainset['name_to_index'], pairwise=args.pairwise)

            scores = cx_model(image_features, question_wids, answer_aids)

            if args.pairwise:
                assert(scores.size(1) == 2)
                zeros = Variable(torch.LongTensor([0] * len(batch))).cuda()
                loss = criterion(scores, zeros) / len(batch)
                correct = recallAtK(scores, zeros, k=1)
            else:
                correct = recallAtK(scores, comp_idxs, k=5)
                loss = criterion(scores, comp_idxs) / len(batch)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_b += 1

            if train_b % args.print_freq == 0:
                if args.pairwise:
                    metrics = {
                        'loss_pairwise': float(loss),
                        'acc_pairwise': (correct.sum() / len(batch))
                    }
                else:
                    metrics = {
                        'loss': float(loss),
                        'recall': (correct.sum() / len(batch))
                    }
                log_results(train_writer, mode='train', epoch=epoch, i=(
                    (epoch - 1) * len(trainset_batched)) + train_b, metrics=metrics)

            if (args.eval_freq > 0 and train_b % args.eval_freq == 0) or train_b == len(trainset_batched):
                eval_results = eval_model(cx_model, valset, features_val, options[
                                          'optim']['batch_size'], pairwise=args.pairwise)
                log_results(val_writer, mode='val', epoch=epoch, i=(
                    (epoch - 1) * len(trainset_batched)) + train_b, metrics=eval_results)

        info.append(eval_results)

        if info[-1]['recall'] > best_recall:
            is_best = True
            best_recall = info[-1]['recall']
        else:
            is_best = False

        save_cx_checkpoint(cx_model, info, save_dir, is_best=is_best)

    if args.test or args.viz:
        if epoch is not None:
            _, best_epoch, _ = load_cx_checkpoint(cx_model, save_dir, resume_best=True)
        else:
            best_epoch = 0

    if args.test:
        test_results = eval_model(cx_model, testset, features_val, options['optim']['batch_size'], pairwise=args.pairwise)
        test_results['best_epoch'] = best_epoch
        with open(os.path.join(save_dir, 'final_results.txt'), 'w') as outfile:
            outstr = json.dumps(test_results)
            outfile.write(outstr)
        print('Saved results to {}'.format(save_dir))
        print('FINAL RESULTS ON BEST EPOCH {}'.format(best_epoch), test_results)
    if args.viz:
        visualize_results(cx_model, valset, features_val, 200,
                          options['coco']['path_val_raw'], viz_dir)



def visualize_results(cx_model, valset, features_val, num_images, datadir, viz_dir):
    print('Building visualizations and saving to', viz_dir)
    cx_model.eval()

    valset_batched = batchify(valset['examples_list'], batch_size=num_images)
    batch = valset_batched[0]
    cx_model.knn_size = 24
    image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(
        batch, features_val, valset['name_to_index'], pairwise=False)
    a_orig, z_orig, a_knns, z_knns, _ = cx_model.vqa_forward(
        image_features, question_wids)
    scores = cx_model(image_features, question_wids, answer_aids).cpu().data.numpy()
    for i, (ex, a, knn, score) in tqdm(enumerate(zip(batch, a_orig, a_knns, scores))):
        img_name = ex['image_name']
        knns = ex['knns']
        question = ex['question']
        answer = ex['answer']
        comp = ex['comp']['image_name']
        comp_answer = ex['comp']['answer']
        a_knns_scores = []
        a_knns_idx = []
        for j in range(24):
            j_a_knns_scores, j_a_knns_idx = knn[j].topk(3)
            j_a_knns_idx = j_a_knns_idx.cpu().data.numpy()
            j_a_knns_scores = j_a_knns_scores.cpu().data.numpy()
            a_knns_scores.append(j_a_knns_scores)
            a_knns_idx.append(j_a_knns_idx)
        # print(list(zip(a_knns_idx, a_knns_scores)))
        a_knns_ws = []
        # print(list(zip(a_knns_idx, a_knns_scores)))
        for idx, knn_score in zip(a_knns_idx, a_knns_scores):
            sublist = []
            for k in range(len(idx)):
                # print('k = ', k)
                # print("appending:", (valset['vocab_answers'][idx[k]], score[k]))
                sublist.append((valset['vocab_answers'][idx[k]], knn_score[k]))
            a_knns_ws.append(sublist)
        # print('len:', len(a_knns_ws))
        # print('a_knns_ws', a_knns_ws)
        # a_knns_ws = [[(valset['vocab_answers'][w[0]],w[1])  for w in idxs] for idxs in zip(a_knns_idx, a_knns_scores)]
        # print(len(score), len(knns), len(a_knns_ws))
        score_knns = sorted(list(zip(score, knns, a_knns_ws)), reverse=True)
        # print('sorted', score_knns)

        knns = [sk[1] for sk in score_knns]
        a_knns_ws = [sk[2] for sk in score_knns]
        # print(knns, a_knns_ws)
        try:
            viz_knns(datadir, img_name, knns, comp, question, answer,
                     24, outfile=os.path.join(viz_dir, 'viz_knns_' + str(i) + '.jpg'))
            viz_qa(datadir, img_name, knns, comp, question, answer, comp_answer,
                   a_knns_ws[:5], 5, outfile=os.path.join(viz_dir, 'viz_qa' + str(i) + '.jpg'))
        except Exception:
            pass
        # except Exception as e:
        #     continue

def eval_model(cx_model, valset, features_val, batch_size, pairwise=False):
    cx_model.eval()

    val_i = val_correct = val_loss = 0
    val_pairwise_correct = val_pairwise_loss = 0

    criterion = nn.CrossEntropyLoss(size_average=False)

    valset_batched = batchify(valset['examples_list'], batch_size=batch_size)
    for batch in tqdm(valset_batched):

        cx_model.knn_size = 24
        image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(
            batch, features_val, valset['name_to_index'], pairwise=False)
        scores = cx_model(image_features, question_wids, answer_aids)
        val_loss += float(criterion(scores, comp_idxs))
        correct = recallAtK(scores, comp_idxs, k=5)
        val_correct += correct.sum()
        val_i += len(batch)

        if pairwise:
            cx_model.knn_size = 2
            image_features, question_wids, answer_aids, comp_idxs = getDataFromBatch(
                batch, features_val, valset['name_to_index'], pairwise=True)
            scores = cx_model(image_features, question_wids, answer_aids)
            zeros = Variable(torch.LongTensor([0] * len(batch))).cuda()
            val_pairwise_loss += float(criterion(scores, zeros))
            val_pairwise_correct += recallAtK(scores, zeros, k=1).sum()

    results = {
        'loss': (val_loss / val_i),
        'recall': (val_correct / val_i),
    }

    if pairwise:
        results['loss_pairwise'] = (val_pairwise_loss / val_i)
        results['acc_pairwise'] = (val_pairwise_correct / val_i)

    cx_model.train()

    return results


def log_results(writer, mode, epoch, i, metrics):
    metrics_str = ''
    for k, v in metrics.items():
        writer.add_scalar(k, v, i)
        metrics_str += '{}: {:.4f}, '.format(k, v)
    print('Epoch {} {}: {}'.format(epoch, mode, metrics_str))


def recallAtK(scores, ground_truth, k=5):
    assert(scores.shape[0] == ground_truth.shape[0])
    _, top_idxs = scores.topk(k)
    ground_truth = ground_truth.cpu().data.numpy()
    return (Tensor(ground_truth.reshape((-1, 1))).expand_as(top_idxs).numpy() ==
            top_idxs.cpu().data.numpy()).sum(axis=1)


def batchify(example_list, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(example_list)
    batched_dataset = []
    for i in range(0, len(example_list), batch_size):
        batched_dataset.append(
            example_list[i:min(i + batch_size, len(example_list))])
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

    # TODO: Make call to .cuda() conditional on model type
    image_features = torch.from_numpy(
        np.array([features[idxs] for idxs in image_idxs])).cuda()
    question_wids = torch.LongTensor(question_wids).cuda()
    answer_aids = torch.LongTensor(answer_aids).cuda()
    comp_idxs = Variable(torch.LongTensor(comp_idxs),
                         requires_grad=False).cuda()

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


def check_grad(cx_model):
    for name, p in cx_model.named_parameters():
        if p.grad is not None:
            print(name, p.grad.norm())
        else:
            print(name, "None")


if __name__ == '__main__':
    main()
