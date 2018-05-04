import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from vqa.models import fusion
from .noatt import AbstractNoAtt


################################################################################
# Baseline models (no VQA)
################################################################################

class RandomBaseline(nn.Module):

    def __init__(self, knn_size):
        super().__init__()

        self.knn_size = knn_size

    def forward(self, image_features, question_wids, answer_aids):
        batch_size = image_features.size(0)
        scores = Variable(torch.rand((batch_size, self.knn_size)))
        return scores


class DistanceBaseline(nn.Module):

    def __init__(self, knn_size):
        super().__init__()

        self.knn_size = knn_size

    def forward(self, image_features, question_wids, answer_aids):
        batch_size = image_features.size(0)
        scores = Variable(Tensor(list(reversed(range(self.knn_size)))).view(1, -1).expand((batch_size, self.knn_size)))
        return scores


################################################################################
# Base class for neural CX models
################################################################################

class CXModelBase(nn.Module):

    def __init__(self, vqa_model, knn_size, cache_train=None, cache_val=None):
        super().__init__()

        self.vqa_model = vqa_model
        self.vqa_model.eval()

        self.cache_train = cache_train
        self.cache_val = cache_val
        self.use_cache = self.cache_train is not None and self.cache_val is not None

        self.knn_size = knn_size

    def vqa_forward(self, image_features, question_wids):
        assert(image_features.size(1) == self.knn_size + 1)
        batch_size = image_features.size(0)

        # Process all image features as a single batch
        image_input = Variable(image_features.view(batch_size * (self.knn_size + 1), -1))

        # Duplicate each question knn_size + 1 times
        question_input = question_wids.view(batch_size, 1, -1).expand(batch_size, self.knn_size + 1, -1).contiguous()
        question_input = Variable(question_input.view(batch_size * (self.knn_size + 1), -1))

        # Run the VQA model
        y, a = self.vqa_model(image_input, question_input)
        a = a.view(batch_size, self.knn_size + 1, -1)
        y = y.view(batch_size, self.knn_size + 1, -1)

        a = a.detach().data
        y = y.detach().data

        # Separate results into original and knns
        a_orig = Variable(a[:, 0, :].contiguous(), requires_grad=True)
        y_orig = Variable(y[:, 0, :].contiguous(), requires_grad=True)
        a_knns = Variable(a[:, 1:, :].contiguous(), requires_grad=True)
        y_knns = Variable(y[:, 1:, :].contiguous(), requires_grad=True)

        return a_orig, y_orig, a_knns, y_knns

    def _vqa_forward_cached(self, image_features, question_wids):
        assert(image_features.size(1) == self.knn_size + 1)
        batch_size = image_features.size(0)
        pass


    def forward(self, image_features, question_wids, answer_aids):
        raise NotImplementedError


################################################################################
# Neural CX models
################################################################################

class BlackBox(CXModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vqa_model.eval()

    def forward(self, image_features, question_wids, answer_aids):
        _, _, a_knns, _ = self.vqa_forward(image_features, question_wids)

        # VQA model's score for the original answer for each KNN
        scores_list = []
        for i, a_idx in enumerate(answer_aids):
            scores_list.append(a_knns[i, :, a_idx])
        scores = torch.stack(scores_list, dim=0)

        print(scores.shape)

        # Flip the sign, since the highest scoring items are the worst counterexamples
        scores = -scores

        return scores


class LinearContext(CXModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: will need to be able to switch this to eval during eval
        self.vqa_model.eval()
        self.dim_y = self.vqa_model.opt['fusion']['dim_mm']

        self.linear = nn.Linear(self.knn_size * self.dim_y, self.knn_size)

    def forward(self, image_features, question_wids, answer_aids):
        a_orig, y_orig, a_knns, y_knns = self.vqa_forward(image_features, question_wids)

        assert(y_knns.requires_grad)

        scores = self.linear(y_knns.view(-1, self.knn_size * self.dim_y))
        # TODO: dropout

        return scores


class PairwiseModel(CXModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: will need to be able to switch this to eval during eval
        self.vqa_model.eval()
        self.dim_v = self.vqa_model.opt['fusion']['dim_v']
        self.dim_q = self.vqa_model.opt['fusion']['dim_q']
        self.dim_y = self.vqa_model.opt['fusion']['dim_mm']

        self.dim_h = 300

        self.linear = nn.Linear((2 * self.dim_v) + self.dim_q, self.dim_h)
        self.out = nn.Linear(self.dim_h, 1)
        self.relu = nn.ReLU()

    def forward(self, image_features, question_wids, answer_aids):
        v_orig = image_features[:, 0]
        v_comp = image_features[:, 1]
        v_other = image_features[:, 2]

        q_emb = self.vqa_model.seq2vec(Variable(question_wids)).detach().data

        input_comp = Variable(torch.cat((v_orig, v_comp, q_emb), dim=1), requires_grad=True)
        input_other = Variable(torch.cat((v_orig, v_other, q_emb), dim=1), requires_grad=True)

        h_comp = self.relu(self.linear(input_comp))
        h_other = self.relu(self.linear(input_other))

        score_comp = self.relu(self.out(h_comp))
        score_other = self.relu(self.out(h_other))

        scores = torch.cat((score_comp, score_other), dim=1)

        return scores
