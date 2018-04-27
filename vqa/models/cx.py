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
# Helper models
################################################################################

# Returns the fusion vector in addition to the answer
class AbstractNoattCX(AbstractNoAtt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_v, input_q):
        x_q = self.seq2vec(input_q)
        y = self._fusion(input_v, x_q)
        a = self._classif(y)
        return y, a


# Same as MutanNoAtt but inherits from AbstractNoattCX
class MutanNoAttCX(AbstractNoattCX):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['fusion']['dim_h'] = opt['fusion']['dim_mm']
        super().__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MutanFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


################################################################################
# Base class for neural CX models
################################################################################

class CXModelBase(nn.Module):

    def __init__(self, vqa_model, knn_size):
        super().__init__()

        self.vqa_model = vqa_model
        self.vqa_model.eval()

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

        # Separate results into original and knns
        a_orig = a[:, 0, :].contiguous()
        y_orig = y[:, 0, :].contiguous()
        a_knns = a[:, 1:, :].contiguous()
        y_knns = y[:, 1:, :].contiguous()

        return a_orig, y_orig, a_knns, y_knns

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
        a_knns = a_knns.detach()

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
        self.dim_y = self.vqa_model.module.opt['fusion']['dim_mm']

        self.linear = nn.Linear(self.knn_size * self.dim_y, self.knn_size)

    def forward(self, image_features, question_wids, answer_aids):
        a_orig, y_orig, a_knns, y_knns = self.vqa_forward(image_features, question_wids)
        y_knns = y_knns.detach()

        scores = self.linear(y_knns.view(-1, self.knn_size * self.dim_y))
        # TODO: dropout

        return scores
