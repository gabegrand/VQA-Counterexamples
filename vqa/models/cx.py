import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from vqa.models import fusion
from .noatt import AbstractNoAtt


##########################################################################
# Baseline models (no VQA)
##########################################################################

class RandomBaseline(nn.Module):

	def __init__(self, knn_size):
		super().__init__()

		self.knn_size = knn_size

	def forward(self, image_features, question_wids, answer_aids):
		batch_size = image_features.size(0)
		scores = Variable(torch.rand((batch_size, self.knn_size))).cuda()
		return scores


class DistanceBaseline(nn.Module):

	def __init__(self, knn_size):
		super().__init__()

		self.knn_size = knn_size

	def forward(self, image_features, question_wids, answer_aids):
		batch_size = image_features.size(0)
		scores = Variable(Tensor(list(reversed(range(self.knn_size)))).view(
			1, -1).expand((batch_size, self.knn_size))).cuda()
		return scores


##########################################################################
# Base class for neural CX models
##########################################################################

class CXModelBase(nn.Module):

	def __init__(self, vqa_model, knn_size, trainable_vqa=False):
		super().__init__()

		self.vqa_model = vqa_model
		self.trainable_vqa = trainable_vqa

		if not self.trainable_vqa:
			self.vqa_model.eval()

		self.knn_size = knn_size

	def vqa_forward(self, image_features, question_wids):
		assert(image_features.size(1) == self.knn_size + 1)
		batch_size = image_features.size(0)

		# Process all image features as a single batch
		v_emb = Variable(image_features.view(
			batch_size * (self.knn_size + 1), -1))
		q_emb = self.vqa_model.seq2vec(Variable(question_wids))

		if self.trainable_vqa:
			v_emb.requires_grad = True
		else:
			q_emb = q_emb.detach()
			# TODO: Unclear if this is redundant, but better safe than sorry
			self.vqa_model.eval()
			for p in self.vqa_model.parameters():
				p.requires_grad = False

		# Duplicate each question knn_size + 1 times
		q_emb_dup = q_emb.view(batch_size, 1, -1).expand(batch_size,
			self.knn_size + 1, -1).contiguous().view(batch_size * (self.knn_size + 1), -1)

		# Run the VQA model
		z = self.vqa_model._fusion(v_emb, q_emb_dup)
		a = self.vqa_model._classif(z)

		a = a.view(batch_size, self.knn_size + 1, -1)
		z = z.view(batch_size, self.knn_size + 1, -1)

		a_orig = a[:, 0, :].contiguous()
		z_orig = z[:, 0, :].contiguous()
		a_knns = a[:, 1:, :].contiguous()
		z_knns = z[:, 1:, :].contiguous()

		if not self.trainable_vqa:
			a_orig = Variable(a_orig.detach().data, requires_grad=True)
			z_orig = Variable(z_orig.detach().data, requires_grad=True)
			a_knns = Variable(a_knns.detach().data, requires_grad=True)
			z_knns = Variable(z_knns.detach().data, requires_grad=True)

		return a_orig, z_orig, a_knns, z_knns, q_emb

	def forward(self, image_features, question_wids, answer_aids):
		raise NotImplementedError


##########################################################################
# Neural CX models
##########################################################################

class BlackBox(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.vqa_model.eval()

	def forward(self, image_features, question_wids, answer_aids):
		_, _, a_knns, _, _ = self.vqa_forward(image_features, question_wids)

		# VQA model's score for the original answer for each KNN
		scores_list = []
		for i, a_idx in enumerate(answer_aids):
			scores_list.append(a_knns[i, :, a_idx])
		scores = torch.stack(scores_list, dim=0)

		# Flip the sign, since the highest scoring items are the worst
		# counterexamples
		scores = -scores

		return scores


class LinearContext(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']

		self.linear = nn.Linear(self.knn_size * self.dim_z, self.knn_size)

	def forward(self, image_features, question_wids, answer_aids):
		a_orig, z_orig, a_knns, z_knns, _ = self.vqa_forward(
			image_features, question_wids)

		assert(z_knns.requires_grad)

		scores = self.linear(z_knns.view(-1, self.knn_size * self.dim_z))
		# TODO: dropout

		return scores


class SemanticBaseline(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']
		# self.lambda = 0.5


	def forward(self, image_features, question_wids, answer_aids):
		a_orig, z_orig, a_knns, z_knns, _ = self.vqa_forward(
			image_features, question_wids)

		assert(z_knns.requires_grad)

		print('sizes', a_orig.size(), a_knns.size())
		# scores = self.linear(z_knns.view(-1, self.knn_size * self.dim_z))
		# TODO: dropout

		return scores


class PairwiseModel(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.dim_v = self.vqa_model.opt['fusion']['dim_v']
		self.dim_q = self.vqa_model.opt['fusion']['dim_q']
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']

		self.dim_h = 300

		self.linear = nn.Linear(
			(2 * self.dim_v) + self.dim_q + self.dim_z, self.dim_h)
		self.out = nn.Linear(self.dim_h, 1)
		self.relu = nn.ReLU()

	def forward(self, image_features, question_wids, answer_aids):
		n_preds = image_features.size(1) - 1
		assert(n_preds == self.knn_size)

		v_orig = image_features[:, 0]

		_, z_orig, _, z_knns, q_emb = self.vqa_forward(image_features, question_wids)
		z_knns = z_knns.detach().data

		scores = []

		for i in range(n_preds):
			v_other = image_features[:, i + 1]
			z_other = z_knns[:, i]

			input = Variable(
				torch.cat((v_orig, v_other, q_emb, z_other), dim=1), requires_grad=True)
			h = self.relu(self.linear(input))
			score = self.relu(self.out(h))

			scores.append(score)

		scores = torch.cat(scores, dim=1)

		return scores


class PairwiseLinearModel(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		assert(self.knn_size == 24)

		self.dim_v = self.vqa_model.opt['fusion']['dim_v']
		self.dim_q = self.vqa_model.opt['fusion']['dim_q']
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']

		self.dim_h = 300
		self.dim_a = 300

		self.answer_embedding = nn.Embedding(len(self.vqa_model.vocab_answers),
			self.dim_a)

		self.linear = nn.Linear((2 * self.dim_v) + self.dim_q + (2 * self.dim_z)
			+ self.dim_a, self.dim_h)
		self.out = nn.Linear(self.dim_h, 1)
		self.relu = nn.ReLU()

	def forward(self, image_features, question_wids, answer_aids):
		n_preds = image_features.size(1) - 1
		assert(n_preds == self.knn_size)

		v_orig = Variable(image_features[:, 0])

		_, z_orig, _, z_knns, q_emb = self.vqa_forward(image_features, question_wids)

		a_emb = self.answer_embedding(Variable(answer_aids))

		scores = []

		for i in range(n_preds):
			v_other = Variable(image_features[:, i + 1])
			z_other = z_knns[:, i]

			input = torch.cat((v_orig, v_other, q_emb, z_orig, z_other, a_emb), dim=1)
			h = self.relu(self.linear(input))
			score = self.relu(self.out(h))

			scores.append(score)

		scores = torch.cat(scores, dim=1)

		return scores


class ContrastiveModel(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.dim_v = self.vqa_model.opt['fusion']['dim_v']
		self.dim_q = self.vqa_model.opt['fusion']['dim_q']
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']

		self.dim_h = 300

		self.dim_a = 300
		self.answer_embedding = nn.Embedding(len(self.vqa_model.vocab_answers),
			self.dim_a)

		# self.linear = nn.Linear(self.dim_v + self.dim_q + self.dim_z
		# 	+ self.dim_a, self.dim_h)
		self.linear = nn.Linear(self.dim_v + self.dim_z, self.dim_h)
		self.relu = nn.ReLU()

	def forward(self, image_features, question_wids, answer_aids):
		batch_size = image_features.size(0)
		n_preds = image_features.size(1)
		assert(n_preds == self.knn_size + 1)

		v_all = Variable(image_features)

		_, z_orig, _, z_knns, q_emb = self.vqa_forward(image_features, question_wids)
		z_all = torch.cat([z_orig.view(batch_size, 1, self.dim_z), z_knns], dim=1)

		# a_emb = self.answer_embedding(Variable(answer_aids))

		out = Variable(torch.zeros([batch_size, n_preds, self.dim_h])).cuda()
		for i in range(n_preds):
			v = v_all[:, i]
			# q = q_emb
			# a = a_emb
			z = z_all[:, i]
			out[:, i] = self.get_hidden(v, z)

		return out

	def get_hidden(self, v, z):
		input = torch.cat([v, z], dim=1)
		return self.relu(self.linear(input))

	# def get_hidden(self, v, q, a, z):
	# 	input = torch.cat([v, q, a, z], dim=1)
	# 	return self.relu(self.linear(input))

	def get_scores(self, h_orig, h_knns):
		batch_size = h_orig.size(0)
		n_preds = h_knns.size(1)

		scores = Variable(torch.zeros([batch_size, n_preds]))
		for i in range(n_preds):
			euclidean_distance = F.pairwise_distance(h_orig, h_knns[:, i])
			scores[:, i] = euclidean_distance

		return scores


class SimilarityModel(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']

	def forward(self, image_features, question_wids, answer_aids):
		batch_size = image_features.size(0)

		a_orig, z_orig, a_knns, z_knns, _ = self.vqa_forward(
			image_features, question_wids)

		v_orig = image_features[:, 0]
		v_cossim = torch.zeros([batch_size, self.knn_size])

		z_orig = z_orig.data
		z_knns = z_knns.data
		z_cossim = torch.zeros([batch_size, self.knn_size])

		a_xent = torch.zeros([batch_size, self.knn_size])

		for i in range(self.knn_size):
			v_cossim[:, i] = F.cosine_similarity(v_orig, image_features[:, i+1])
			z_cossim[:, i] = F.cosine_similarity(z_orig, z_knns[:, i])
			a_xent[:, i] = F.cross_entropy(a_knns[:, i], Variable(answer_aids), reduce=False).data

		scores = v_cossim + z_cossim + a_xent

		return Variable(scores).cuda()
