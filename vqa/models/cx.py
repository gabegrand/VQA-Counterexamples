import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from vqa.models import fusion
from .noatt import AbstractNoAtt

from sklearn.metrics.pairwise import cosine_similarity


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
			q_emb.requires_grad = True
		else:
			# TODO: Unclear if this is redundant, but better safe than sorry
			self.vqa_model.eval()
			for p in self.vqa_model.parameters():
				p.requires_grad = False

		# Duplicate each question knn_size + 1 times
		q_emb = q_emb.view(batch_size, 1, -1).expand(batch_size, self.knn_size +
													 1, -1).contiguous().view(batch_size * (self.knn_size + 1), -1)

		# Run the VQA model
		z = self.vqa_model._fusion(v_emb, q_emb)
		a = self.vqa_model._classif(z)

		a = a.view(batch_size, self.knn_size + 1, -1)
		z = z.view(batch_size, self.knn_size + 1, -1)

		if not self.trainable_vqa:
			a = a.detach().data
			z = z.detach().data

		# Separate results into original and knns
		a_orig = Variable(a[:, 0, :].contiguous(), requires_grad=True)
		z_orig = Variable(z[:, 0, :].contiguous(), requires_grad=True)
		a_knns = Variable(a[:, 1:, :].contiguous(), requires_grad=True)
		z_knns = Variable(z[:, 1:, :].contiguous(), requires_grad=True)

		return a_orig, z_orig, a_knns, z_knns

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
		_, _, a_knns, _ = self.vqa_forward(image_features, question_wids)

		# VQA model's score for the original answer for each KNN
		scores_list = []

		for i, a_idx in enumerate(answer_aids):
			a_distribution = F.softmax(a_knns[i, :], dim=-1)
			scores_list.append(a_distribution[:, a_idx])

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
		a_orig, z_orig, a_knns, z_knns = self.vqa_forward(
			image_features, question_wids)

		assert(z_knns.requires_grad)

		scores = self.linear(z_knns.view(-1, self.knn_size * self.dim_z))
		# TODO: dropout

		return scores


class SemanticBaseline(CXModelBase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dim_z = self.vqa_model.opt['fusion']['dim_mm']
		print(self.vqa_model.opt)
		self.emb = np.zeros((2000, 2400))
		self.emb_pairs = np.zeros((2000,2000))

		self.lam = 0.5 # default

	def set_lambda(self, lam):
		self.lam = lam

	def set_answer_embedding(self, emb):
		self.emb = emb
		self.emb_pairs = cosine_similarity(self.emb)

	def softmax(self, w):
		e = np.exp(np.array(w))
		dist = e / np.sum(e)
		return dist

	def forward(self, image_features, question_wids, answer_aids):
		a_orig, z_orig, a_knns, z_knns = self.vqa_forward(
			image_features, question_wids)
		a_orig = a_orig.cpu().data.numpy()
		a_knns = a_knns.cpu().data.numpy()
		answer_aids = answer_aids.cpu().numpy()

		ret_scores = []
		for ex, knn, aid in zip(a_orig, a_knns, answer_aids):
			scores = []
			for nn_i, nb in enumerate(knn):
				nb = self.softmax(nb)
				weighted_sim = self.emb_pairs[aid,:].dot(nb)

				# THIS IS IMPORTANT
				weighted_sim -= nb[aid] * self.emb_pairs[aid,aid]

				logp = nb[aid]
				p = nb[aid] + 1e-8
				logp = np.log(p)
				# score = -1 * (self.lam * weighted_sim + (1-self.lam) * logp)
				# THIS IS ALSO IMPORTANT
				score = (self.lam * weighted_sim) - ((1-self.lam) * logp)
				scores.append(score)
			softmax_scores = self.softmax(scores)
			ret_scores.append(softmax_scores)
		ret_scores = np.array(ret_scores)
		ret_scores = Variable(torch.cuda.FloatTensor(ret_scores), requires_grad=True)
		return ret_scores

		# sizes torch.Size([64, 2000]) torch.Size([64, 24, 2000])

		# scores = self.linear(z_knns.view(-1, self.knn_size * self.dim_z))
		# TODO: dropout

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

		# TODO: this is redundant, can get from VQA forward
		q_emb = self.vqa_model.seq2vec(Variable(question_wids)).detach().data

		_, z_orig, _, z_knns = self.vqa_forward(image_features, question_wids)
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

		self.linear = nn.Linear(
			(2 * self.dim_v) + self.dim_q + self.dim_z, self.dim_h)
		self.out = nn.Linear(self.dim_h, 1)
		self.final = nn.Linear(self.knn_size, self.knn_size)
		self.relu = nn.ReLU()

	def forward(self, image_features, question_wids, answer_aids):
		n_preds = image_features.size(1) - 1
		assert(n_preds == self.knn_size)

		v_orig = image_features[:, 0]

		# TODO: this is redundant, can get from VQA forward
		q_emb = self.vqa_model.seq2vec(Variable(question_wids)).detach().data

		_, z_orig, _, z_knns = self.vqa_forward(image_features, question_wids)
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

		scores = self.final(scores)

		return scores
