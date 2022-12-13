import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y, eps=1e-12):
	"""
	Args:
	  x: pytorch Tensor, with shape [m, d]
	  y: pytorch Tensor, with shape [n, d]
	Returns:
	  dist: pytorch Tensor, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(x, y.t(), beta=1, alpha=-2) #dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=eps).sqrt()

	return dist

def hard_example_mining(dist_mat, target):
	"""For each anchor, find the hardest positive and negative sample.
	Args:
	  dist_mat: pytorch Tensor, pair wise distance between samples, shape [N, N]
	  target: pytorch LongTensor, with shape [N]
	  return_inds: whether to return the indices. Save time if `False`(?)
	Returns:
	  dist_ap: pytorch Tensor, distance(anchor, positive); shape [N]
	  dist_an: pytorch Tensor, distance(anchor, negative); shape [N]
	  p_inds: pytorch LongTensor, with shape [N];
	    indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
	  n_inds: pytorch LongTensor, with shape [N];
	    indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
	NOTE: Only consider the case in which all target have same num of samples,
	  thus we can cope with all anchors in parallel.
	"""
	assert len(dist_mat.size()) == 2
	assert dist_mat.size(0) == dist_mat.size(1)
	N = dist_mat.size(0)

	# shape [N, N]
	is_pos = target.expand(N, N).eq(target.expand(N, N).t())
	is_neg = target.expand(N, N).ne(target.expand(N, N).t())

	dist_ap, relative_p_inds = torch.max(
		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

	dist_an, relative_n_inds = torch.min(
		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

	dist_ap = dist_ap.squeeze(1)
	dist_an = dist_an.squeeze(1)

	return dist_ap, dist_an


class TripletLoss(nn.Module):
	"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
	Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
	Loss for Person Re-Identification'."""
	def __init__(self, margin, feat_norm='yes'):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.feat_norm = feat_norm
		if margin >= 0:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def forward(self, global_feat1, global_feat2, target):
		if self.feat_norm == 'yes':
			global_feat1 = F.normalize(global_feat1, p=2, dim=-1)
			global_feat2 = F.normalize(global_feat2, p=2, dim=-1)

		dist_mat = euclidean_dist(global_feat1, global_feat2)
		dist_ap, dist_an = hard_example_mining(dist_mat, target)

		y = dist_an.new().resize_as_(dist_an).fill_(1)
		if self.margin >= 0:
			loss = self.ranking_loss(dist_an, dist_ap, y)
		else:
			loss = self.ranking_loss(dist_an - dist_ap, y)

		return loss



