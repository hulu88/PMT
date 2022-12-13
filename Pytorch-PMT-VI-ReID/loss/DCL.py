import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self,inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        N = inputs.size(0)
        id_num = N // 2 // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  #  c-i

        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask
        an = an * mask_an

        list_an = []
        for i in range (id_num):
            list_an.append(torch.mean(an[i][an[i]>1e-6]))
        an_mean = sum(list_an) / len(list_an)

        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap>1e-6])

        loss = ap_mean / an_mean

        return loss