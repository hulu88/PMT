import torch
from torch import nn
import torch.nn.functional as F


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

class MSEL(nn.Module):
    def __init__(self,num_pos,feat_norm = 'no'):
        super(MSEL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        target, _ = targets.chunk(2,0)
        N = target.size(0)

        dist_mat = pdist_torch(inputs, inputs)

        dist_intra_rgb = dist_mat[0 : N, 0 : N]
        dist_cross_rgb = dist_mat[0 : N, N : 2*N]
        dist_intra_ir = dist_mat[N : 2*N, N : 2*N]
        dist_cross_ir = dist_mat[N : 2*N, 0 : N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1 ,largest = True, sorted = False) # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim =1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss


class MSEL_Cos(nn.Module):          # for features after bn
    def __init__(self,num_pos):
        super(MSEL_Cos, self).__init__()
        self.num_pos = num_pos

    def forward(self, inputs, targets):

        inputs = nn.functional.normalize(inputs, p=2, dim=1)

        target, _ = targets.chunk(2,0)
        N = target.size(0)

        dist_mat = 1 - torch.matmul(inputs, torch.t(inputs))

        dist_intra_rgb = dist_mat[0: N, 0: N]
        dist_cross_rgb = dist_mat[0: N, N: 2*N]
        dist_intra_ir = dist_mat[N: 2*N, N: 2*N]
        dist_cross_ir = dist_mat[N: 2*N, 0: N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)  # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
               torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss


class MSEL_Feat(nn.Module):    # compute MSEL loss by the distance between sample and center
    def __init__(self, num_pos):
        super(MSEL_Feat, self).__init__()
        self.num_pos = num_pos

    def forward(self, input1, input2):
        N = input1.size(0)
        id_num = N // self.num_pos

        feats_rgb = input1.chunk(id_num, 0)
        feats_ir = input2.chunk(id_num, 0)

        loss_list = []
        for i in range(id_num):
            cross_center_rgb = torch.mean(feats_rgb[i], dim=0)  # cross center
            cross_center_ir = torch.mean(feats_ir[i], dim=0)

            for j in range(self.num_pos):

                feat_rgb = feats_rgb[i][j]
                feat_ir = feats_ir[i][j]

                intra_feats_rgb = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j+1:]), dim=0)  # intra center
                intra_feats_ir = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j+1:]), dim=0)

                intra_center_rgb = torch.mean(intra_feats_rgb, dim=0)
                intra_center_ir = torch.mean(intra_feats_ir, dim=0)

                dist_intra_rgb = pdist_torch(feat_rgb.view(1, -1), intra_center_rgb.view(1, -1))
                dist_intra_ir = pdist_torch(feat_ir.view(1, -1), intra_center_ir.view(1, -1))

                dist_cross_rgb = pdist_torch(feat_rgb.view(1, -1), cross_center_ir.view(1, -1))
                dist_cross_ir = pdist_torch(feat_ir.view(1, -1), cross_center_rgb.view(1, -1))

                loss_list.append(torch.pow(dist_cross_rgb - dist_intra_rgb, 2) + torch.pow(dist_cross_ir - dist_intra_ir, 2))

        loss = sum(loss_list) / N / 2

        return loss
