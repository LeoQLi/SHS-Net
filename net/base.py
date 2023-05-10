import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def get_sign(pred, min_val=-1.0):
    distance_pos = pred >= 0.0         # logits to bool
    distance_sign = torch.full_like(distance_pos, min_val, dtype=torch.float32)
    distance_sign[distance_pos] = 1.0  # bool to sign factor
    return distance_sign


def knn_group_v2(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, F, N)
    :param  idx:    (B, M, k)
    :return (B, F, M, k)
    """
    B, F, N = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(2).expand(B, F, M, N)
    idx = idx.unsqueeze(1).expand(B, F, M, k)

    return torch.gather(x, dim=3, index=idx)


def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    _, knn_idx, _ = knn_points(pos, query, K=k+offset, return_nn=False)
    return knn_idx[:, :, offset:]


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x


class PileConv(nn.Module):
    def __init__(self, input_dim, output_dim, with_mid=True):
        super(PileConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_mid = with_mid

        self.conv_in = Conv1D(input_dim, input_dim*2, with_bn=True, with_relu=with_mid)

        if with_mid:
            self.conv_mid = Conv1D(input_dim*2, input_dim//2, with_bn=True)
            self.conv_out = Conv1D(input_dim + input_dim//2, output_dim, with_bn=True)
        else:
            self.conv_out = Conv1D(input_dim + input_dim*2, output_dim, with_bn=True)

    def forward(self, x, num_out, dist_w=None):
        """
            x: (B, C, N)
        """
        BS, _, N = x.shape

        if dist_w is None:
            y = self.conv_in(x)
        else:
            y = self.conv_in(x * dist_w[:,:,:N])        # (B, C*2, N)

        feat_g = torch.max(y, dim=2, keepdim=True)[0]   # (B, C*2, 1)
        if self.with_mid:
            feat_g = self.conv_mid(feat_g)              # (B, C/2, 1)

        x = torch.cat([x[:, :, :num_out],
                    feat_g.view(BS, -1, 1).repeat(1, 1, num_out),
                    ], dim=1)

        x = self.conv_out(x)
        return x


