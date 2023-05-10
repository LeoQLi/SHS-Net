import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .encode import EncodeNet
from .base import PileConv, Conv1D, get_sign


class FusionNet(nn.Module):
    def __init__(self,
                 d_aug=64,
                 d_code=0,
                 d_mid=128,
                 d_out=64,
                 n_mid=1,
                 with_grad=False,
            ):
        super(FusionNet, self).__init__()
        assert n_mid > 3
        dims = [d_aug + d_code] + [d_mid for _ in range(n_mid)] + [d_out]
        self.skip_in = [n_mid // 2 + 1]    # TODO

        self.with_grad = with_grad
        if with_grad:
            dims += [3]

        self.mlp_1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, d_aug, 1),
        )

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                assert dims[l + 1] > d_aug + d_code
                out_dim = dims[l + 1] - d_aug - d_code  # TODO
            else:
                out_dim = dims[l + 1]
            lin = nn.Conv1d(dims[l], out_dim, 1)

            setattr(self, "lin" + str(l), lin)

    def forward(self, pos, code=None):
        """
            pos: (B, C, N)
            code: (B, C, N)
        """
        pos = self.mlp_1(pos)
        x = torch.cat([pos, code], dim=1)

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, pos, code], dim=1)

            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                xx = x if self.with_grad else None
                x = F.relu(x)

        return x, xx


class PointEncoder(nn.Module):
    def __init__(self, knn, use_w=True):
        super(PointEncoder, self).__init__()
        self.use_w = use_w
        d_out = 64
        self.encodeNet = EncodeNet(num_convs=2,
                                    in_channels=3,
                                    conv_channels=24,
                                    knn=knn,
                                )
        d_code = self.encodeNet.out_channels   # 60
        self.fusion_net = FusionNet(d_code=d_code, d_mid=128, d_out=d_out, n_mid=4)

        dim_1 = 128
        self.conv_1 = Conv1D(d_code + d_out, dim_1)

        self.pconv1 = PileConv(dim_1, dim_1*2)
        self.pconv2 = PileConv(dim_1*2, dim_1*2)
        self.pconv3 = PileConv(dim_1*2, dim_1*2)
        self.pconv4 = PileConv(dim_1*2, dim_1*2)
        self.pconv5 = PileConv(dim_1*2, dim_1*2)

        self.conv_2 = Conv1D(dim_1*2, 256)
        self.conv_3 = Conv1D(256, 128)

        if self.use_w:
            self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            self.beta = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            torch.nn.init.ones_(self.alpha.data)
            torch.nn.init.ones_(self.beta.data)

    def forward(self, pos, num_pcl, knn_idx):
        """
            pos:  (B, 3, N)
            knn_idx: (B, N, K)
        """
        data_tuple = None
        y = self.encodeNet(x=pos, pos=pos, knn_idx=knn_idx)                # (B, C, N)
        yy, _ = self.fusion_net(pos=pos, code=y)                           # (B, C, N)
        y = torch.cat([y, yy], dim=1)                                      # (B, C, N)
        y = self.conv_1(y)

        if self.use_w:
            ### compute distance weights from points to its center point (ref: FKAConv, POCO)
            dist = torch.sqrt((pos.detach() ** 2).sum(dim=1))              # (B, N)
            dist_w = torch.sigmoid(-self.alpha * dist + self.beta)
            dist_w_s = dist_w.sum(dim=1, keepdim=True)                     # (B, 1)
            dist_w_s = dist_w_s + (dist_w_s == 0) + 1e-6
            dist_w = (dist_w / dist_w_s * dist.shape[1]).unsqueeze(1)      # (B, 1, N)

            data_tuple = (dist_w[:, :, :num_pcl//2], y[:, :, :num_pcl//2])
        else:
            dist_w = None
            data_tuple = (dist_w, y[:, :, :num_pcl//2])

        ### decrease the number of points and pile the features
        y1 = self.pconv1(y, num_pcl*2, dist_w=dist_w)                      # (B, C, n*2)
        y2 = self.pconv2(y1, num_pcl, dist_w=dist_w)                       # (B, C, n)
        y2 = y2 + y1[:, :, :num_pcl]

        y3 = self.pconv3(y2, num_pcl, dist_w=dist_w)
        y4 = self.pconv4(y3, num_pcl//2, dist_w=dist_w)
        y4 = y4 + y3[:, :, :num_pcl//2]

        y5 = self.pconv5(y4, num_pcl//2, dist_w=dist_w)

        y = self.conv_3(self.conv_2(y5))                                  # (B, C, n)

        return y, data_tuple


class Network(nn.Module):
    def __init__(self, num_pat=1, num_sam=1, encode_knn=16):
        super(Network, self).__init__()
        self.num_pcl = num_pat // 4
        self.num_pcl_g = num_sam // 4
        self.encode_knn = encode_knn

        self.encode_knn_g = 1
        self.pointEncoder = PointEncoder(knn=self.encode_knn, use_w=True)
        self.pointEncoder_g = PointEncoder(knn=self.encode_knn_g, use_w=True)

        self.conv_q = nn.Conv1d(128, 64, 1)
        self.conv_v = nn.Conv1d(128, 64, 1)
        self.mlp_n1  = nn.Linear(64, 64)

        self.conv_p = Conv1D(128*2, 128)
        self.pconv_1 = PileConv(128, 128)
        self.pconv_2 = PileConv(128, 128)
        self.pconv_3 = PileConv(128, 128)

        self.conv_1 = Conv1D(128, 128)
        self.conv_n = Conv1D(128, 128)
        self.conv_w = nn.Conv1d(128, 1, 1)
        self.mlp_n  = nn.Linear(64, 4)
        self.mlp_nn = nn.Linear(128, 3)


    def forward(self, pcl_pat, pcl_sample, mode_test=False):
        """
            pcl_pat: (B, N, 3)
            pcl_sample: (B, N', 3), N' < N
        """
        normal, weights, neighbor_normal = None, None, None
        self.normal_s = None

        _, knn_idx, _ = knn_points(pcl_pat, pcl_pat, K=self.encode_knn+1, return_nn=False)  # (B, N, K+1)

        ### Encoder
        pcl_pat = pcl_pat.transpose(2, 1)                     # (B, 3, N)
        y, data_tuple = self.pointEncoder(pcl_pat,
                                            num_pcl=self.num_pcl,
                                            knn_idx=knn_idx[:,:,1:self.encode_knn+1],
                                        )                     # (B, C, n)
        wd, _ = data_tuple                                    # (B, C, n)

        pcl_sample = pcl_sample.transpose(2, 1)               # (B, 3, N')
        y_g, _ = self.pointEncoder_g(pcl_sample,
                                            num_pcl=self.num_pcl_g,
                                            knn_idx=None,
                                        )                     # (B, C, n')
        y_g = y_g.max(dim=2, keepdim=True)[0].repeat(1, 1, self.num_pcl//2)

        xc = torch.cat([y, y_g], dim=1)

        y0 = self.conv_p(xc)
        y1 = self.pconv_1(y0, self.num_pcl//2, dist_w=wd)
        y2 = self.pconv_2(y1, self.num_pcl//4, dist_w=wd)
        y2 = y2 + y1[:, :, :self.num_pcl//4] + y0[:, :, :self.num_pcl//4]
        feat = self.pconv_3(y2, self.num_pcl//4, dist_w=wd)

        feat = self.conv_1(feat)

        ### Output
        weights = 0.01 + torch.sigmoid(self.conv_w(feat))               # (B, 1, n)
        feat_w = self.conv_n(feat * weights)
        # normal = self.mlp_n(feat_w.max(dim=2, keepdim=False)[0])      # (B, 3)

        query = self.conv_q(feat_w)                                     # (B, C, n)
        value = self.conv_v(feat_w)                                     # (B, C, n)
        attn = torch.softmax(query, dim=2).max(dim=1, keepdim=True)[0]  # (B, 1, n)
        feat_w = torch.matmul(value, attn.transpose(1,2)).squeeze()     # (B, C, n) * (B, n, 1) = (B, C, 1) -> (B, C)
        feat_w = self.mlp_n1(feat_w)
        normal = self.mlp_n(feat_w)

        self.normal_s = normal[:, 3]
        normal = F.normalize(normal[:, :3], p=2, dim=-1)

        neighbor_normal = self.mlp_nn(feat.transpose(2, 1))             # (B, n, 3)
        neighbor_normal = F.normalize(neighbor_normal, p=2, dim=-1)

        if mode_test:
            s = get_sign(self.normal_s, min_val=-1.0)
            normal = normal * s[:, None]
            return normal

        return normal, weights, neighbor_normal


    def get_loss(self, q_target, q_pred, ne_target=None, ne_pred=None, pred_weights=None, pcl_in=None,
                normal_loss_type='mse_loss'):
        """
            q_target: query point normal, (B, 3)
            q_pred: query point normal, (B, 3)
            ne_target: patch point normal, (B, N, 3)
            ne_pred: patch point normal, (B, N, 3)
            pred_weights: patch point weight, (B, 1, N)
            pcl_in: input (noisy) point clouds, (B, N, 3)
        """
        def cos_angle(v1, v2):
            return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

        _device, _dtype = q_target.device, q_target.dtype
        s_loss = torch.zeros(1, device=_device, dtype=_dtype)
        weight_loss = torch.zeros(1, device=_device, dtype=_dtype)
        normal_loss = torch.zeros(1, device=_device, dtype=_dtype)
        consistency_loss = torch.zeros(1, device=_device, dtype=_dtype)

        ### center point normal
        if q_pred is not None:
            # if normal_loss_type == 'mse_loss':
            #     normal_loss = 0.25 * F.mse_loss(q_pred, q_target)
            # elif normal_loss_type == 'ms_euclidean':
            #     normal_loss = 0.1 * torch.min((q_pred-q_target).pow(2).sum(1), (q_pred+q_target).pow(2).sum(1)).mean()
            # elif normal_loss_type == 'ms_oneminuscos':
            #     cos_ang = cos_angle(q_pred, q_target)
            #     normal_loss = 1.0 * (1-torch.abs(cos_ang)).pow(2).mean()
            # elif normal_loss_type == 'sin':
            normal_loss = 0.1 * torch.linalg.norm(torch.cross(q_pred, q_target, dim=-1), ord=2, dim=1).mean()
            # else:
            #     raise ValueError('Unsupported loss type: %s' % (normal_loss_type))

        ### neighboring point normal
        if ne_pred is not None:
            num_out = ne_pred.shape[1]
            weights = pred_weights.squeeze()
            ne_target = ne_target[:, :num_out, :].contiguous()
            batch_size, n_points, _ = ne_target.shape

            if normal_loss_type == 'mse_loss':
                consistency_loss = 0.5 * torch.mean(weights * (ne_pred - ne_target).pow(2).sum(2))
            elif normal_loss_type == 'ms_euclidean':
                consistency_loss = torch.mean(weights * torch.min((ne_pred - ne_target).pow(2).sum(2),
                                                                  (ne_pred + ne_target).pow(2).sum(2)) )
            elif normal_loss_type == 'ms_oneminuscos':
                cos_ang = cos_angle(ne_pred.view(-1, 3), ne_target.view(-1, 3)).view(batch_size, n_points)
                consistency_loss = torch.mean(weights * (1 - torch.abs(cos_ang)).pow(2))
            elif normal_loss_type == 'sin':
                consistency_loss = 0.25 * torch.mean(weights *
                                    torch.linalg.norm(torch.cross(ne_pred.view(-1, 3), ne_target.view(-1, 3), dim=-1).view(batch_size, n_points, 3),
                                                ord=2, dim=2))
            else:
                raise ValueError('Unsupported loss type: %s' % (normal_loss_type))

        ### compute true_weight by fitting distance
        if pred_weights is not None:
            pred_weights = pred_weights.squeeze()
            num_out = pred_weights.shape[1]
            pcl_local = pcl_in[:, :num_out, :] - pcl_in[:, 0:1, :]                                      # (B, N, 3)
            scale = (pcl_local ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0]      # (B, 1, 1)
            pcl_local = pcl_local / scale

            ### the distances between the neighbor points and ground truth tangent plane
            gamma = 0.3
            thres_d = 0.05 ** 2
            normal_dis = torch.bmm(q_target.unsqueeze(1), pcl_local.transpose(2, 1)).pow(2).squeeze()   # (B, N), dis^2
            sigma = torch.mean(normal_dis, dim=1) * gamma + 1e-5                                        # (B,)
            threshold_matrix = torch.ones_like(sigma) * thres_d                                         # (B,)
            sigma = torch.where(sigma < thres_d, threshold_matrix, sigma)                               # (B,), all sigma >= thres_d
            true_weights_plane = torch.exp(-1 * torch.div(normal_dis, sigma.unsqueeze(-1)))             # (B, N), -dis/mean(dis') -> (-∞, 0) -> (0, 1)

            true_weights = true_weights_plane
            weight_loss = 1.0 * (true_weights - pred_weights).pow(2).mean()

        if self.normal_s is not None:
            cos_ang_q = cos_angle(q_pred, q_target)            # (B,)
            sign_q = get_sign(cos_ang_q, min_val=0.0)          # (B,)
            s_loss = 0.1 * F.binary_cross_entropy_with_logits(input=self.normal_s.squeeze(), target=sign_q, reduction='none').mean()

        loss = normal_loss + consistency_loss + weight_loss + s_loss
        return loss, (normal_loss, consistency_loss, weight_loss, s_loss)


