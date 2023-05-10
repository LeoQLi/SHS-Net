import torch
import torch.nn as nn

from .base import knn_group_v2, get_knn_idx, Conv1D


class Aggregator(nn.Module):
    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


class GraphConv(nn.Module):
    def __init__(self, knn, in_channels, growth_rate, num_fc_layers, aggr='max', with_bn=True, with_relu=True, relative_feat_only=False):
        super().__init__()
        self.knn = knn
        self.in_channels = in_channels
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers
        self.growth_rate = growth_rate
        self.relative_feat_only = relative_feat_only

        if knn > 1:
            if relative_feat_only:
                self.layer_first = Conv1D(in_channels+3, growth_rate, with_bn=with_bn, with_relu=with_relu)
            else:
                self.layer_first = Conv1D(in_channels*3, growth_rate, with_bn=with_bn, with_relu=with_relu)
        else:
            self.layer_first = Conv1D(in_channels, growth_rate, with_bn=with_bn, with_relu=with_relu)

        self.layers_mid = nn.ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers_mid.append(Conv1D(in_channels + i * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu))

        self.layer_last = Conv1D(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu)

        dim = in_channels + num_fc_layers * growth_rate
        self.layer_out = Conv1D(dim, dim, with_bn=False, with_relu=False)

        if knn > 1:
            self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        knn_idx: (B, N, K)
        return: (B, C, N, K)
        """
        knn_feat = knn_group_v2(x, knn_idx)                  # (B, C, N, K)
        x_tiled = x.unsqueeze(-1).expand_as(knn_feat)
        if self.relative_feat_only:
            knn_pos = knn_group_v2(pos, knn_idx)
            pos_tiled = pos.unsqueeze(-1)
            edge_feat = torch.cat([knn_pos - pos_tiled, knn_feat - x_tiled], dim=1)
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=1)
        return edge_feat

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return y: (B, C, N)
          knn_idx: (B, N, K)
        """
        B, C, N = x.shape
        K = self.knn

        if knn_idx is None and self.knn > 1:
            pos_t = pos.transpose(2, 1)                                   # (B, N, 3)
            knn_idx = get_knn_idx(pos_t, pos_t, k=self.knn, offset=1)     # (B, N, K)

        if K > 1:
            edge_feat = self.get_edge_feature(x, pos, knn_idx=knn_idx)    # (B, C, N, K)
            edge_feat = edge_feat.view(B, -1, N * K)                      # (B, C, N*K)
            x = x.unsqueeze(-1).repeat(1, 1, 1, K).view(B, C, N * K)      # (B, C, N*K)
        else:
            edge_feat = x

        # First Layer
        y = torch.cat([
            self.layer_first(edge_feat),     # (B, c, N*K)
            x,                               # (B, d, N*K)
        ], dim=1)                            # (B, c+d, N*K)

        # Intermediate Layers
        for layer in self.layers_mid:
            y = torch.cat([
                layer(y),                    # (B, c, N*K)
                y                            # (B, c+d, N*K)
            ], dim=1)                        # (B, d+(L-1)*c, N*K)

        # Last Layer
        y = torch.cat([
            self.layer_last(y),              # (B, c, N*K)
            y                                # (B, d+(L-1)*c, N*K)
        ], dim=1)                            # (B, d+L*c, N*K)

        # y = torch.cat([y, x], dim=1)
        y = self.layer_out(y)                # (B, C, N*K)

        # Pooling Layer
        if K > 1:
            y = y.reshape(B, -1, N, K)
            y = self.aggr(y, dim=-1)             # (B, C, N)

        return y, knn_idx


class EncodeNet(nn.Module):
    def __init__(self,
        knn,
        num_convs=1,
        in_channels=3,
        conv_channels=24,
        growth_rate=12,
        num_fc_layers=3,
    ):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels

        self.trans = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            tran = Conv1D(in_channels, conv_channels, with_bn=True, with_relu=True)
            conv = GraphConv(
                knn=knn,
                in_channels=conv_channels,
                growth_rate=growth_rate,
                num_fc_layers=num_fc_layers,
                relative_feat_only=(i == 0),
            )
            self.trans.append(tran)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return: (B, C, N), C = conv_channels+num_fc_layers*growth_rate
        """
        for i in range(self.num_convs):
            x = self.trans[i](x)
            x, knn_idx = self.convs[i](x, pos=pos, knn_idx=knn_idx)
        return x