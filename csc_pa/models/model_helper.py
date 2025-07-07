import importlib

import torch.nn as nn
from torch.nn import functional as F

from .decoder import Aux_Module
import torch


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(BidirectionalCrossAttention, self).__init__()
        self.query1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.query2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        # x1: (batch_size, channels, height, width)
        # x2: (batch_size, channels, height, width)

        #  Query, Key 和 Value
        Q1 = self.query1(x1).view(x1.size(0), x1.size(1), -1)  # (batch_size, channels, height*width)
        K2 = self.key2(x2).view(x2.size(0), x2.size(1), -1)
        V2 = self.value2(x2).view(x2.size(0), x2.size(1), -1)

        Q2 = self.query2(x2).view(x2.size(0), x2.size(1), -1)
        K1 = self.key1(x1).view(x1.size(0), x1.size(1), -1)
        V1 = self.value1(x1).view(x1.size(0), x1.size(1), -1)

        attention1 = torch.bmm(Q1.transpose(1, 2), K2)  # (batch_size, height*width, height*width)
        attention1 = F.softmax(attention1, dim=-1)  # softmax

        attention2 = torch.bmm(Q2.transpose(1, 2), K1)  # (batch_size, height*width, height*width)
        attention2 = F.softmax(attention2, dim=-1)  # softmax

        out1 = torch.bmm(V2, attention1.transpose(1, 2))  # (batch_size, height*width, height*width)
        out1 = out1.view(x1.size(0), x1.size(1), x1.size(2), x1.size(3))

        out2 = torch.bmm(V1, attention2.transpose(1, 2))
        out2 = out2.view(x2.size(0), x2.size(1), x2.size(2), x2.size(3))

        fused_x1 = x1 + out1
        fused_x2 = x2 + out2

        return fused_x1, fused_x2

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x, memobank, features):
        if self._use_auxloss:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x)
                outs = self.decoder([f1, f2, feat1, feat2])
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat1)

            outs.update({"aux": pred_aux})
            return outs
        else:
            feat = self.encoder(x)
            outs = self.decoder(feat, memobank, features)
            return outs
