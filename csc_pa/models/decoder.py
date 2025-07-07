import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import ASPP, get_syncbn
from .FPA import Foreground_Prototype_Module


class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res

class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.FPA_fu = Foreground_Prototype_Module(512)

        if self.rep_head:
            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )
        self.memory_linear = nn.Linear(512, 256)

    def forward(self, x, memobank, features):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        if aspp_out.shape[0] != 1:
            num_l = int(aspp_out.shape[0] / 2)
            aspp_out_l = aspp_out[:num_l]
            aspp_out_u = aspp_out[num_l:]
            aspp_out_l, aspp_out_u = self.FPA_fu(aspp_out_l, aspp_out_u)
            aspp_out_fu = torch.cat((aspp_out_l, aspp_out_u))

            if memobank.shape[0] != 0:
                if features.shape[0] != 0:
                    h = 16
                    if memobank.shape[0] < h*h and features.shape[0] == h*h:
                        memobank_updated = features
                    elif memobank.shape[0] == h * h and features.shape[0] == h * h:
                        memobank_reshaped = memobank.unsqueeze(0).permute(0, 2, 1)
                        feat_reshaped = features.unsqueeze(0).permute(0, 2, 1)
                        memobank_cat = torch.cat((memobank_reshaped, feat_reshaped), dim=2)
                        memobank_updated = self.memory_linear(memobank_cat)
                        memobank_updated = memobank_updated.permute(0, 2, 1).squeeze(0)
                    else:
                        memobank_updated = memobank
                else:
                    memobank_updated = memobank
                # --------- EPA ---------------------
                a, rep_dim, _, _ = aspp_out.shape
                fg = torch.sum(memobank_updated, dim=0) / memobank_updated.shape[0]  # 512
                fg_all = fg.unsqueeze(0).expand(a, -1).unsqueeze(-1).unsqueeze(-1).detach()
                cos = F.cosine_similarity(aspp_out, fg.view(1, rep_dim, 1, 1)).detach()
                att_all = F.softmax(cos, dim=1).unsqueeze(1).detach()
                global_feat = fg_all.expand_as(aspp_out) * att_all
                aspp_out = aspp_out + global_feat
            else:
                memobank_updated = memobank
            aspp_out = aspp_out + aspp_out_fu
        else:
            memobank_updated = memobank

        res = {"pred": self.classifier(aspp_out)}
        if aspp_out.shape[0] != 1:
            res["feat"] = aspp_out
            res["memory"] = memobank_updated
        else:
            res["feat"] = aspp_out

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res
