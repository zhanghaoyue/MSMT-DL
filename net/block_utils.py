import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union

sys.path.append("..")
from config import opt

# Nets
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        if opt.eval == True:
            print('regression')
            print(out.shape)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        if opt.eval == True:
            print('ClassificationModel')
            print(out2.shape)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

# Tools
class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes

class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


# Non-local for 1/2/3 dimension
class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

        # self.to_fpn = conv_nd(in_channels=self.in_channels, out_channels=256, kernel_size=1)

    def forward(self, x1, x2):
        """
        args
            x: (N, C, T, H, W) for dimension=3;
               (N, C, H, W) for dimension 2;
               (N, C, T) for dimension 1
        """
        batch_size = x1.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x2).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # print(g_x.shape)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x2).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            f = torch.matmul(theta_x, phi_x)
            # print(f.shape)


        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
            # print(f_div_C.shape)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        # print(y.shape)
        # return 0

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x1.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x1

        return z

class MC_se_block(nn.Module):
    def __init__(self, in_c):
        super(MC_se_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_c * 3, in_c * 3 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c * 3 // 16, in_c * 3, bias=False),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.concat = nn.Conv2d(in_c * 3, in_c, 1)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)

        x_SE = x * (avg_out + max_out).expand_as(x)
        x_out = self.concat(x_SE)

        return x_out

class CASA_block(nn.Module):
    def __init__(self, in_c):
        super(CASA_block, self).__init__()
        self.CA_fc = nn.Sequential(
            nn.Linear(in_c * 3, in_c * 3 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c * 3 // 16, in_c * 3, bias=False),
            nn.Sigmoid()
        )

        self.CA_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_max_pool = nn.AdaptiveMaxPool2d(1)

        self.SA_fc = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Conv2d(in_c * 3, in_c, 1)

    def forward(self, x):
        b, c, _, _ = x.size()
        CA_avg_out = self.CA_fc(self.CA_avg_pool(x).view(b, c)).view(b, c, 1, 1)
        CA_max_out = self.CA_fc(self.CA_max_pool(x).view(b, c)).view(b, c, 1, 1)

        x_CA = x * (CA_avg_out + CA_max_out).expand_as(x)

        x_SA_avg = torch.mean(x, dim=1, keepdim=True)
        x_SA_max, _ = torch.max(x, dim=1, keepdim=True)
        x_SA_w = self.SA_fc(torch.cat([x_SA_avg, x_SA_max], dim=1))
        x_SA = x * x_SA_w.expand_as(x)

        x = x_CA + x_SA

        x_out = self.conv_out(x)

        return x_out

####### mask rcnn + cls branch #######
def mr_cls_backbone_pre(self, x):
    final_output = self.body(x)
    x = self.fpn(final_output)
    return x, final_output

def mr_cls_forward(self, images, targets=None):
    # print('in my forward')
    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")
    if self.training:
        assert targets is not None
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError("Expected target boxes to be a tensor"
                                     "of shape [N, 4], got {:}.".format(
                        boxes.shape))
            else:
                raise ValueError("Expected target boxes to be of type "
                                 "Tensor, got {:}.".format(type(boxes)))

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width."
                                 " Found invalid box {} for target at index {}."
                                 .format(degen_bb, target_idx))

    losses = {}

    features, backbone_features = self.backbone(images.tensors)

    if self.mode == 'det':
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses.update(detector_losses)
        losses.update(proposal_losses)
    elif self.mode in ['cls' ,'cam']:
        detections = None

    if self.training:
        cls_label_list = []
        for target in targets:
            tmp_label = target['labels'].clone()[0].squeeze().unsqueeze(0)
            cls_label_list.append(tmp_label)

        cls_label = torch.cat(cls_label_list, dim=0)
        if opt.cls_loss == 'bce':
            cls_label = cls_label.unsqueeze(1).type(torch.FloatTensor).cuda()
        cls_losses = {}

        for k, v in backbone_features.items():
            if int(k) not in opt.cls_num:
                continue
            cls_out = eval('self.cls_branch_%s(v)'%k)
            cls_losses['cls_loss_%s'%k] = self.cls_loss(cls_out, cls_label)
        # return
        losses.update(cls_losses)
    else:
        cls_out = self.cls_branch_3(backbone_features['3'])
        if self.mode == 'det':
            for det_idx in range(len(detections)):
                detections[det_idx]['cls_out'] = cls_out[det_idx]
        elif self.mode in ['cls', 'cam']:
            detections = []
            for cls_out_score in cls_out:
                detections.append({'cls_out':cls_out_score.squeeze()})

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)

####### mask rcnn + multi-modal #######
def MC_forward(self, x):
    x_list = torch.split(x, 1, dim=1)
    out_list = []
    for x_idx, x in enumerate(x_list):
        x = x.repeat(1,3,1,1)
        if opt.MC_share == True:
            x = self.body(x)
        elif x_idx == 0:
            x = self.body(x)
        else:
            x = eval('self.body_%s(x)' % x_idx)
        out_list.append(x)

    final_output = out_list[-1]
    for k in final_output.keys():
        if int(k) not in opt.dv_num:
            continue
        k_list = []
        for output in out_list:
            v = output[k]
            k_list.append(v)

        if opt.MC_block == 'NL':
            k_Q = k_list[-1]
            k_K = torch.cat(k_list, dim=2)
            k_out = eval('self.block_%s(k_Q, k_K)' % k)
        elif opt.MC_block in ['CO', 'SE', 'CASA']:
            k_concat = torch.cat(k_list, dim=1)
            k_out = eval('self.block_%s(k_concat)' % k)

        final_output[k] = k_out

    x = self.fpn(final_output)
    return x

def MP_forward(self, x):
    x_list = torch.split(x, 1, dim=1)
    out_list = []
    for x_idx, x in enumerate(x_list):
        x = x.repeat(1,3,1,1)
        out = OrderedDict()
        for b in range(5):
            if b in opt.share_b:
                x = eval('self.main_%s(x)' % b)
            else:
                x = eval('self.branch_%s_%s(x)' % (b, x_idx))

            if b != 0:
                out[str(int(b-1))] = x

        out_list.append(out)

    final_output = out_list[-1]
    for k in final_output.keys():
        if int(k) not in opt.dv_num:
            continue
        k_list = []
        for output in out_list:
            v = output[k]
            k_list.append(v)

        if opt.MC_block == 'NL':
            k_Q = k_list[-1]
            k_K = torch.cat(k_list, dim=2)
            k_out = eval('self.block_%s(k_Q, k_K)' % k)
        elif opt.MC_block in ['CO','SE']:
            k_concat = torch.cat(k_list, dim=1)
            k_out = eval('self.block_%s(k_concat)' % k)

        final_output[k] = k_out

    x = self.fpn(final_output)

    return x

def MC_cls_backbone_forward(self, x):
    x_list = torch.split(x, 1, dim=1)
    out_list = []
    for x_idx, x in enumerate(x_list):
        x = x.repeat(1, 3, 1, 1)
        if opt.MC_share == True:
            x = self.body(x)
        elif x_idx == 0:
            x = self.body(x)
        else:
            x = eval('self.body_%s(x)' % x_idx)
        out_list.append(x)

    final_output = out_list[-1]
    for k in final_output.keys():
        if int(k) not in opt.dv_num:
            continue
        k_list = []
        for output in out_list:
            v = output[k]
            k_list.append(v)

        if opt.MC_block == 'NL':
            k_Q = k_list[-1]
            k_K = torch.cat(k_list, dim=2)
            k_out = eval('self.block_%s(k_Q, k_K)' % k)
        elif opt.MC_block in ['CO', 'SE', 'CASA']:
            k_concat = torch.cat(k_list, dim=1)
            k_out = eval('self.block_%s(k_concat)' % k)

        final_output[k] = k_out

    x = self.fpn(final_output)
    return x, final_output
