import torch.nn as nn
import torch
import math
from torchvision.models import *
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from torchvision.ops import box_iou
import timm

import sys
sys.path.append("..")
from .block_utils import *
from config import opt

r'''
 'res2net50_14w_8s',
 'res2net50_26w_4s',
 'res2net50_26w_6s',
 'res2net50_26w_8s',
 'res2net50_48w_2s',
 'res2net101_26w_4s',
 'res2next50',]
'''

def encoder(name, pre_train):
    model_dict = {
        'tr2n50w14s8': 'res2net50_14w_8s',
        'tr2n50w26s4': 'res2net50_26w_4s',
        'tr2n50w26s6': 'res2net50_26w_6s',
        'tr2n50w26s8': 'res2net50_26w_8s',
        'tr2n50w48s2': 'res2net50_48w_2s',
        'tr2n101w26s4': 'res2net101_26w_4s',
        'tr2nx50': 'res2next50',
    }

    model = timm.create_model(model_dict[name], pretrained=opt.pre_train)
    if opt.pre_train:
        print('use pretrained model')
    return model

class Timm(nn.Module):
    def __init__(self, num_classes):
        super(Timm, self).__init__()
        FE = encoder(opt.model, opt.pre_train)

        if opt.dim == 3:
            self.conv1 = FE.conv1
        else:
            self.conv1 = nn.Conv2d(opt.dim, 64, (7, 7), (2, 2), (3, 3), bias=False)
            conv_weights = FE.conv1.state_dict()['weight'][:, 1, :, :].unsqueeze(dim=1).repeat((1, opt.dim, 1, 1))
            self.conv1.state_dict()['weight'] = conv_weights

        self.bn1 = FE.bn1
        self.relu = FE.act1
        self.maxpool = FE.maxpool
        self.layer1 = FE.layer1
        self.layer2 = FE.layer2
        self.layer3 = FE.layer3
        self.layer4 = FE.layer4

        layers = [len(self.layer1), len(self.layer2), len(self.layer3), len(self.layer4)]

        del FE

        fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                     self.layer3[layers[2] - 1].conv3.out_channels,
                     self.layer4[layers[3] - 1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.anchors = Anchors()
        num_anchors = self.anchors.ratios.shape[0] * self.anchors.scales.shape[0]

        self.regressionModel = RegressionModel(256, num_anchors=num_anchors)
        self.classificationModel = ClassificationModel(256, num_anchors=num_anchors, num_classes=num_classes)

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss(alpha=opt.alpha, lth=opt.lth, hth=opt.hth)

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        if opt.fix_BN:
            self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        if opt.eval == True:

            for feature in features:
                print(feature.shape)

            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
            print(regression.shape)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            print(classification.shape)

            anchors = self.anchors(img_batch)
            print('================anchors==============')
            print(anchors.shape)
            return


        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > opt.cls_th)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, opt.nms_th)

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]