import torch.nn as nn
import torch
import math
from torchvision.models import *
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms, box_iou

from retinanet.anchors import Anchors
from retinanet import losses

import timm

import sys
sys.path.append("..")
from .block_utils import *
from config import opt
r'''
 ['adv_inception_v3', 
 
 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenetblur121d', 
 
 'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6', 
 
 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 
 
 'ecaresnet26t', 'ecaresnet50d', 'ecaresnet50d_pruned', 'ecaresnet50t', 'ecaresnet101d', 
 'ecaresnet101d_pruned', 'ecaresnet269d', 'ecaresnetlight', 
 
 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b1_pruned', 
 'efficientnet_b2', 'efficientnet_b2_pruned', 'efficientnet_b2a', 
 'efficientnet_b3', 'efficientnet_b3_pruned', 'efficientnet_b3a', 
 'efficientnet_em', 'efficientnet_es', 'efficientnet_lite0', 
 
 'ens_adv_inception_resnet_v2', 
 
 'ese_vovnet19b_dw', 'ese_vovnet39b', 
 
 'fbnetc_100', 
 
 'gernet_l', 'gernet_m', 'gernet_s', 
 
 'gluon_inception_v3', 
 'gluon_resnet18_v1b', 'gluon_resnet34_v1b', 
 'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 
 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 
 'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 
 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d', 
 'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 
 'gluon_xception65', 
 
 'hrnet_w18', 'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 
 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 
 
 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 
 
 'inception_resnet_v2', 'inception_v3', 'inception_v4', 
 
 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101', 
 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d', 
 
 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 
 
 'mnasnet_100', 
 
 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140', 
 'mobilenetv3_large_100', 'mobilenetv3_rw', 
 
 'nasnetalarge', 
 
 'nf_regnet_b1', 'nf_resnet50', 'nfnet_l0c', 
 
 'pnasnet5large', 
 
 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032', 'regnetx_040', 
 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 
 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 
 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320', 
 
 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b1g4', 'repvgg_b2', 'repvgg_b2g4', 'repvgg_b3', 'repvgg_b3g4', 
 
 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 
 'res2net101_26w_4s', 'res2next50', 
 
 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 
 'resnest101e', 'resnest200e', 'resnest269e', 
 
 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 
 'resnet101d', 'resnet152d', 'resnet200d', 'resnetblur50', 
 
 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k', 'resnetv2_50x3_bitm', 'resnetv2_50x3_bitm_in21k', 
 'resnetv2_101x1_bitm', 'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm', 'resnetv2_101x3_bitm_in21k', 
 'resnetv2_152x2_bitm', 'resnetv2_152x2_bitm_in21k', 'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k', 
 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d', 
 
 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 
 
 'selecsls42b', 'selecsls60', 'selecsls60b', 
 
 'semnasnet_100', 
 
 'seresnet50', 'seresnet152d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d', 
 
 'skresnet18', 'skresnet34', 'skresnext50_32x4d', 
 
 'spnasnet_100', 
 
 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 
 'ssl_resnext101_32x16d', 
 'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 
 'swsl_resnext101_32x16d', 
 
 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 
 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 
 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 
 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 
 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 
 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 
 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 
 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es', 
 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 
 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 
 
 'tf_inception_v3', 
 
 'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s', 
 
 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 
 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 
 
 'tresnet_l', 'tresnet_l_448', 'tresnet_m', 'tresnet_m_448', 'tresnet_xl', 'tresnet_xl_448', 
 
 'tv_densenet121', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 
 
 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
 
 'vit_base_patch16_224', 'vit_base_patch16_224_in21k', 'vit_base_patch16_384', 'vit_base_patch32_224_in21k', 
 'vit_base_patch32_384', 'vit_base_resnet50_224_in21k', 'vit_base_resnet50_384', 
 
 'vit_deit_base_distilled_patch16_224', 'vit_deit_base_distilled_patch16_384', 'vit_deit_base_patch16_224', 
 'vit_deit_base_patch16_384', 'vit_deit_small_distilled_patch16_224', 'vit_deit_small_patch16_224', 
 'vit_deit_tiny_distilled_patch16_224', 'vit_deit_tiny_patch16_224', 
 
 'vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_384', 'vit_large_patch32_224_in21k', 
 'vit_large_patch32_384', 'vit_small_patch16_224', 
 
 'wide_resnet50_2', 'wide_resnet101_2', 
 
 'xception', 'xception41', 'xception65', 'xception71']
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
    all_model_list = [
 'gluon_inception_v3',
 'gluon_resnet18_v1b', 'gluon_resnet34_v1b',
 'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s',
 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s',
 'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s',
 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d',
 'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d',
 'gluon_xception65',

 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d',

 'legacy_senet154',
 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101', 'legacy_seresnet152',
 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d',

 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s',
 'res2net101_26w_4s', 'res2next50',

 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d',
 'resnest101e', 'resnest200e', 'resnest269e',

 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d',
 'resnet101d', 'resnet152d', 'resnet200d', 'resnetblur50',

 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k', 'resnetv2_50x3_bitm', 'resnetv2_50x3_bitm_in21k',
 'resnetv2_101x1_bitm', 'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm', 'resnetv2_101x3_bitm_in21k',
 'resnetv2_152x2_bitm', 'resnetv2_152x2_bitm_in21k', 'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k',

 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d',

 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200',

 'selecsls42b', 'selecsls60', 'selecsls60b',

 'seresnet50', 'seresnet152d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d',

 'skresnet18', 'skresnet34', 'skresnext50_32x4d',

 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1',
 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap',
 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns',
 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5',
 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap',
 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns',
 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e',
 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es',
 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1',
 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4',

 'tf_inception_v3',

 'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s',

 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100',
 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100',

 'tresnet_l', 'tresnet_l_448', 'tresnet_m', 'tresnet_m_448', 'tresnet_xl', 'tresnet_xl_448',

 'tv_densenet121', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d',

 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',

 'wide_resnet50_2', 'wide_resnet101_2',

 'xception', 'xception41', 'xception65', 'xception71']

    #model = timm.create_model(model_dict[name], pretrained=opt.pre_train, features_only=True)
    model_name = 'gluon_resnet50_v1d'
    model = timm.create_model(model_name, pretrained=opt.pre_train, features_only=True)
    print(model_name)
    # print(f'Feature channels: {model.feature_info.channels()}')

    if opt.pre_train:
        print('use pretrained model')
    return model

class Get_FE(nn.Module):
    def __init__(self):
        super(Get_FE, self).__init__()
        FE = encoder(opt.model, opt.pre_train)

        # if opt.dim == 3:
        #     self.conv1 = FE.conv1
        # else:
        #     self.conv1 = nn.Conv2d(opt.dim, 64, (7, 7), (2, 2), (3, 3), bias=False)
        #     conv_weights = FE.conv1.state_dict()['weight'][:, 1, :, :].unsqueeze(dim=1).repeat((1, opt.dim, 1, 1))
        #     self.conv1.state_dict()['weight'] = conv_weights
        #
        # self.bn1 = FE.bn1
        # self.relu = FE.act1
        # self.maxpool = FE.maxpool
        # self.layer1 = FE.layer1
        # self.layer2 = FE.layer2
        # self.layer3 = FE.layer3
        # self.layer4 = FE.layer4
        #
        # layers = [len(self.layer1), len(self.layer2), len(self.layer3), len(self.layer4)]
        #
        # del FE
        #
        # fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
        #              self.layer3[layers[2] - 1].conv3.out_channels,
        #              self.layer4[layers[3] - 1].conv3.out_channels]
        #
        # # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        # self.fpn = PyramidFeatures(*fpn_sizes)
        #
        # self.anchors = Anchors()
        # num_anchors = self.anchors.ratios.shape[0] * self.anchors.scales.shape[0]
        #
        # self.regressionModel = RegressionModel(256, num_anchors=num_anchors)
        # self.classificationModel = ClassificationModel(256, num_anchors=num_anchors, num_classes=num_classes)
        #
        # self.regressBoxes = BBoxTransform()
        #
        # self.clipBoxes = ClipBoxes()
        #
        # self.focalLoss = losses.FocalLoss(alpha=opt.alpha, lth=opt.lth, hth=opt.hth)
        #
        # prior = 0.01
        # self.classificationModel.output.weight.data.fill_(0)
        # self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.regressionModel.output.weight.data.fill_(0)
        # self.regressionModel.output.bias.data.fill_(0)
        #
        # if opt.fix_BN:
        #     self.freeze_bn()

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

            return self.focalLoss(classification, regression, anchors, annotations)
            # return


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