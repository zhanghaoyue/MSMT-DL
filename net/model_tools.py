from config import opt
import torchvision
from . import model_resnet, model_resnet_new, model_timm_net


def get_model():
    global net
    if opt.model[:2] == 'nr':
        net = model_resnet_new.ResNet(num_classes=opt.label_length)
    elif opt.model[0] == 't':
        net = model_timm_net.Timm(num_classes=opt.label_length)
    elif opt.model[0] == 'r':
        # Create the model
        if opt.model == 'r18':
            net = model_resnet.resnet18(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r34':
            net = model_resnet.resnet34(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r50':
            net = model_resnet.resnet50(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r101':
            net = model_resnet.resnet101(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r152':
            net = model_resnet.resnet152(num_classes=opt.label_length, pretrained=opt.pre_train)
    elif opt.model[-4:] == 'rcnn':
        if opt.model == 'faster_rcnn':
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'mask_rcnn':
            net = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=opt.label_length, pretrained=opt.pre_train)

    return net


def get_result():
    return
