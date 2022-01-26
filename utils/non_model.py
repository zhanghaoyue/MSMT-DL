import numpy as np
import os
import json
import random
import torch
from builtins import range
from config import opt
import sys


################################## For config ##################################
def read_kwargs(kwargs):
    if 'path_key' not in kwargs:
        print('Error: no path key')
        sys.exit()
    else:
        dict_path = '../config/%s_dict.json' % kwargs['path_key']
        with open(dict_path, 'r') as f:
            data_info_dict = json.load(f)
        kwargs['path_img'] = data_info_dict["path_img"]

    if 'kidx' not in kwargs:
        kwargs['kidx'] = list(range(len(data_info_dict['Train'])))

    if 'model' not in kwargs:
        print('Error: no model')
        sys.exit()

    if 'net_idx' not in kwargs:
        print('Error: no net idx')
        sys.exit()

    # model
    # optim set
    if 'optim' in kwargs and kwargs['optim'] == 'SGD':
        if 'wd' not in kwargs:
            kwargs['wd'] = 0.00001
        if 'lr' not in kwargs:
            kwargs['lr'] = 0.01

    # cycle learning
    if 'cycle_r' in kwargs and int(kwargs['cycle_r']) > 0:
        if 'Tmax' not in kwargs:
            kwargs['Tmax'] = 20
        kwargs['cos_lr'] = True
        kwargs['epoch'] = int(kwargs['cycle_r']) * 2 * kwargs['Tmax']
        kwargs['gap_epoch'] = kwargs['epoch'] + 1

    # loss
    return kwargs, data_info_dict


def update_kwargs(init_model_path, kwargs):
    save_dict = torch.load(init_model_path, map_location=torch.device('cpu'))
    config_dict = save_dict['config_dict']
    del save_dict

    config_dict.pop('gpu_idx')
    config_dict['mode'] = 'test'

    if 'val_bs' in kwargs:
        config_dict['val_bs'] = val_bs

    return config_dict


################################## For path ##################################
def make_path_folder(path):
    path_split = path.split('/')
    path_length = len(path_split)
    for i in range(2, path_length):
        tmp_path = '/'.join(path_split[:i]) + '/'
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
    return


################################## For class-balance in mini-batch ##################################
def get_class_list(data_list, class_dict):
    class_list = []

    if opt.label_length == 2:
        for data in data_list:
            class_list.append(class_dict[data])
    else:
        # to do
        pass

    return class_list


def batch_class_balance(data_list, class_list):
    class_list = np.array(class_list)
    neg_idx_list = np.where(class_list == 0)[0].tolist()
    pos_idx_list = np.where(class_list == 1)[0].tolist()

    random.shuffle(pos_idx_list)

    bc_train_idx_list = []

    while True:
        pos_num = random.randint(int(opt.train_bs / 3), int(opt.train_bs / 3 * 2))
        neg_num = opt.train_bs - pos_num
        tmp_batch = pos_idx_list[:pos_num] + random.sample(neg_idx_list, neg_num)

        if len(pos_idx_list) == 0:
            break
        elif pos_num > len(pos_idx_list):
            break
        else:
            pos_idx_list = pos_idx_list[pos_num:]

        random.shuffle(tmp_batch)
        bc_train_idx_list += tmp_batch

    bc_train_list = np.array(data_list)[bc_train_idx_list].tolist()

    return bc_train_list


################################## For batch preprocess ##################################
def num_collate(data):
    case_list = []
    img_list = []
    annot_list = []
    batch_size = len(data)

    for tmp_data in data:
        tmp_case, tmp_img, tmp_annot = tmp_data
        case_list.append(tmp_case)
        img_list.append(tmp_img)
        annot_list.append(tmp_annot)

    max_num_annots = max(annot.shape[0] for annot in annot_list)

    if max_num_annots > 0:
        annot_padded = np.ones((batch_size, max_num_annots, 5)) * -1
        for idx, annot in enumerate(annot_list):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((batch_size, 1, 5)) * -1

    img_batch = torch.tensor(np.array(img_list))
    annot_batch = torch.tensor(annot_padded)

    return case_list, img_batch, annot_batch


################################## For Metric ##################################
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
