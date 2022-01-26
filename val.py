import os
import random
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision.ops import box_iou
from config import opt
from utils import non_model
from make_dataset import val_Dataset
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))


def val(**kwargs):
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/all.txt')
    config_dict = opt._spec(kwargs)

    save_model_folder = '../model/%s/' % (opt.path_key) + str(opt.net_idx) + '/'
    save_output_folder = '../val_output/%s/' % (opt.path_key) + str(opt.net_idx) + '/'
    non_model.make_path_folder(save_output_folder)

    save_model_list = sorted(os.listdir(save_model_folder))
    init_model_path = save_model_folder + sorted(os.listdir(save_model_folder))[0]
    config_dict = non_model.update_kwargs(init_model_path, kwargs)
    config_dict.pop('kidx')
    config_dict.pop('path_img')
    config_dict.pop('cls_th')
    config_dict.pop('nms_th')
    config_dict.pop('s_th')
    config_dict.pop('max_dets')
    config_dict.pop('iou_th')
    opt._spec(config_dict)
    print('load config done')

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    fold_list = data_info_dict['Train']

    for k in opt.kidx:
        GLOBAL_SEED = 2021
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        data_gpu = opt.gpu_idx
        torch.cuda.set_device(data_gpu)

        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        GLOBAL_WORKER_ID = None

        def worker_init_fn(worker_id):
            global GLOBAL_WORKER_ID
            GLOBAL_WORKER_ID = worker_id
            set_seed(GLOBAL_SEED + worker_id)

        val_slice_list = fold_list[str(k)]['val']
        val_set = val_Dataset(val_slice_list)
        val_data_num = len(val_set.img_list)
        val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
                                    num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
        print('load val data done, num =', val_data_num)

        tmp_save_model_list = [each for each in save_model_list if each.startswith('K%s' % k)]

        for save_model in tmp_save_model_list:
            print(save_model)
            save_model_path = save_model_folder + save_model
            save_dict = torch.load(save_model_path, map_location=torch.device('cpu'))
            config_dict = save_dict['config_dict']
            config_dict.pop('kidx')
            config_dict.pop('path_img')
            config_dict.pop('cls_th')
            config_dict.pop('nms_th')
            config_dict.pop('s_th')
            config_dict.pop('max_dets')
            config_dict.pop('iou_th')
            config_dict['mode'] = 'test'
            opt._spec(config_dict)
            net = save_dict['net']
            del save_dict

            net = net.cuda()
            net = net.eval()

            data_length = val_data_num
            all_slices = [None for j in range(data_length)]
            all_detections = [None for j in range(data_length)]
            all_annotations = [None for j in range(data_length)]

            with torch.no_grad():
                for i, return_list in tqdm(enumerate(val_batch)):
                    case_name, x, y = return_list
                    all_slices[i] = case_name[0]

                    ##################### Get detections ######################
                    im = Variable(x.type(torch.FloatTensor).cuda())

                    # forward
                    scores, labels, boxes = net(im)
                    scores = scores.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    boxes = boxes.detach().cpu().numpy()

                    indices = np.where(scores > opt.s_th)[0]

                    if indices.shape[0] > 0:
                        scores = scores[indices]
                        boxes = boxes[indices]
                        labels = labels[indices]

                        # find the order with which to sort the scores
                        scores_sort = np.argsort(-scores)[:opt.max_dets]

                        # select detections
                        image_boxes = boxes[scores_sort]
                        image_scores = scores[scores_sort]
                        image_labels = labels[scores_sort]

                        image_detections = np.concatenate(
                            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)],
                            axis=1)

                        all_detections[i] = image_detections[:, :-1]
                    else:
                        all_detections[i] = np.zeros((0, 5))
                    ###########################################################

                    ##################### Get annotations #####################
                    annotations = y.detach().cpu().numpy()[0]
                    all_annotations[i] = annotations[:, :4]
                    ###########################################################

            np.savez(save_output_folder + 'K%s_output.npz' % k, case=all_slices, det=all_detections,
                     anno=all_annotations)

            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(data_length):
                detections = all_detections[i]
                annotations = all_annotations[i]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    d_tensor = torch.tensor(d[:4][np.newaxis])
                    a_tensor = torch.tensor(annotations)
                    overlaps = box_iou(d_tensor, a_tensor).numpy()
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= opt.iou_th and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            if len(false_positives) == 0 and len(true_positives) == 0:
                print('No detection')
            else:
                # sort by score
                indices = np.argsort(-scores)
                scores = scores[indices]
                false_positives = false_positives[indices]
                true_positives = true_positives[indices]

                # compute false positives and true positives
                false_positives = np.cumsum(false_positives)
                true_positives = np.cumsum(true_positives)

                # compute F1 recall and precision
                recall = true_positives / num_annotations
                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
                F1_score = (2 * precision * recall) / (recall + precision)
                average_precision = non_model.compute_ap(recall, precision)

                # recall > 0.8
                indices = np.where(recall > 0.8)[0]
                scores = scores[indices]
                recall = recall[indices]
                precision = precision[indices]
                F1_score = F1_score[indices]
                F1_max_idx = np.argmax(F1_score)

                # output
                print('mAP: %.4f' % average_precision)
                print('Score TH: %.4f' % scores[F1_max_idx])
                print('F1: %.4f' % F1_score[F1_max_idx])
                print("Precision: %.4f" % precision[F1_max_idx])
                print("Recall: %.4f" % recall[F1_max_idx])


if __name__ == '__main__':
    import fire

    fire.Fire()
