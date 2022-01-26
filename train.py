import json
import random
import collections
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision.ops import box_iou
from config import opt
from utils import non_model
from make_dataset import train_Dataset, val_Dataset
from net import model_tools
import numpy as np
from tqdm import tqdm
import warnings
import resource

warnings.filterwarnings("ignore")



rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))


def train(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/all.txt')
    config_dict = opt._spec(kwargs)

    # stage 2
    save_model_folder = '../model/%s/' % (opt.path_key) + str(opt.net_idx) + '/'
    # info
    save_info_folder = '../info/%s/' % (opt.path_key) + str(opt.net_idx) + '/'
    non_model.make_path_folder(save_model_folder)
    non_model.make_path_folder(save_info_folder)
    with open(save_info_folder + 'config.json', 'w', encoding='utf-8') as json_file:
        json.dump(config_dict, json_file, ensure_ascii=False, indent=4)

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

        net = model_tools.get_model()
        net = net.cuda()

        lr = opt.lr
        if opt.optim == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=lr, weight_decay=opt.wd, momentum=0.9)
            print('================== SGD lr = %.6f ==================' % lr)

        elif opt.optim == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                          lr=lr, weight_decay=opt.wd)
            print('================== AdamW lr = %.6f ==================' % lr)
        if opt.cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Tmax,
                                                                  eta_min=opt.lr / opt.lr_gap)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.patience)

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

        train_slice_list = fold_list[str(k)]['train']
        train_set = train_Dataset(train_slice_list)
        train_data_num = len(train_set.img_list)
        train_batch = Data.DataLoader(dataset=train_set, batch_size=opt.train_bs, shuffle=True,
                                      num_workers=opt.num_workers, worker_init_fn=worker_init_fn,
                                      drop_last=True, collate_fn=non_model.num_collate)
        print('load train data done, num =', train_data_num)

        val_slice_list = fold_list[str(k)]['val']
        val_set = val_Dataset(val_slice_list)
        val_data_num = len(val_set.img_list)
        val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
                                    num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
        print('load val data done, num =', val_data_num)

        # return
        best_net = None
        epoch_save = 0
        best_metric = 0
        lr_change = 0

        loss_hist = collections.deque(maxlen=500)

        for e in range(opt.epoch):
            tmp_epoch = e + opt.start_epoch
            print('====================== Folder %s Epoch %s ========================' % (k, tmp_epoch))
            tmp_lr = optimizer.__getstate__()['param_groups'][0]['lr']

            if opt.cycle_r > 0:
                if e % (2 * opt.Tmax) == 0:
                    best_net = None
                    best_metric_list = np.zeros((opt.label_length - 1))
                    best_metric = 0
                    min_loss = 10
            else:
                if tmp_epoch > epoch_save + opt.gap_epoch:
                    break
                if lr_change == 2:
                    break

            net.train()

            for i, return_list in tqdm(enumerate(train_batch)):
                case_name, x, y = return_list

                im = Variable(x.type(torch.FloatTensor).cuda())
                label = Variable(y.type(torch.FloatTensor).cuda())

                if e == 0 and i == 0:
                    print('input size:', im.shape)
                # forward
                if opt.model[-4:] == 'rcnn':
                    loss_dict = net(im, label)
                    loss = sum(ls for ls in loss_dict.values())
                else:
                    classification_loss, regression_loss = net([im, label])

                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))

                if i % 50 == 0:
                    print(
                        'Ep: {} | Iter: {} | Cls loss: {:1.4f} | Reg loss: {:1.4f} | Running loss: {:1.4f}'.format(
                            tmp_epoch, i, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss

            torch.cuda.empty_cache()

            net = net.eval()
            val_loss = 0
            data_length = val_data_num

            all_detections = [None for j in range(data_length)]
            all_annotations = [None for j in range(data_length)]

            with torch.no_grad():
                for i, return_list in tqdm(enumerate(val_batch)):
                    case_name, x, y = return_list

                    ##################### Get detections ######################
                    im = Variable(x.type(torch.FloatTensor).cuda())

                    if e == 0 and i == 0:
                        print('input size:', im.shape)

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

                # compute recall and precision
                recall = true_positives / num_annotations
                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

                # compute average precision
                average_precision = non_model.compute_ap(recall, precision)

                print('mAP: {}'.format(average_precision))
                print("Precision: ", precision[-1])
                print("Recall: ", recall[-1])

                if average_precision > best_metric:
                    best_metric = average_precision
                    epoch_save = tmp_epoch
                    save_dict = {}
                    save_dict['net'] = net
                    save_dict['config_dict'] = config_dict
                    torch.save(save_dict, save_model_folder + 'K%s_%s_AP_%.4f_Pr_%.4f_Re_%.4f.pkl' %
                               (k, str(epoch_save).rjust(3, '0'), best_metric, precision[-1], recall[-1]))

                    info_dict = {
                        'fp': false_positives.tolist(),
                        'tp': true_positives.tolist(),
                        'score': scores.tolist(),
                        'anno': num_annotations
                    }
                    with open(save_info_folder + 'K%s_%s_AP_%.4f_Pr_%.4f_Re_%.4f.json' %
                              (k, str(epoch_save).rjust(3, '0'), best_metric, precision[-1], recall[-1]), 'w') as f:
                        json.dump(info_dict, f, indent=2)

                    del save_dict
                    del info_dict
                    print('====================== model save ========================')

            if opt.cos_lr == True:
                scheduler.step()
            else:
                scheduler.step(best_metric)

            before_lr = optimizer.__getstate__()['param_groups'][0]['lr']

            if before_lr != tmp_lr:
                epoch_save = tmp_epoch
                lr_change += 1
                print('================== lr change to %.6f ==================' % before_lr)

            torch.cuda.empty_cache()


if __name__ == '__main__':
    import fire

    fire.Fire()
