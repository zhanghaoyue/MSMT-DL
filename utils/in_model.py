import numpy as np
import cv2
import random
from copy import deepcopy
from config import opt

from builtins import range


################################## for data ##################################
def get_img(img_path, path_name):
    tmp_file = np.load(img_path + path_name + '.npz')
    tmp_T1 = tmp_file['T1']
    tmp_T2 = tmp_file['T2']
    tmp_FL = tmp_file['FL']
    tmp_BBOX = tmp_file['BBOX']

    tmp_data = [tmp_T1, tmp_T2, tmp_FL, tmp_BBOX]

    return tmp_data


def get_bbox(bbox_array):
    annotations = np.zeros((0, 5)).astype('int16')

    bbox_array[bbox_array != 0] = 1

    if bbox_array.max() == 0:
        return annotations

    anno_contour_list = cv2.findContours(
        bbox_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    for anno_contour in anno_contour_list:
        x_min = np.min(anno_contour[:, :, 0])
        x_max = np.max(anno_contour[:, :, 0])

        y_min = np.min(anno_contour[:, :, 1])
        y_max = np.max(anno_contour[:, :, 1])

        if (x_max - x_min) < 1 or (y_max - y_min) < 1:
            continue

        annotation = np.zeros((1, 5))
        annotation[0, 0] = x_min
        annotation[0, 1] = y_min
        annotation[0, 2] = x_max
        annotation[0, 3] = y_max
        annotation[0, 4] = 1

        annotations = np.append(annotations, annotation, axis=0)
    return annotations


################################## for process ##################################
# tools

# resize
# f - fix
# r - random
# fr - fix + random
def resize_slice(input_list, h, w, inp_list):
    output_list = []

    for idx, input_array in enumerate(input_list):
        if isinstance(input_array, int):
            output_list.append(0)
        else:
            output_array = []
            for input_slice in input_array:
                output_slice = cv2.resize(input_slice, (w, h), interpolation=inp_list[idx])
                output_array.append(output_slice)
            output_array = np.array(output_array)
            output_list.append(output_array)

    return output_list[0], output_list[1], output_list[2:]


def rot_slice(input_list, rot_angle):
    output_list = []

    if opt.rot == 'r':
        img = input_list[0]
        img_z, img_y, img_x = img.shape
        center = (img_x // 2, img_y // 2)
        rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1)

    for idx, input_array in enumerate(input_list):
        if isinstance(input_array, int):
            output_list.append(0)
        else:
            output_array = []
            for input_slice in input_array:
                if opt.rot == 'r':
                    output_slice = cv2.warpAffine(input_slice, rot_mat, flags=3, )
                else:
                    output_slice = np.rot90(input_slice, rot_angle)
                output_array.append(output_slice)
            output_array = np.array(output_array)
            output_list.append(output_array)

    return output_list[0], output_list[1], output_list[2:]


# crop
# mc: center crop with mask
# mr: random crop with mask
# c: center crop
# r: random crop
def crop_slice(img, mask, matrix_list):
    img_z, img_y, img_x = img.shape

    crop_y_s = 0
    crop_y_e = img_y

    crop_x_s = 0
    crop_x_e = img_x

    if 'm' in opt.crop:
        mask_pos = np.where(mask != 0)

        if img_y > opt.c_y:
            mask_y_s = mask_pos[1].min()
            mask_y_e = mask_pos[1].max() + 1

            if opt.crop == 'mc':
                mask_y_mid = (mask_y_s + mask_y_e) // 2
                crop_y_s = max(0, mask_y_mid - opt.c_y // 2)
                crop_y_e = min(img_y, crop_y_s + opt.c_y)
                crop_y_s = min(crop_y_s, crop_y_e - opt.c_y)

            elif opt.crop == 'mr':
                crop_y_s_left = max(0, mask_y_e - opt.c_y)
                crop_y_s_right = min(img_y - opt.c_y, mask_y_s)
                crop_y_s = random.randint(crop_y_s_left, crop_y_s_right)
                crop_y_e = crop_y_s + opt.c_y

        if img_x > opt.c_x:
            mask_x_s = mask_pos[2].min()
            mask_x_e = mask_pos[2].max() + 1

            if opt.crop == 'mc':
                mask_x_mid = (mask_x_s + mask_x_e) // 2
                crop_x_s = max(0, mask_x_mid - opt.c_x // 2)
                crop_x_e = min(img_x, crop_x_s + opt.c_x)
                crop_x_s = min(crop_x_s, crop_x_e - opt.c_x)

            elif opt.crop == 'mr':
                crop_x_s_left = max(0, mask_x_e - opt.c_x)
                crop_x_s_right = min(img_x - opt.c_x, mask_x_s)
                crop_x_s = random.randint(crop_x_s_left, crop_x_s_right)
                crop_x_e = crop_x_s + opt.c_x

    else:
        if img_y > opt.c_y:
            if opt.crop == 'c':
                crop_y_s = (img_y - opt.c_y) // 2
                crop_y_e = crop_y_s + opt.c_y
            elif opt.crop == 'r':
                y_gap = img_y - opt.c_y
                crop_y_s_left = y_gap // 4
                crop_y_s_right = y_gap // 4 * 3
                crop_y_s = random.randint(crop_y_s_left, crop_y_s_right)
                crop_y_e = crop_y_s + opt.c_y

        if img_x > opt.c_x:
            if opt.crop == 'c':
                crop_x_s = (img_x - opt.c_x) // 2
                crop_x_e = crop_x_s + opt.c_x
            elif opt.crop == 'r':
                x_gap = img_x - opt.c_x
                crop_x_s_left = x_gap // 4
                crop_x_s_right = x_gap // 4 * 3
                crop_x_s = random.randint(crop_x_s_left, crop_x_s_right)
                crop_x_e = crop_x_s + opt.c_x

    img = img[:, crop_y_s:crop_y_e, crop_x_s:crop_x_e]
    mask = mask[:, crop_y_s:crop_y_e, crop_x_s:crop_x_e]

    for m_idx in range(len(matrix_list)):
        if isinstance(matrix_list[m_idx], int) == False:
            matrix_list[m_idx] = matrix_list[m_idx][:, crop_y_s:crop_y_e, crop_x_s:crop_x_e]

    return img, mask, matrix_list


# pad
# c: center pad
# r: random pad
def pad_slice(input_list):
    img = input_list[0]
    value_list = [img.min()] + [0] * len(input_list)

    img_z, img_y, img_x = img.shape

    if opt.pad == 'c':
        pad_top = (opt.p_y - img_y) // 2
        pad_bottom = opt.p_y - img_y - pad_top

        pad_left = (opt.p_x - img_x) // 2
        pad_right = opt.p_x - img_x - pad_left

    elif opt.pad == 'r':
        pad_top = random.randint(0, opt.p_y - img_y)
        pad_bottom = opt.p_y - img_y - pad_top

        pad_left = random.randint(0, opt.p_x - img_x)
        pad_right = opt.p_x - img_x - pad_left

    output_list = []
    for idx, input_array in enumerate(input_list):
        if isinstance(input_array, int):
            output_list.append(0)

        else:
            tmp_v = float(value_list[idx])
            output_array = []
            for input_slice in input_array:
                output_slice = cv2.copyMakeBorder(input_slice, pad_top, pad_bottom, pad_left, pad_right, \
                                                  cv2.BORDER_CONSTANT, value=tmp_v)
                output_array.append(output_slice)
            output_array = np.array(output_array)
            output_list.append(output_array)

    return output_list[0], output_list[1], output_list[2:]


# mirror
# h: h flip
# w: v flip
# hw: h flip + w flip
def mirror_slice(input_list):
    output_list = []
    mirror_flag = False
    if np.random.uniform() < 0.5:
        mirror_flag = True

    for input_array in input_list:
        output_array = deepcopy(input_array)
        if mirror_flag:
            output_array[:, :] = output_array[:, ::-1]

        output_list.append(output_array)

    return output_list


# win
# white: (x - x.mean) / x.std
# win: (x[win_min:win_max] - win_min) / (win_max - win_min)
# un: (x - win_min) / (win_max - win_min)
def add_win(img):
    if opt.win == 'white':
        img_mean = np.mean(img)
        img_std = np.std(img)
        img = (img - img_mean) / img_std
    elif opt.win == 'un':
        left_win = img.min()
        right_win = img.max()
        img[img < left_win] = left_win
        img[img > right_win] = right_win
        img = (img - left_win) / (right_win - left_win)
    elif opt.win == 'ci':
        left_win = np.percentile(img, 2.5)
        right_win = np.percentile(img, 97.5)
        img[img < left_win] = left_win
        img[img > right_win] = right_win
        img = (img - left_win) / (right_win - left_win)

    return img


# 对比度增强
def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True):
    mn = data_sample.mean()
    if preserve_range:
        minm = data_sample.min()
        maxm = data_sample.max()
    if np.random.random() < 0.5 and contrast_range[0] < 1:
        factor = np.random.uniform(contrast_range[0], 1)
    else:
        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
    data_sample = (data_sample - mn) * factor + mn
    if preserve_range:
        data_sample[data_sample < minm] = minm
        data_sample[data_sample > maxm] = maxm

    return data_sample


# brightness
def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2)):
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    data_sample *= multiplier

    return data_sample


# rician_noise
def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2)
    return data_sample


# gaussian_noise
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample
