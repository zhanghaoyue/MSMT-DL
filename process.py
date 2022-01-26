from config import opt
from PIL import ImageFile
from utils import in_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_preprocess(img_list):
    # mirror input
    if opt.mirror:
        for_opt_list = img_list
        for_opt_list = in_model.mirror_slice(for_opt_list)

    # img dict
    img_dict = {
        'T1': for_opt_list[0],
        'T2': for_opt_list[1],
        'FL': for_opt_list[2],
        'BBOX': for_opt_list[3],
    }

    ######### pixel-wise transform #########
    # add win
    if opt.win:
        for key in img_dict:
            if key == 'BBOX':
                continue
            img_dict[key] = in_model.add_win(img_dict[key])

    # contrast
    if opt.contrast:
        for key in img_dict:
            if key == 'BBOX':
                continue
            img_dict[key] = in_model.augment_contrast(img_dict[key], contrast_range=opt.contrast_range)

    # brightness
    if opt.brightness:
        for key in img_dict:
            if key == 'BBOX':
                continue
            img_dict[key] = in_model.augment_brightness_multiplicative(img_dict[key], multiplier_range=opt.b_range)

    # noise
    if opt.noise == 'r':
        for key in img_dict:
            if key == 'BBOX':
                continue
            img_dict[key] = in_model.augment_rician_noise(img_dict[key])
    elif opt.noise == 'g':
        for key in img_dict:
            if key == 'BBOX':
                continue
            img_dict[key] = in_model.augment_gaussian_noise(img_dict[key])

    return img_dict


def val_preprocess(img_list):
    # img dict
    img_dict = {
        'T1': img_list[0],
        'T2': img_list[1],
        'FL': img_list[2],
        'BBOX': img_list[3],
    }

    if opt.win:
        for key in img_dict:
            if key == 'BBOX':
                continue
            img_dict[key] = in_model.add_win(img_dict[key])

    return img_dict
