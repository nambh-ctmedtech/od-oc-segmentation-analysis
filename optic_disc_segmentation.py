import os, json, sys
import os.path as osp
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torchvision
from models.get_model import get_arch
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model
from skimage import measure
import pandas as pd
from skimage.morphology import remove_small_objects
import logging
from itertools import product

#================================================================#

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None, help='.cfg file')

def swap(a, b):
    return b, a

def prediction_eval(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8, test_loader, device):
    # seg_results_small_path = f'results/resized/'
    # if not os.path.isdir(seg_results_small_path):
    #     os.makedirs(seg_results_small_path)

    # seg_results_raw_path = f'images/segmented/'
    # if not os.path.isdir(seg_results_raw_path):
    #     os.makedirs(seg_results_raw_path)

    with tqdm(total=len(test_loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            imgs = batch['image']
            img_name = batch['name']
            ori_width = batch['original_sz'][0]
            ori_height = batch['original_sz'][1]
            mask_pred_tensor_small_all = 0

            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                _, mask_pred = model_1(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_1 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_1.type(torch.FloatTensor)
                
                
                _, mask_pred= model_2(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_2 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_2.type(torch.FloatTensor)
                

                _, mask_pred = model_3(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_3 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_3.type(torch.FloatTensor)                
                

                _, mask_pred = model_4(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_4 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_4.type(torch.FloatTensor)    
                

                _, mask_pred = model_5(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_5 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_5.type(torch.FloatTensor)    
                

                _, mask_pred = model_6(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_6 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_6.type(torch.FloatTensor)   
                

                _, mask_pred = model_7(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_7 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_7.type(torch.FloatTensor)   
                

                _, mask_pred = model_8(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_8 = F.softmax(mask_pred_tensor_small, dim=1)
                mask_pred_tensor_small_all += mask_pred_tensor_small_8.type(torch.FloatTensor)   
                
                mask_pred_tensor_small_all = (mask_pred_tensor_small_all/8).to(device=device) 

                _, prediction_decode = torch.max(mask_pred_tensor_small_all, 1)
                prediction_decode = prediction_decode.type(torch.FloatTensor)

                n_img = prediction_decode.shape[0]
                
                if len(prediction_decode.size())==3:
                    torch.unsqueeze(prediction_decode, 0)

                for i in range(n_img):
                    img_r = np.zeros((prediction_decode[i, ...].shape[0], prediction_decode[i, ...].shape[1]))
                    img_g = np.zeros((prediction_decode[i, ...].shape[0], prediction_decode[i, ...].shape[1]))
                    img_b = np.zeros((prediction_decode[i, ...].shape[0], prediction_decode[i, ...].shape[1]))
                    mask_disc = np.zeros((prediction_decode[i, ...].shape[0], prediction_decode[i, ...].shape[1]))
                    mask_cup = np.zeros((prediction_decode[i, ...].shape[0], prediction_decode[i, ...].shape[1]))
                    
                    mask_disc[prediction_decode[i,...]==1]=255
                    mask_disc[prediction_decode[i,...]==2]=255
                    mask_cup[prediction_decode[i,...]==2]=255
                    
                    img_r[prediction_decode[i,...]==1]=150
                    img_r = np.uint8(img_r)
                    edge = cv2.Canny(img_r, 30, 200)
                    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(img_r, contours, -1, (255, 0, 0), 3)
                    img_r[prediction_decode[i,...]==1]=0

                    img_b[prediction_decode[i,...]==2]=150
                    img_b = np.uint8(img_b)
                    edge = cv2.Canny(img_b, 30, 200)
                    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(img_b, contours, -1, (255, 0, 0), 3)
                    img_b[prediction_decode[i,...]==2]=0

                    img_b = remove_small_objects(img_b > 0, 50)
                    img_r = remove_small_objects(img_r > 0, 100)

                    img_ = np.concatenate((img_b[..., np.newaxis], img_g[..., np.newaxis], img_r[..., np.newaxis]), axis=2)
                    img_ = np.float32(img_) * 255

                    tmp = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
                    b, g, r = cv2.split(img_)
                    rgba = [b, g, r, alpha]
                    dst = cv2.merge(rgba, 4)

                    # cv2.imwrite(seg_results_small_path + img_name[i] + '.png', dst)
                    
                    img_ww = cv2.resize(dst, (int(ori_width[i]), int(ori_height[i])), interpolation=cv2.INTER_NEAREST)
                    # cv2.imwrite(seg_results_raw_path + img_name[i] + '.png', img_ww)

                pbar.update(imgs.shape[0])
    
    return img_ww, mask_disc, mask_cup

def disc_cup_analysis(disc_cup_: np.ndarray):
    optic_vertical_CDR, optic_vertical_disc, optic_vertical_cup = [], [], []
    optic_horizontal_CDR, optic_horizontal_disc,optic_horizontal_cup = [], [], []

    optic_centre_list = []

    # disc_cup_list = sorted(os.listdir(seg_path))
    # resolution_list = pd.read_csv('crop_info.csv')

    # for i in disc_cup_list:
    #     path_ = seg_path + i
    #     disc_cup_ = cv2.imread(path_)
    #     # disc_cup_ = cv2.resize(disc_cup_, (1024, 1024), interpolation = cv2.INTER_NEAREST)
    #     # resolution_scale = resolution_list['Scale_resolution'][resolution_list['Name'] == i].values[0]

    try:
        disc_ = disc_cup_[..., 2]
        cup_ = disc_cup_[..., 0]

        ## judgement the optic disc/cup segmentation
        disc_mask = measure.label(disc_)                       
        regions = measure.regionprops(disc_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[2:]:
                disc_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
        disc_[disc_mask != 0] = 255
        
        cup_mask = measure.label(cup_)                       
        regions = measure.regionprops(cup_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[2:]:
                cup_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
        cup_[cup_mask != 0] = 255

        disc_index = np.where(disc_ > 0)
        disc_index_width = disc_index[1]
        disc_index_height = disc_index[0]
        disc_horizontal_width = np.max(disc_index_width) - np.min(disc_index_width)
        disc_vertical_height = np.max(disc_index_height) - np.min(disc_index_height)
        
        cup_index = np.where(cup_ > 0)
        cup_index_width = cup_index[1]
        cup_index_height = cup_index[0]
        cup_horizontal_width = np.max(cup_index_width) - np.min(cup_index_width)
        cup_vertical_height = np.max(cup_index_height) - np.min(cup_index_height)

        cup_width_centre = np.mean(cup_index_width)
        cup_height_centre = np.mean(cup_index_height)

        if disc_horizontal_width < (disc_.shape[0] / 3) and disc_vertical_height < (disc_.shape[1] / 3) and cup_width_centre <= np.max(disc_index_width) and cup_width_centre >= np.min(disc_index_width) and cup_height_centre <= np.max(disc_index_height) and cup_height_centre >= np.min(disc_index_height) and cup_vertical_height < disc_vertical_height and cup_horizontal_width < disc_horizontal_width:
            whole_index = np.where(disc_cup_ > 0)
            whole_index_width = whole_index[1]
            whole_index_height = whole_index[0]

            horizontal_distance = np.absolute(np.mean(whole_index_height) - disc_cup_.shape[1] / 2)
            vertical_distance = np.absolute(np.mean(whole_index_width) - disc_cup_.shape[0] / 2)
            distance_ = np.sqrt(np.square(horizontal_distance) + np.square(vertical_distance))

            if (distance_/disc_cup_.shape[1]) > 0.1:
                # optic_centre_list.append(i)

                # optic_vertical_disc.append(disc_vertical_height * resolution_scale)
                # optic_horizontal_disc.append(disc_horizontal_width * resolution_scale)

                # optic_vertical_cup.append(cup_vertical_height * resolution_scale)
                # optic_horizontal_cup.append(cup_horizontal_width * resolution_scale)

                optic_vertical_CDR.append((cup_vertical_height / disc_vertical_height).round(4))
                optic_horizontal_CDR.append((cup_horizontal_width / disc_horizontal_width).round(4))

    except:
        pass
    pd_optic_centre = pd.DataFrame({'CDR_vertical': optic_vertical_CDR, 'CDR_horizontal': optic_horizontal_CDR})
    # pd_optic_centre.to_csv(seg_path + 'disc_cup_results.csv', index = None, encoding='utf8')

    return pd_optic_centre.to_dict()


def ISNT(mask_cup: np.ndarray, mask_disc: np.ndarray, eye: str):
    # ====== Measure vertical and horizontal diameters of the cup and disc ===== #
    horizontal_cup_diameters = []
    horizontal_cup_len = 0
    vertical_cup_diameters = []
    vertical_cup_len = 0
    for i, j in product(range(mask_cup.shape[0]), range(mask_cup.shape[1])):
        if j == mask_cup.shape[1] - 1:
            horizontal_cup_diameters.append(horizontal_cup_len)
            horizontal_cup_len = 0
            vertical_cup_diameters.append(vertical_cup_len)
            vertical_cup_len = 0
        elif mask_cup[i, j] == 255:
            horizontal_cup_len += 1
        elif mask_cup[j, i] == 255:
            vertical_cup_len += 1

    horizontal_cup_diameter = max(horizontal_cup_diameters)            
    vertical_cup_diameter = max(vertical_cup_diameters)
    # print(horizontal_cup_diameter, vertical_cup_diameter)

    horizontal_disc_diameters = []
    horizontal_disc_len = 0
    vertical_disc_diameters = []
    vertical_disc_len = 0
    for i, j in product(range(mask_disc.shape[0]), range(mask_disc.shape[1])):
        if j == mask_disc.shape[1] - 1:
            horizontal_disc_diameters.append(horizontal_disc_len)
            horizontal_disc_len = 0
            vertical_disc_diameters.append(vertical_disc_len)
            vertical_disc_len = 0
        elif mask_disc[i, j] == 255:
            horizontal_disc_len += 1
        elif mask_disc[j, i] == 255:
            vertical_disc_len += 1

    horizontal_disc_diameter = max(horizontal_disc_diameters)            
    vertical_disc_diameter = max(vertical_disc_diameters)
    # print(horizontal_disc_diameter, vertical_disc_diameter)

    # ===== Define coordinates of the cup and disc ===== #
    chx, dhy = 0, 0
    for cx in range(len(horizontal_cup_diameters)):
        if horizontal_cup_diameters[cx] == horizontal_cup_diameter:
            chx = cx
            break
    for cy in range(chx, len(horizontal_cup_diameters)):
        if horizontal_cup_diameters[cy] != horizontal_cup_diameter:
            chy = cy
            break
    # print(chx, chy)
    horizontal_cup_diameter_idx = chx + round((chy - chx) / 2)

    dhx, dhy = 0, 0
    for dx in range(len(horizontal_disc_diameters)):
        if horizontal_disc_diameters[dx] == horizontal_disc_diameter:
            dhx = dx
            break
    for dy in range(dhx, len(horizontal_disc_diameters)):
        if horizontal_disc_diameters[dy] != horizontal_disc_diameter:
            dhy = dy
            break
    # print(dhx, dhy)
    horizontal_disc_diameter_idx = dhx + round((dhy - chx) / 2)

    cvx, cvy = 0, 0
    for cx in range(len(vertical_cup_diameters)):
        if vertical_cup_diameters[cx] == vertical_cup_diameter:
            cvx = cx
            break
    for cy in range(cvx, len(vertical_cup_diameters)):
        if vertical_cup_diameters[cy] != vertical_cup_diameter:
            cvy = cy
            break
    # print(cvx, cvy)
    vertical_cup_diameter_idx = cvx + round((cvy - cvx) / 2)

    dvx, dvy = 0, 0
    for dx in range(len(vertical_disc_diameters)):
        if vertical_disc_diameters[dx] == vertical_disc_diameter:
            dvx = dx
            break
    for dy in range(dvx, len(vertical_disc_diameters)):
        if vertical_disc_diameters[dy] != vertical_disc_diameter:
            dvy = dy
            break
    # print(dvx, dvy)
    vertical_disc_diameter_idx = dvx + round((dvy - dvx) / 2)

    # ===== Calculate ISNT ===== #
    # ISNT_I
    cup_bound_i = 0
    for ci in reversed(range(mask_cup.shape[0])):
        if mask_cup[ci, vertical_cup_diameter_idx] == 0:
            mask_cup[ci, vertical_cup_diameter_idx] = 150
            cup_bound_i += 1
        elif mask_cup[ci, vertical_cup_diameter_idx] == 255:
            break
    # print(cup_bound_i)

    disc_bound_i = 0
    for di in reversed(range(mask_disc.shape[0])):
        if mask_disc[di, vertical_disc_diameter_idx] == 0:
            mask_disc[di, vertical_disc_diameter_idx] = 150
            disc_bound_i += 1
        elif mask_disc[di, vertical_disc_diameter_idx] == 255:
            break
    # print(disc_bound_i)

    isnt_i = cup_bound_i - disc_bound_i

    # ISNT_S
    cup_bound_s = 0
    for ci in range(mask_cup.shape[0]):
        if mask_cup[ci, vertical_cup_diameter_idx] == 0:
            mask_cup[ci, vertical_cup_diameter_idx] = 150
            cup_bound_s += 1
        elif mask_cup[ci, vertical_cup_diameter_idx] == 255:
            break
    # print(cup_bound_s)

    disc_bound_s = 0
    for di in range(mask_disc.shape[0]):
        if mask_disc[di, vertical_disc_diameter_idx] == 0:
            mask_disc[di, vertical_disc_diameter_idx] = 150
            disc_bound_s += 1
        elif mask_disc[di, vertical_disc_diameter_idx] == 255:
            break
    # print(disc_bound_s)

    isnt_s = cup_bound_s - disc_bound_s

    # ISNT_N
    cup_bound_n = 0
    for cj in reversed(range(mask_cup.shape[1])):
        if mask_cup[horizontal_cup_diameter_idx, cj] == 0:
            mask_cup[horizontal_cup_diameter_idx, cj] = 150
            cup_bound_n += 1
        elif mask_cup[horizontal_cup_diameter_idx, cj] == 255:
            break
    # print(cup_bound_n)

    disc_bound_n = 0
    for dj in reversed(range(mask_disc.shape[1])):
        if mask_disc[horizontal_disc_diameter_idx, dj] == 0:
            mask_disc[horizontal_disc_diameter_idx, dj] = 150
            disc_bound_n += 1
        elif mask_disc[horizontal_disc_diameter_idx, dj] == 255:
            break
    # print(disc_bound_n)

    isnt_n = cup_bound_n - disc_bound_n

    # ISNT_T
    cup_bound_t = 0
    for cj in range(mask_cup.shape[1]):
        if mask_cup[horizontal_cup_diameter_idx, cj] == 0:
            mask_cup[horizontal_cup_diameter_idx, cj] = 150
            cup_bound_t += 1
        elif mask_cup[horizontal_cup_diameter_idx, cj] == 255:
            break
    # print(cup_bound_t)

    disc_bound_t = 0
    for dj in range(mask_disc.shape[1]):
        if mask_disc[horizontal_disc_diameter_idx, dj] == 0:
            mask_disc[horizontal_disc_diameter_idx, dj] = 150
            disc_bound_t += 1
        elif mask_disc[horizontal_disc_diameter_idx, dj] == 255:
            break
    # print(disc_bound_t)

    isnt_t = cup_bound_t - disc_bound_t

    if eye == 'r':
        isnt_n, isnt_t = swap(isnt_n, isnt_t)

    pd_isnt = pd.DataFrame({'I': [isnt_i], 'S': [isnt_s], 'N': [isnt_n], 'T': [isnt_t]})

    return pd_isnt.to_dict()

#======================================================================#

