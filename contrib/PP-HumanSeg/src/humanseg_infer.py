import os, sys
import time
import random
import yaml
import codecs

import numpy as np
import cv2
import paddle
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))

import paddleseg.transforms as T
from paddleseg.core.infer import reverse_transform
from paddleseg.cvlibs import manager
from paddleseg.utils import TimeAverager

from optic_flow_process import optic_flow_process

import matplotlib.pyplot as plt


class DeployConfig:
    def __init__(self, path, vertical_screen):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

            [width, height] = self.dic['Deploy']['transforms'][0]['target_size']
            if vertical_screen and width > height:
                self.dic['Deploy']['transforms'][0][
                    'target_size'] = [height, width]

        self._transforms = self._load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def target_size(self):
        [width, height] = self.dic['Deploy']['transforms'][0]['target_size']
        return [width, height]

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return transforms


class Predictor:
    def __init__(self, args):
        self.args = args
        self.cfg = DeployConfig(args.config, args.vertical_screen)
        self.compose = T.Compose(self.cfg.transforms)

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_glog_info()
        if self.args.use_gpu:
            pred_cfg.enable_use_gpu(100, 0)

        self.predictor = create_predictor(pred_cfg)
        if args.use_optic_flow:
            self.disflow = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            width, height = self.cfg.target_size()
            self.prev_gray = np.zeros((height, width), np.uint8)
            self.prev_cfd = np.zeros((height, width), np.float32)
            self.is_first_frame = True
    
    def preproc_img(self, img: np.ndarray):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        data = self.compose({'img': img})
        input_data = np.array([data['img']])

        input_handle.reshape(input_data.shape)
        input_handle.copy_from_cpu(input_data)
        return data
    
    def predict_img(self, img: np.ndarray):
        '''
        get same size foreground human mask (alpha) from img
        '''
        data = self.preproc_img(img)  # dict
        self.predictor.run()
        return self.postproc_img(data)  # same size as img
    
    def postproc_img(self, data: dict):
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        pred_img = output_handle.copy_to_cpu()

        score_map = pred_img[0, 1, :, :]  # 1=foreground mask,0=inverse(background) mask
        score_map = score_map[np.newaxis, np.newaxis, ...]
        score_map = reverse_transform(
            paddle.to_tensor(score_map), data['trans_info'], mode='bilinear')  # (1, 1, H, W)
        alpha = score_map.numpy().squeeze()

        # print('score_map.shape, alpha.shape: ', score_map.shape, alpha.shape)
        return alpha

    def run(self, img, bg):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        data = self.compose({'img': img})
        input_data = np.array([data['img']])

        print('type(input_data): ', type(input_data), input_data.shape)
        print('type(input_handle): ', type(input_handle))
        print('input_names, ouput_names: ', input_names, output_names)

        input_handle.reshape(input_data.shape)
        input_handle.copy_from_cpu(input_data)

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output = output_handle.copy_to_cpu() # numpy.ndarray
        # print('type(output): ', type(output), output.shape)

        return self.postprocess(output, img, data, bg)

    def postprocess(self, pred_img, origin_img, data, bg):
        trans_info = data['trans_info']
        score_map = pred_img[0, 1, :, :]  # 1=foreground mask,0=inverse(background) mask
        # for i in range(pred_img.shape[1]):
        #     m = pred_img[0, i]
        #     m = (m * 255).astype("uint8")
        #     cv2.imwrite('pred_' + str(i) + '.jpg', m)

        # post process
        if self.args.use_post_process:
            mask_original = score_map.copy()
            mask_original = (mask_original * 255).astype("uint8")
            _, mask_thr = cv2.threshold(mask_original, 240, 1,
                                        cv2.THRESH_BINARY)
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
            mask_erode = cv2.erode(mask_thr, kernel_erode)
            mask_dilate = cv2.dilate(mask_erode, kernel_dilate)
            score_map *= mask_dilate

        # optical flow
        if self.args.use_optic_flow:
            score_map = 255 * score_map
            cur_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.resize(cur_gray,
                                  (pred_img.shape[-1], pred_img.shape[-2]))
            optflow_map = optic_flow_process(cur_gray, score_map, self.prev_gray, self.prev_cfd, \
                    self.disflow, self.is_first_frame)
            self.prev_gray = cur_gray.copy()
            self.prev_cfd = optflow_map.copy()
            self.is_first_frame = False
            score_map = optflow_map / 255.

        score_map = score_map[np.newaxis, np.newaxis, ...]
        score_map = reverse_transform(
            paddle.to_tensor(score_map), trans_info, mode='bilinear')
        alpha = np.transpose(score_map.numpy().squeeze(1), [1, 2, 0])
        print(score_map.shape, origin_img.shape, bg.shape, alpha.shape)

        h, w, _ = origin_img.shape
        bg = cv2.resize(bg, (w, h))
        if bg.ndim == 2:
            bg = bg[..., np.newaxis]

        out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)
        return out

def crop_mask(mask, mask_th=0.5, exp_ratio=0.1):
    h, w = mask.shape[:2]
    mask_ = (mask > mask_th).astype(np.uint8)
    bbox = cv2.boundingRect(mask_)
    bx, by, bw, bh = bbox
    offset_y = int(exp_ratio * bh)
    offset_x = int(exp_ratio * bw)
    keep_bottom_edge = False
    if by + bh > h - 5:  # foreground is at bottom edge of image
        bh = bh + offset_y
        keep_bottom_edge = True
    else:
        bh = bh + 2 * offset_y
    bw = bw + 2 * offset_x

    bx = max(0, bx - offset_x)
    by = max(0, by - offset_y)
    by2 = min(by + bh, h)
    bx2 = min(bx + bw, w)
    bh = by2 - by
    bw = bx2 - bx
    mask_bb = mask[by:by2, bx:bx2]
    return mask_bb, (bx, by, bw, bh), keep_bottom_edge

def merge_img(img: np.ndarray, mask: np.ndarray, img_bg: np.ndarray, mask_th=0.6, crop=False, resize_factor=1.0):
    h, w = img.shape[:2]
    hb, wb = img_bg.shape[:2]
    # print('img.shape, img_bg.shape: ', img.shape, img_bg.shape)

    # crop mask
    mask_bb, (bx, by, bw, bh), keep_bottom_edge = crop_mask(mask, mask_th)
    # print('mask.shape, mask_bb.shape, (bh, bw): ', mask.shape, mask_bb.shape, (bh, bw))
    img_bb = img[by:(by+bh), bx:(bx+bw)]

    # target inner region
    border_offset_ratio = 0.05
    border_offset_y = int(border_offset_ratio * hb)
    border_offset_x = int(border_offset_ratio * wb)
    if keep_bottom_edge:
        hb_bh = hb - border_offset_y
    else:
        hb_bh = hb - 2 * border_offset_y
    wb_bw = wb - 2 * border_offset_x

    alpha = np.zeros((hb, wb), dtype=np.float32)
    output = np.zeros_like(img_bg)
    # print('img, mask, mask_bb, img_bg.shape: ', img.shape, mask.shape, mask_bb.shape, img_bg.shape)

    if bh > hb or bw > wb:  # bbox larger than img_bg, need to resize mask
        resize_factor = min(hb_bh / bh, wb_bw / bw)  # minimum resize
    else:
        # print(img.shape, img_bg.shape, mask_bb.shape, (bh, bw), (hb_bh, wb_bw))
        resize_factor0 = min(hb_bh / bh, wb_bw / bw)  # minimum resize
        resize_factor = random.uniform(1.0, resize_factor0)
        resize_factor = min(resize_factor, resize_factor0)
        # print(f'resize_factor: [1.0, {resize_factor0:.2f}] {resize_factor:.2f}')
    mask_bb = cv2.resize(mask_bb, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    img_bb = cv2.resize(img_bb, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    
    mh, mw = mask_bb.shape[:2]
    start_x = max(0, wb//2 - mw//2)  # center
    start_y = max(0, hb//2 - mh//2)
    start_x_min = border_offset_x
    start_y_min = border_offset_y
    start_x_max = wb - border_offset_x - mw
    start_y_max = hb - border_offset_y - mh
    start_x = random.randint(start_x_min, start_x_max)
    start_y = random.randint(start_y_min, start_y_max)
    if keep_bottom_edge:
        start_y = hb - mh
    alpha[start_y:start_y+mh, start_x:start_x+mw] = mask_bb
    output[start_y:start_y+mh, start_x:start_x+mw] = img_bb

    alpha = alpha[..., np.newaxis]
    output = (alpha * output + (1 - alpha) * img_bg).astype(np.uint8)
    return output

def merge_img_with_box(img: np.ndarray, img_mask: np.ndarray, bboxes: list, img_bg: np.ndarray, mask_th=0.6, crop=False, resize_factor=1.0):
    h, w = img.shape[:2]
    hb, wb = img_bg.shape[:2]  # h_bg, w_bg

    mask_bb, (bx, by, bw, bh), keep_bottom_edge = crop_mask(img_mask, mask_th)
    img_bb = img[by:(by+bh), bx:(bx+bw)]
    # print(img.shape, img_mask.shape, mask_bb.shape, img_bb.shape)

    # target inner region
    border_offset_ratio = 0.05
    border_offset_y = int(border_offset_ratio * hb)
    border_offset_x = int(border_offset_ratio * wb)
    if keep_bottom_edge:
        hb_bh = hb - border_offset_y
    else:
        hb_bh = hb - 2 * border_offset_y
    wb_bw = wb - 2 * border_offset_x

    alpha = np.zeros((hb, wb), dtype=np.float32)
    output = np.zeros_like(img_bg)

    if bh > hb or bw > wb:  # bbox larger than img_bg, need to resize mask
        resize_factor = min(hb_bh / bh, wb_bw / bw)  # minimum resize
    else:
        # print(img.shape, img_bg.shape, mask_bb.shape, (bh, bw), (hb_bh, wb_bw))
        resize_factor0 = min(hb_bh / bh, wb_bw / bw)  # minimum resize
        resize_factor = random.uniform((1.0 + resize_factor0)/2, resize_factor0)
        resize_factor = min(resize_factor, resize_factor0)
        print(f'resize_factor: [1.0, {resize_factor0:.2f}] {resize_factor:.2f}')
    mask_bb = cv2.resize(mask_bb, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    img_bb = cv2.resize(img_bb, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    
    mh, mw = mask_bb.shape[:2]
    start_x = max(0, wb//2 - mw//2)  # center
    start_y = max(0, hb//2 - mh//2)
    start_x_min = border_offset_x
    start_y_min = border_offset_y
    start_x_max = max(wb - border_offset_x - mw, start_x_min)
    start_y_max = max(hb - border_offset_y - mh, start_y_min)
    start_x = random.randint(start_x_min, start_x_max)
    start_y = random.randint(start_y_min, start_y_max)
    if keep_bottom_edge:
        start_y = hb - mh
    alpha[start_y:start_y+mh, start_x:start_x+mw] = mask_bb
    output[start_y:start_y+mh, start_x:start_x+mw] = img_bb

    alpha = alpha[..., np.newaxis]
    output = (alpha * output + (1 - alpha) * img_bg).astype(np.uint8)

    # process bboxes
    x2, y2 = start_x, start_y
    bboxes2 = []
    
    # out_cpy = output.copy()
    # img_cpy = img.copy()
    # cv2.rectangle(out_cpy, (x2, y2), (x2 + mw, y2 + mh), (0, 255, 0), 2)
    for bbox in bboxes:
        class_id, px, py, pw, ph = bbox
        pw = int(pw * w)
        ph = int(ph * h)
        px = int(px * w) - pw // 2
        py = int(py * h) - ph // 2
        # cv2.rectangle(img_cpy, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
        # cv2.rectangle(img_cpy, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        px2, py2, pw2, ph2 = px - bx, py - by, pw, ph

        px3 = x2 + resize_factor* px2 
        py3 = y2 + resize_factor* py2
        pw3 = resize_factor * pw
        ph3 = resize_factor * ph

        # px3i, py3i, pw3i, ph3i = tuple(map(int, [px3, py3, pw3, ph3]))
        # cv2.rectangle(out_cpy, (px3i, py3i), (px3i + pw3i, py3i + ph3i), (0, 255, 0), 2)

        px3 = (2*px3 + pw3)/2 / wb
        py3 = (2*py3 + ph3)/2 / hb
        pw3 = pw3 / wb
        ph3 = ph3 / hb
        bboxes2.append((class_id, px3, py3, pw3, ph3))
    # plt.imshow(img_cpy, cmap="gray")
    # plt.imshow(out_cpy, cmap="gray")
    # plt.show()
    # plt.close()
    
    return output, bboxes2