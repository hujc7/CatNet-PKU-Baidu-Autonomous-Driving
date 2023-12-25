import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from math import sin, cos
from scipy.optimize import minimize

import albumentations  # albumentations >= 1.0.0
from albumentations import pytorch as AT

from config import *

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    # TODO: Change all rotation to sin cos
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    
    return img

class ConfiguredUtils():
    def __init__(self, cfg: dict):
        # need the following params
        # SIDE_EXT
        # IMG_WIDTH
        # IMG_HEIGHT
        # MODEL_SCALE
        # DISTANCE_THRESH_CLEAR
        # OPTIMIZE_XY

        self.cfg = cfg

    def preprocess_image(self, img, flip=False, perspective_transform=None):
        if perspective_transform is not None:
            img = img[img.shape[0] // 2:]
            img = cv2.warpPerspective(img, perspective_transform, (img.shape[1], img.shape[0]))
        else:
            img = img[img.shape[0] // 2:]
            bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
            bg = bg[:, :img.shape[1] // self.cfg["SIDE_EXT"]]
            img = np.concatenate([bg, img, bg], 1)
        img = cv2.resize(img, (self.cfg["IMG_WIDTH"], self.cfg["IMG_HEIGHT"]))
        if flip:
            img = img[:,::-1]
        return (img / 255).astype('float32')

    def get_mask_and_regr(self, img, labels, flip=False, perspective_transform=None):
        mask = np.zeros([self.cfg["IMG_HEIGHT"] // self.cfg["MODEL_SCALE"], self.cfg["IMG_WIDTH"] // self.cfg["MODEL_SCALE"]], dtype='float32')
        regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        regr = np.zeros([self.cfg["IMG_HEIGHT"] // self.cfg["MODEL_SCALE"], self.cfg["IMG_WIDTH"] // self.cfg["MODEL_SCALE"], 7], dtype='float32')
        coords = str2coords(labels)
        xs, ys = get_img_coords(labels)
        for x, y, regr_dict in zip(xs, ys, coords):
            x, y = y, x
            if perspective_transform is not None:
                x = x - img.shape[0] // 2
                y, x = cv2.perspectiveTransform(np.array([[[y, x]]]).astype('float32'), perspective_transform)[0][0]
                x = x * self.cfg["IMG_HEIGHT"] / (img.shape[0] // 2) / self.cfg["MODEL_SCALE"]
                y = y * self.cfg["IMG_WIDTH"] / img.shape[1] / self.cfg["MODEL_SCALE"]
            else:
                x = (x - img.shape[0] // 2) * self.cfg["IMG_HEIGHT"] / (img.shape[0] // 2) / self.cfg["MODEL_SCALE"]
                y = (y + img.shape[1] // self.cfg["SIDE_EXT"]) * self.cfg["IMG_WIDTH"] / (img.shape[1] + img.shape[1] // self.cfg["SIDE_EXT"] * 2) / self.cfg["MODEL_SCALE"]
                
            x = np.round(x).astype('int')
            y = np.round(y).astype('int')
            if x >= 0 and x < self.cfg["IMG_HEIGHT"] // self.cfg["MODEL_SCALE"] and y >= 0 and y < self.cfg["IMG_WIDTH"] // self.cfg["MODEL_SCALE"]:
                mask[x, y] = 1
                regr_dict = _regr_preprocess(regr_dict, flip)
                regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
        if flip:
            mask = np.array(mask[:,::-1])
            regr = np.array(regr[:,::-1])
        return mask, regr

    def clear_duplicates(self, coords):
        for c1 in coords:
            xyz1 = np.array([c1['x'], c1['y'], c1['z']])
            for c2 in coords:
                xyz2 = np.array([c2['x'], c2['y'], c2['z']])
                distance = np.sqrt(((xyz1 - xyz2)**2).sum())
                if distance < self.cfg["DISTANCE_THRESH_CLEAR"]:
                    if c1['confidence'] < c2['confidence']:
                        c1['confidence'] = -1
        return [c for c in coords if c['confidence'] > 0]

    def extract_coords(self, prediction, flipped=False, channel_first=False):
        if channel_first:
            prediction = np.rollaxis(prediction, 0, 3)
        logits = prediction[:,:,0]
        regr_output = prediction[:,:,1:]
        points = np.argwhere(logits > 0)
        col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
        coords = []
        for r, c in points:
            regr_dict = dict(zip(col_names, regr_output[r, c, :]))
            coords.append(_regr_back(regr_dict))
            coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
            # This might have bug if deleted
            if self.cfg["OPTIMIZE_XY"]:
                coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
                        self.optimize_xy(r, c,
                                    coords[-1]['x'],
                                    coords[-1]['y'],
                                    coords[-1]['z'], flipped)
            # The function does not get called in training, and flip does not happen in validation
            # if flipped:
            #     coords[-1]['x'] = -coords[-1]['x']
        coords = self.clear_duplicates(coords)
        return coords

    # This is changed a lot in the PS 0.5 version
    def optimize_xy(self, r, c, x0, y0, z0, xzy_slope, flipped=False):
        def distance_fn(xyz):
            x, y, z = xyz
            xx = -x if flipped else x
            slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
            x, y = convert_3d_to_2d(x, y, z0)
            y, x = x, y
            x = (x - IMG_SHAPE[0] // 2) * self.cfg["IMG_HEIGHT"] / (IMG_SHAPE[0] // 2) / self.cfg["MODEL_SCALE"]
            y = (y + IMG_SHAPE[1] // self.cfg["SIDE_EXT"]) * self.cfg["IMG_WIDTH"] / (IMG_SHAPE[1] +  IMG_SHAPE[1] // self.cfg["SIDE_EXT"] * 2) / self.cfg["MODEL_SCALE"]
            # return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
            return (x-r)**2 + (y-c)**2
        
        res = minimize(distance_fn, [x0, y0, z0], method='Powell')
        x_new, y_new, z_new = res.x
        return x_new, y_new, z_new

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

def randAugment(N=2, p=1.0, color_aug:int = None, cut_aug:int = None):
    scale = np.linspace(0, 0.4, 100)
    translate = np.linspace(0, 0.4, 100)
    rot = np.linspace(0, 30, 100)
    shear_x = np.linspace(0, 20, 100)
    shear_y = np.linspace(0, 20, 100)
    contrast = np.linspace(0.0, 0.4, 100)
    bright = np.linspace(0.0, 0.4, 100)
    sat = np.linspace(0.0, 0.2, 100)
    hue = np.linspace(0.0, 0.2, 100)
    shar = np.linspace(0.0, 0.9, 100)
    blur = np.linspace(0, 0.2, 100)
    noise = np.linspace(0, 1, 100)
    cut = np.linspace(0.0, 0.8, 100)
    grip_drop_out_ratio = np.linspace(0.0, 0.5, 100)

    color_trans = None
    if color_aug is not None:
        color_trans = albumentations.SomeOf([
            albumentations.RandomContrast(limit=contrast[color_aug], p=p),
            albumentations.RandomBrightness(limit=bright[color_aug], p=p),
            albumentations.ColorJitter(brightness=0.0, contrast=0.0, saturation=sat[color_aug], hue=0.0, p=p),
            albumentations.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=hue[color_aug], p=p),
            albumentations.Sharpen(alpha=(0.0, shar[color_aug]), lightness=(0.5, 1.0), p=p),
            # albumentations.core.composition.PerChannel(
            #     albumentations.OneOf([
            #     albumentations.MotionBlur(p=0.5),
            #     albumentations.MedianBlur(blur_limit=3, p=1),
            #     albumentations.Blur(blur_limit=3, p=1), ]), p=blur[color_aug] * p
            # ),
            albumentations.GaussNoise(var_limit=(8.0 * noise[color_aug], 64.0 * noise[color_aug]), per_channel=True, p=p)
        ], N)
        
    cut_trans = None
    if cut_aug is not None:
        cut_trans = albumentations.OneOf([
            albumentations.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=1),
            albumentations.GridDropout(ratio=grip_drop_out_ratio[cut_aug], p=1),
            albumentations.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=1),
        ], p=cut[cut_aug])

    transforms = albumentations.Compose([x for x in [color_trans, cut_trans] if x is not None])

    return transforms

class CarDataset(Dataset, ConfiguredUtils):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, cfg, training=True, transform=None, perspective_transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training
        self.perspective_transform = perspective_transform
        ConfiguredUtils.__init__(self, cfg)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        # Augmentation
        flip = False
        # 50% of flip
        if self.training and self.cfg["FLIP_AUG"]:
            flip = np.random.randint(10) < 5
        
        # why rot axis?
        # Read image
        img0 = imread(img_name, True)
        img = self.preprocess_image(img0, flip=flip, perspective_transform=self.perspective_transform)
        # img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        mask, regr = self.get_mask_and_regr(img0, labels, flip=flip, perspective_transform=self.perspective_transform)
        regr = np.rollaxis(regr, 2, 0)

        # Apply transformations
        if self.transform is not None:
            img = self.transform(image = img)['image']
        return [img, mask, regr]