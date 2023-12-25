import cv2
import numpy as np
import torch 
from config import *
from dataset import str2coords, euler_to_Rot, imread, ConfiguredUtils

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        if any(img_cor_points[:, 2] < 0):
            print("point is behind camera")
            continue
        
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img

import gc
import matplotlib.pyplot as plt

def visualize_model(model, dataset, dataset_df, image_path_format, configured_utils: ConfiguredUtils, num_image=8, device='cuda'):
    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    for idx in range(num_image):
        img, mask, regr = dataset[idx]
        print(img.shape, mask.shape, regr.shape)
        
        output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
        coords_pred = configured_utils.extract_coords(output[0], channel_first=True)
        coords_true = configured_utils.extract_coords(np.concatenate([mask[None], regr], 0), channel_first=True)
        
        img = imread(image_path_format.format(dataset_df['ImageId'].iloc[idx]))
        fig, axes = plt.subplots(1, 2, figsize=(30,30))
        axes[0].set_title('Ground truth')
        axes[0].imshow(visualize(img, coords_true))
        axes[1].set_title('Prediction')
        axes[1].imshow(visualize(img, coords_pred))
        plt.show()