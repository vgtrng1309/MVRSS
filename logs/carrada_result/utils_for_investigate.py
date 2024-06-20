import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json

DOPPLER_RES = 0.41968030701528203
RANGE_RES = 0.1953125
ANGLE_RES = 0.01227184630308513
WORLD_RADAR = (0.0, 0.0)
MAX_RANGE_PIX = 255
MAX_ANGLE_PIX = 255
MAX_DOPPLE_PIX = 63

RANGE_MAX = 50
RANGE_MIN =  0
ANGLE_MAX = 90
ANGLE_MIN =-90
DOPPLER_MAX = 15
DOPPLER_MIN =-15

IMG_SIZE = (1232, 1028)
COLOR_CODE = np.array([[0, 0, 0],
                       [255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255]])
GRID_SIZE = 10

def load_frame(path, index):
    try:
        mask = cv2.imread(path+"/mask_"+str(index)+".png")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        output = cv2.imread(path+"/output_"+str(index)+".png")
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    except:
        return None, None
    return mask, output

def clustering(mask, output, boxes):
    box_center = np.array([np.array([(box[0]+box[2])/2, (box[1]+box[3])/2]) \
                            for box in boxes])
    pixels = np.argwhere(output)
    pixels_T = pixels.T
    pixels_set = list(zip(pixels_T[0], pixels_T[1]))
    dist = np.column_stack((np.linalg.norm(pixels - box_center[0], axis=1),
                        np.linalg.norm(pixels - box_center[1], axis=1)))
    pixels_cls = np.argmin(dist, axis=1)

    pixels = dict(zip(pixels_set, pixels_cls.T))
    outputs = np.zeros(shape=(box_center.shape[0], output.shape[0], output.shape[1]))
    masks = np.zeros(shape=(box_center.shape[0], mask.shape[0], mask.shape[1]))
    output_clone = output.copy()

    for i, box in enumerate(boxes):
        # Create mask from bbox gt
        k_box_mask = cv2.rectangle(np.zeros(mask.shape), 
                            (box[1], box[0]), (box[3], box[2]), 1, -1)
        
        # Use box mask on gt mask
        masks[i] = (k_box_mask * mask).astype(np.uint8)

        # Use box mask on output mask
        outputs[i] = (k_box_mask * output).astype(np.uint8)
    
        # Remove pixel in output mask from output_clone
        output_clone = output_clone - outputs[i]
        output_clone[output_clone==-255] = 0
    
    pixels_clone = np.argwhere(output_clone==255)
    for pix in pixels_clone:
        outputs[pixels[tuple(pix)]][pix[0], pix[1]] = 255

    # cv2.imshow("test frame", np.hstack((outputs[0], np.full((256, 30), 255, np.uint8),
    #                                     outputs[1])))
    return masks, outputs, box_center

def get_mask_centroid(bin_mask):
    pixels = np.argwhere(bin_mask)
    centroid = np.sum(pixels, axis=0) / pixels.shape[0]
    return centroid

def get_IOU(mask, output):
    bin_mask = (mask != 0) 
    bin_output = (output != 0)
    intercept_mask = np.bitwise_and(bin_mask, bin_output)
    intercept = np.sum(intercept_mask)
    union = np.sum(bin_mask) + np.sum(bin_output) - intercept
    # cv2.imshow("maskoutput", np.hstack((mask, np.full((256, 30), 255, np.uint8),
    #                                     output)))
    return intercept / union

def get_centroid_distance(mask, output):
    pixels_output = np.argwhere(output)
    if (pixels_output.shape[0] == 0):
        return np.nan
    output_centroid = np.sum(pixels_output, axis=0) / pixels_output.shape[0] 

    pixels_mask = np.argwhere(mask)
    if (pixels_mask.shape[0] == 0):
        return np.nan
    mask_centroid = np.sum(pixels_mask, axis=0) / pixels_mask.shape[0] 

    print(output_centroid, mask_centroid)
    return np.linalg.norm(output_centroid - mask_centroid)

def load_result_template(path="./result_template.json"):
    with open(path, "r") as f:
        template = json.load(f)
    return template

def get_bin_index(box_center, is_range_angle):
    range_pix_res = 256 / GRID_SIZE
    angle_pix_res = 256 / GRID_SIZE
    doppler_pix_res = 64 / GRID_SIZE
    i, j = -1, -1
    if (is_range_angle):
        i = box_center[0] // range_pix_res
        j = box_center[1] // angle_pix_res
    else:
        i = GRID_SIZE - 1 - box_center[0] // range_pix_res
        j = GRID_SIZE - 1 - box_center[1] // doppler_pix_res
    return int(i), int(j)

def create_2D_list(grid_size):
    return [[np.nan for i in range(grid_size)] for j in range(grid_size)]