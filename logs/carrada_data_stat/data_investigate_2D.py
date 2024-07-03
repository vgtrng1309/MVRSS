import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json
from sklearn.cluster import DBSCAN
from utils_for_investigate import *

# Create grid
step = 1
R_axis = np.linspace(0, 255, 256)
A_axis = np.linspace(0, 255, 256)
D_axis = np.linspace(0, 63, 64)

R_axis = (MAX_RANGE_PIX - R_axis) * RANGE_RES
A_axis = (A_axis - MAX_ANGLE_PIX // 2) * ANGLE_RES * 180.0 / np.pi
D_axis = (D_axis - MAX_DOPPLE_PIX // 2) * DOPPLER_RES
RA_x, RA_y = np.meshgrid(A_axis, R_axis) 
RD_x, RD_y = np.meshgrid(D_axis, R_axis)


data_dir = "../../data/Carrada/"
seq_list = []
for seq in os.listdir(data_dir):
    if ("20" in seq):
        seq_list.append(seq)
seq_list.sort()
grid_size = GRID_SIZE

data_template = load_result_template("./data_2D_template.json")
power_logs = {"pedestrian": [[[[] for j in range(grid_size)] for k in range(grid_size)] for i in range(2)],
              "cyclist"   : [[[[] for j in range(grid_size)] for k in range(grid_size)] for i in range(2)],
              "car"       : [[[[] for j in range(grid_size)] for k in range(grid_size)] for i in range(2)]}
areas_logs = {"pedestrian": [[[[] for j in range(grid_size)] for k in range(grid_size)] for i in range(2)],
              "cyclist"   : [[[[] for j in range(grid_size)] for k in range(grid_size)] for i in range(2)],
              "car"       : [[[[] for j in range(grid_size)] for k in range(grid_size)] for i in range(2)]}

is_processed = False

class_name = ["pedestrian", "cyclist", "car"]
for seq in seq_list:
    # seq = seq_list[1]
    print(seq)

    # Load RA boxes
    with open(data_dir + seq + "/annotations/box/range_angle_light.json", "r") as f:
        ra_boxes = json.load(f)
    # Load RD boxes
    with open(data_dir + seq + "/annotations/box/range_doppler_light.json", "r") as f:
        rd_boxes = json.load(f)

    n_frame = len(os.listdir(data_dir+seq+"/range_angle_raw/"))
    for i in range(n_frame):
        box_idx = str(i).zfill(6)
        
        ra_frames, rd_frames, ra_masks, rd_masks = load_data_frame(data_dir + seq + "/", box_idx, is_processed)
        if (ra_frames is None or rd_frames is None or ra_masks is None or rd_masks is None):
            continue

        # object class iteration
        # k = 1 - pedestrian, 2 - cyclist, 3 - car
        for k in range(1,4):
            ra_mask = (ra_masks==k)
            rd_mask = (rd_masks==k)

            # Load bounding boxes and labels
            ra_box, rd_box, ra_label, rd_label = None, None, None, None
            if (box_idx in ra_boxes):
                ra_box = np.array(ra_boxes[box_idx]['boxes'])
                ra_label = np.array(ra_boxes[box_idx]['labels'])
            
            if (box_idx in rd_boxes):
                rd_box = np.array(rd_boxes[box_idx]['boxes'])
                rd_label = np.array(rd_boxes[box_idx]['labels'])

            if (ra_box is None or k not in ra_label):
                continue
            # Count number unique label k in bboxes
            k_labels = np.argwhere(ra_label == k)
            k_ra_boxes = np.squeeze(ra_box[k_labels], axis=1)
            k_rd_boxes = np.squeeze(rd_box[k_labels], axis=1)
            # print(class_name[k-1], end=" ")
            for k_ra_box, k_rd_box in zip(k_ra_boxes, k_rd_boxes):
                ra_box_center = np.array([(k_ra_box[0] + k_ra_box[2])//2,
                                          (k_ra_box[1] + k_ra_box[3])//2])
                rd_box_center = np.array([(k_rd_box[0] + k_rd_box[2])//2,
                                          (k_rd_box[1] + k_rd_box[3])//2])
                ra_r, ra_a = get_bin_index(ra_box_center, is_range_angle=True, grid_size=grid_size)
                rd_r, rd_d = get_bin_index(rd_box_center, is_range_angle=False, grid_size=grid_size)
                # print(r, a, d, end=" ")
                # boxes mask
                ra_box_mask = np.array(cv2.rectangle(np.zeros(ra_mask.shape), 
                                        (k_ra_box[1], k_ra_box[0]), 
                                        (k_ra_box[3], k_ra_box[2]), 1, -1))
                rd_box_mask = np.array(cv2.rectangle(np.zeros(rd_mask.shape), 
                                        (k_rd_box[1], k_rd_box[0]), 
                                        (k_rd_box[3], k_rd_box[2]), 1, -1))
                
                # ground truth mask
                ra_gt_mask = (ra_box_mask * ra_mask).astype(np.float64)
                rd_gt_mask = (rd_box_mask * rd_mask).astype(np.float64)

                # masked power frame
                ra_masked_frame = ra_gt_mask * ra_frames
                ra_area = np.sum(ra_gt_mask)
                ra_mean_power = np.sum(ra_masked_frame) / ra_area
                rd_masked_frame = rd_gt_mask * rd_frames
                rd_area = np.sum(rd_gt_mask)
                rd_mean_power = np.sum(rd_masked_frame) / rd_area

                # write result
                power_logs[class_name[k-1]][0][ra_r][ra_a].append(ra_mean_power)
                power_logs[class_name[k-1]][1][rd_r][rd_d].append(rd_mean_power)

                areas_logs[class_name[k-1]][0][ra_r][ra_a].append(ra_area)
                areas_logs[class_name[k-1]][1][rd_r][rd_d].append(rd_area)
            # print()
        # k = cv2.waitKey(1)
        # if (k == ord('q')):
        #     break

    # break

cv2.destroyAllWindows()
for cls in class_name:
    data_template[cls]["range_angle"]["power"] = power_logs[cls][0]
    data_template[cls]["range_doppler"]["power"] = power_logs[cls][1]
    data_template[cls]["range_angle"]["area"] = areas_logs[cls][0]
    data_template[cls]["range_doppler"]["area"] = areas_logs[cls][1]

with open("data_stat_2D.json", "w") as f:
    json.dump(data_template, f)
