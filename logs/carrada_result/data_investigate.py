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

seq_list = ["2019-09-16-13-13-01", "2019-09-16-13-18-33", "2020-02-28-12-13-54",
            "2020-02-28-12-23-30", "2020-02-28-13-08-51", "2020-02-28-13-14-35"]
result = load_result_template("./result_template.json")
# ra_FN = [[np.nan] * GRID_SIZE] * GRID_SIZE
# ra_FP = [[np.nan] * GRID_SIZE] * GRID_SIZE
# rd_FN = [[np.nan] * GRID_SIZE] * GRID_SIZE
# rd_FP = [[np.nan] * GRID_SIZE] * GRID_SIZE
false_rate = {"pedestrian": [create_2D_list(GRID_SIZE) for i in range(4)],
              "cyclist":    [create_2D_list(GRID_SIZE) for i in range(4)],
              "car":        [create_2D_list(GRID_SIZE) for i in range(4)]}

for seq in seq_list:
    # seq = seq_list[1]
    ra_boxes = None
    rd_boxes = None

    with open("box/" + seq + "/range_angle_light.json", "r") as f:
        ra_boxes = json.load(f)
    with open("box/" + seq + "/range_doppler_light.json", "r") as f:
        rd_boxes = json.load(f)

    n_frame = len(os.listdir("range_angle/" + seq)) // 2
    for i in range(4, 4+n_frame):
        box_idx = str(i).zfill(6)
        
        ra_masks, ra_outputs = load_frame("range_angle/" + seq, i)
        rd_masks, rd_outputs = load_frame("range_doppler/" + seq, i)
        if (ra_masks is None or ra_outputs is None or rd_masks is None or rd_outputs is None):
            continue

        # Flip range-doppler
        # rd_masks = np.flip(rd_masks, 0)
        # rd_masks = np.flip(rd_masks, 1)
        # rd_outputs = np.flip(rd_outputs, 0)
        # rd_outputs = np.flip(rd_outputs, 1)

        ra_viz_frame = ra_masks.copy()
        rd_viz_frame = rd_masks.copy()

        # Load bounding boxes and labels
        ra_box, rd_box, ra_label, rd_label = None, None, None, None
        if (box_idx in ra_boxes):
            ra_box = np.array(ra_boxes[box_idx]['boxes'])
            ra_label = np.array(ra_boxes[box_idx]['labels'])
            for box in ra_box:
                ra_viz_frame = cv2.rectangle(ra_viz_frame, (box[1], box[0]), (box[3], box[2]), (255, 255, 255), 1)
        
        if (box_idx in rd_boxes):
            rd_box = np.array(rd_boxes[box_idx]['boxes'])
            rd_label = np.array(rd_boxes[box_idx]['labels'])
            for box in rd_box:
                rd_viz_frame = cv2.rectangle(rd_viz_frame, (box[1], box[0]), (box[3], box[2]), (255, 255, 255), 1)

        if (ra_box is None):
            continue

        # object class iteration
        # k = 1 - pedestrian, 2 - cyclist, 3 - car
        class_name = ["pedestrian", "cyclist", "car"]
        for k in range(1,4):
            ra_iou = create_2D_list(GRID_SIZE)
            rd_iou = create_2D_list(GRID_SIZE)
            ra_distance = create_2D_list(GRID_SIZE)
            rd_distance = create_2D_list(GRID_SIZE)

            ra_mask = ra_masks[:,:,k-1]
            rd_mask = rd_masks[:,:,k-1]
            ra_output = ra_outputs[:,:,k-1]
            rd_output = rd_outputs[:,:,k-1]

            # If current mask doesn't have k class
            # TODO: consider cases when output have label that doesn't exist in gt (False Positive)
            if (k not in ra_label or k not in rd_label):
                ra_pixels = np.argwhere(ra_output)
                if (ra_pixels.shape[0] != 0):
                    ra_pixel = np.sum(ra_pixels, axis=0) // ra_pixels.shape[0]
                    r, c = get_bin_index(ra_pixel, True)
                    if (false_rate[class_name[k-1]][1][r][c] is np.nan):
                        false_rate[class_name[k-1]][1][r][c] = 1
                    else:
                        false_rate[class_name[k-1]][1][r][c] += 1

                rd_pixels = np.argwhere(rd_output)
                if (rd_pixels.shape[0] != 0):
                    rd_pixel = np.sum(rd_pixels, axis=0) // rd_pixels.shape[0]
                    r, c = get_bin_index(rd_pixel, False)
                    if (false_rate[class_name[k-1]][3][r][c] is np.nan):
                        false_rate[class_name[k-1]][3][r][c] = 1
                    else:
                        false_rate[class_name[k-1]][3][r][c] += 1

                continue

            # Count number unique label k in bboxes
            k_labels = np.argwhere(ra_label == k)
            k_ra_boxes = np.squeeze(ra_box[k_labels], axis=1)
            k_rd_boxes = np.squeeze(rd_box[k_labels], axis=1)

            # Perform clustering on the mask if have multiple boxes in the same class k
            
            if (k_labels.shape[0] > 1):
                ra_obj_masks, ra_obj_outputs, ra_box_centers = clustering(ra_mask, ra_output,
                                                                        k_ra_boxes)
                for ra_obj_mask, ra_obj_output, ra_box_center in zip(ra_obj_masks, \
                                                    ra_obj_outputs, ra_box_centers):
                    iou = get_IOU(ra_obj_mask, ra_obj_output)
                    distance = get_centroid_distance(ra_obj_mask, ra_obj_output)
                    r, c = get_bin_index(ra_box_center, is_range_angle=True)
                    # print(ra_iou[r][c], iou)
                    # False Negative
                    if (distance is np.nan):
                        if (false_rate[class_name[k-1]][0][r][c] is np.nan):
                            false_rate[class_name[k-1]][0][r][c] = 1
                        else:
                            false_rate[class_name[k-1]][0][r][c] += 1
                    else:
                        if (ra_iou[r][c] is np.nan or ra_distance[r][c] is np.nan):
                            ra_iou[r][c] = iou
                            ra_distance[r][c] = distance
                        else:
                            ra_iou[r][c] = np.max([ra_iou[r][c], iou])
                            ra_distance[r][c] = np.max([ra_distance[r][c], distance])

                rd_obj_masks, rd_obj_outputs, rd_box_centers = clustering(rd_mask, rd_output,
                                                                        k_rd_boxes)
                for rd_obj_mask, rd_obj_output, rd_box_center in zip(rd_obj_masks, \
                                                    rd_obj_outputs, rd_box_centers):
                    iou = get_IOU(rd_obj_mask, rd_obj_output)
                    distance = get_centroid_distance(rd_obj_mask, rd_obj_output)
                    r, c = get_bin_index(rd_box_center, is_range_angle=False)
                    # False Negative
                    if (distance is np.nan):
                        if (false_rate[class_name[k-1]][2][r][c] is np.nan):
                            false_rate[class_name[k-1]][2][r][c] = 1
                        else:
                            false_rate[class_name[k-1]][2][r][c] += 1
                    else:
                        if (rd_iou[r][c] is np.nan or rd_distance[r][c] is np.nan):
                            rd_iou[r][c] = iou
                            rd_distance[r][c] = distance
                        else:
                            rd_iou[r][c] = np.max([rd_iou[r][c], iou])
                            rd_distance[r][c] = np.max([rd_distance[r][c], distance])
            else:
                ra_box_center = k_ra_boxes[0]
                rd_box_center = k_rd_boxes[0]
                iou = get_IOU(ra_mask, ra_output)
                distance = get_centroid_distance(ra_mask, ra_output)
                r, c = get_bin_index(ra_box_center, is_range_angle=True)
                # False Negative
                if (distance is np.nan):
                    if (false_rate[class_name[k-1]][0][r][c] is np.nan):
                        false_rate[class_name[k-1]][0][r][c] = 1
                    else:
                        false_rate[class_name[k-1]][0][r][c] += 1
                else:
                    if (ra_iou[r][c] is np.nan or ra_distance[r][c] is np.nan):
                        ra_iou[r][c] = iou
                        ra_distance[r][c] = distance
                    else:
                        ra_iou[r][c] = np.max([ra_iou[r][c], iou])
                        ra_distance[r][c] = np.max([ra_distance[r][c], distance])

                iou = get_IOU(rd_mask, rd_output)
                distance = get_centroid_distance(rd_mask, rd_output)
                r, c = get_bin_index(rd_box_center, is_range_angle=False)
                # False Negative
                if (distance is np.nan):
                    if (false_rate[class_name[k-1]][2][r][c] is np.nan):
                        false_rate[class_name[k-1]][2][r][c] = 1
                    else:
                        false_rate[class_name[k-1]][2][r][c] += 1
                else:
                    if (rd_iou[r][c] is np.nan or rd_distance[r][c] is np.nan):
                        rd_iou[r][c] = iou
                        rd_distance[r][c] = distance
                    else:
                        rd_iou[r][c] = np.max([rd_iou[r][c], iou])
                        rd_distance[r][c] = np.max([rd_distance[r][c], distance])

            # print(ra_iou)
            result[class_name[k-1]]["range_angle"]["iou"].append(ra_iou)
            result[class_name[k-1]]["range_doppler"]["iou"].append(rd_iou)
            result[class_name[k-1]]["range_angle"]["centroid_distance"].append(ra_distance)
            result[class_name[k-1]]["range_doppler"]["centroid_distance"].append(rd_distance)

            # cv2.imshow("range_angle_box", k_ra_box_masks)
            # cv2.imshow("range_doppler_box", k_rd_box_masks)
            # cv2.imshow("range_angle_gt", np.hstack((ra_mask, np.full((256, 30), 255, np.uint8),
                                                    # ra_output)))
            # cv2.imshow("range_doppler_gt", rd_mask)
            # cv2.imshow("range_angle_output", ra_output)
            # cv2.imshow("range_doppler_output", rd_output)

        cv2.imshow("output_ra", ra_outputs)
        cv2.imshow("output_rd", rd_outputs)
        cv2.imshow("range_angle", ra_viz_frame)
        cv2.imshow("range_doppler", rd_viz_frame)
        k = cv2.waitKey(10)
        if (k == ord('q')):
            break

    # break

cv2.destroyAllWindows()
for cls in class_name:
    result[cls]["range_angle"]["FN"] = false_rate[cls][0]
    result[cls]["range_doppler"]["FN"] = false_rate[cls][2]
    result[cls]["range_angle"]["FP"] = false_rate[cls][1]
    result[cls]["range_doppler"]["FP"] = false_rate[cls][3]


with open("result.json", "w") as f:
    json.dump(result, f)
