import os
import math
import pandas as pd
import csv
import json

from __init__ import find_nearest
from mappings import confmap2ra, labelmap2ra

from config import class_ids
from config import radar_configs, t_cl2rh

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')
range_grid_label = labelmap2ra(radar_configs, name='range')
angle_grid_label = labelmap2ra(radar_configs, name='angle')

def read_ra_labels_csv(seq_path):

    label_csv_name = os.path.join(seq_path, 'annotations/box/ramap_labels.csv')
    data = pd.read_csv(label_csv_name)
    n_row, n_col = data.shape
    obj_info_list = []
    cur_idx = -1
    for r in range(n_row):
        filename = data['filename'][r]
        frame_idx = int(filename.split('.')[0].split('_')[-1])
        if cur_idx == -1:
            obj_info = []
            cur_idx = frame_idx
        if frame_idx > cur_idx:
            obj_info_list.append(obj_info)
            obj_info = []
            cur_idx = frame_idx

        region_count = data['region_count'][r]
        region_id = data['region_id'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])

            cx = region_shape_attri['cx']
            cy = region_shape_attri['cy']
            if (cx >= radar_configs["ramap_asize_label"]
             or cy >= radar_configs["ramap_rsize_label"]):
                continue
            distance = range_grid_label[cy]
            angle = angle_grid_label[cx]
            if distance > radar_configs['rr_max'] or distance < radar_configs['rr_min']:
                continue
            if angle > radar_configs['ra_max'] or angle < radar_configs['ra_min']:
                continue
            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)
            try:
                class_str = region_attri['class']
            except:
                print("missing class at row %d" % r)
                continue
            try:
                class_id = class_ids[class_str]
            except:
                if class_str == '':
                    print("no class label provided!")
                    raise ValueError
                else:
                    class_id = -1000
                    print("Warning class not found! %s %010d" % (seq_path, frame_idx))
            obj_info.append([rng_idx, agl_idx, class_id])

    obj_info_list.append(obj_info)

    return obj_info_list


