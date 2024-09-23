"""
This script is for converting box annotation from Carrada to RAMP-CNN ramap_labels.csv
"""

import os
import json
import pandas as pd

root_dir = "../Carrada/"
seqs = []
for seq in os.listdir(root_dir):
    if ("20" in seq):
        seqs.append(seq)
seqs.sort()

box_anno_dir = "annotations/box"
ra_box_anno_file = "range_angle_light.json"

for seq in seqs:
    ra_file_list = os.listdir(os.path.join(root_dir, seq, "range_angle_raw"))
    ra_file_list.sort()

    # Load json
    with open(os.path.join(root_dir, seq, box_anno_dir, ra_box_anno_file), "r") as f:
        ra_box_anno = json.load(f)

    class_name = ["background", "pedestrian", "cyclist", "car"]

    file_attributes = '{}'
    region_shape_attributes = {"name": "point", "cx": 0, "cy": 0}
    region_attributes = {"class": None}
    columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
               'region_shape_attributes', 'region_attributes']
    data = []

    for ra_file in ra_file_list:
        curr_idx = ra_file.replace(".npy", "")
        img_name = ra_file.replace(".npy", ".jpg")
        is_img_exist = os.path.exists(os.path.join(root_dir, seq, "camera_images", img_name))
        if (not is_img_exist):
            continue
        img_size = os.path.getsize(os.path.join(root_dir, seq, "camera_images", img_name))
        region_count = 0
        obj_info = []

        if (curr_idx in ra_box_anno):
            for i in range(len(ra_box_anno[curr_idx]["boxes"])):
                box = ra_box_anno[curr_idx]["boxes"][i]
                agl_idx, rng_idx = (box[0]+box[2]) // 2, (box[1] + box[3]) // 2
                type_ = class_name[ra_box_anno[curr_idx]["labels"][i]]
                obj_info.append([rng_idx, agl_idx, type_])
                region_count += 1

        for objId, obj in enumerate(obj_info):  # set up rows for different objs
            row = []
            row.append(img_name)
            row.append(img_size)
            row.append(file_attributes)
            row.append(region_count)
            row.append(objId)
            region_shape_attributes["cx"] = int(obj[1])
            region_shape_attributes["cy"] = int(obj[0])
            region_attributes["class"] = obj[2]
            row.append(json.dumps(region_shape_attributes))
            row.append(json.dumps(region_attributes))
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(root_dir, seq, box_anno_dir, "ramap_labels.csv"), index=None, header=True)
    print("\tSuccess!")


