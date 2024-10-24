import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

range_res = (RANGE_MAX-RANGE_MIN)/GRID_SIZE
doppler_res = (DOPPLER_MAX - DOPPLER_MIN)/GRID_SIZE
angle_res = (ANGLE_MAX - ANGLE_MIN)/GRID_SIZE

axes_grid = {
    "range": [range_res*i + RANGE_MIN for i in range(GRID_SIZE+1)],
    "doppler": [doppler_res*i + DOPPLER_MIN for i in range(GRID_SIZE+1)],
    "angle": [angle_res*i + ANGLE_MIN for i in range(GRID_SIZE+1)]
}

units = {"range": "m", "angle": "degree", "doppler": "m/s"}
classes = ["pedestrian", "cyclist", "car"]
views = ["range", "angle", "doppler"]
metrics = ["power_level"]
is_processed = False
show_frame = True
use_cfar = True
skip_frame = 45

for seq in seq_list[5:]:
    # seq = seq_list[1]
    print(seq)

    fig, ax = plt.subplots(1, 3)
    show_ra = None
    show_rd = None
    show_im = None
    txt_ra = None
    txt_rd = None

    # Load RA boxes
    with open(data_dir + seq + "/annotations/box/range_angle_light.json", "r") as f:
        ra_boxes = json.load(f)
    # Load RD boxes
    with open(data_dir + seq + "/annotations/box/range_doppler_light.json", "r") as f:
        rd_boxes = json.load(f)

    n_frame = len(os.listdir(data_dir+seq+"/range_angle_raw/"))
    for i in range(skip_frame, n_frame):
        # refresh patch
        ax[1].patches.clear()
        ax[2].patches.clear()

        box_idx = str(i).zfill(6)
        
        ra_frames, rd_frames, ra_masks, rd_masks = load_data_frame(data_dir + seq + "/", box_idx, is_processed)
        if (ra_frames is None or rd_frames is None or ra_masks is None or rd_masks is None):
            continue
        
        img_viz_frame = plt.imread(data_dir+seq+"/camera_images/"+box_idx+".jpg")

        print(box_idx)
        if (show_frame):
            if (use_cfar):
                ra_viz_frame = CFAR_2D(ra_frames, 5, 10, 1e-10)
                rd_viz_frame = CFAR_2D(rd_frames, 2, 5, 1e-10)
            else:
                ra_viz_frame = ra_frames.copy()
                rd_viz_frame = rd_frames.copy()
        else:
            ra_viz_frame = ra_masks.copy()
            rd_viz_frame = rd_masks.copy()

        max_ra, min_ra = np.max(ra_viz_frame), np.min(ra_viz_frame)
        max_rd, min_rd = np.max(rd_viz_frame), np.min(rd_viz_frame)

        # plot contour
        if (i == 0 or i == skip_frame):
            ax[1].set_title("Range-angle view", y=-0.2)
            ax[2].set_title("Range-Doppler view", y=-0.2)
            
            txt_ra = ax[1].text(-90.0, 50.5, "max: {}\nmin: {}".format(max_ra, min_ra))
            txt_rd = ax[2].text(-13.0, 50.5, "max: {}\nmin: {}".format(max_rd, min_rd))
            
            if (show_frame):
                if (is_processed):
                    show_ra = ax[1].pcolormesh(RA_x, RA_y, ra_viz_frame, vmin=40.0, vmax=90.0)
                    show_rd = ax[2].pcolormesh(RD_x, RD_y[::-1], rd_viz_frame, vmin=50.0, vmax=100.0)
                else:
                    show_ra = ax[1].pcolormesh(RA_x, RA_y, ra_viz_frame)
                    show_rd = ax[2].pcolormesh(RD_x, RD_y[::-1], rd_viz_frame)
            else:
                # show_ra = ax[1].imshow(RA_x, RA_y, ra_viz_frame, vmin=40.0, vmax=90.0)
                # show_rd = ax[2].imshow(RD_x, RD_y[::-1], rd_viz_frame, vmin=50.0, vmax=100.0)
                show_ra = ax[1].imshow(ra_viz_frame)
                show_rd = ax[2].imshow(rd_viz_frame)

            show_im = ax[0].imshow(img_viz_frame)

            fig.colorbar(show_ra, ax=ax[1])
            fig.colorbar(show_rd, ax=ax[2])
            ax[1].set_xlabel("Angle (degree)")
            ax[1].set_ylabel("Range (m)")
            ax[2].set_xlabel("Doppler (m/s)")
            ax[2].set_ylabel("Range (m)")

        else:
            if (show_frame):
                show_ra.set_array(ra_viz_frame.ravel())
                show_rd.set_array(rd_viz_frame.ravel())
            else:
                show_ra.set_data(ra_viz_frame)
                show_rd.set_data(rd_viz_frame)

            show_im.set_data(img_viz_frame)
            txt_ra.set_text("max: {}\nmin: {}".format(max_ra, min_ra))
            txt_rd.set_text("max: {}\nmin: {}".format(max_rd, min_rd))


        # Load bounding boxes and labels
        ra_box, rd_box, ra_label, rd_label = None, None, None, None
        if (box_idx in ra_boxes):
            ra_box = np.array(ra_boxes[box_idx]['boxes'])
            ra_label = np.array(ra_boxes[box_idx]['labels'])
            for box in ra_box:
                w, h = (box[3]-box[1])*ANGLE_RES*180.0/np.pi, (box[2]-box[0])*RANGE_RES
                p = (pixel_to_actual(box[1], "angle")*180.0/np.pi, pixel_to_actual(MAX_RANGE_PIX-box[2], "range"))
                rect = patches.Rectangle(p, w, h, linewidth=1, edgecolor='w', facecolor='none')
                ax[1].add_patch(rect)
        
        if (box_idx in rd_boxes):
            rd_box = np.array(rd_boxes[box_idx]['boxes'])
            rd_label = np.array(rd_boxes[box_idx]['labels'])
            for box in rd_box:
                w, h = (box[3]-box[1])*DOPPLER_RES, (box[2]-box[0])*RANGE_RES
                p = (pixel_to_actual(box[1], "doppler"), pixel_to_actual(box[0], "range"))
                rect = patches.Rectangle(p, w, h, linewidth=1, edgecolor='w', facecolor='none')
                ax[2].add_patch(rect)

        plt.draw()
        plt.pause(0.001)
    plt.close("all")