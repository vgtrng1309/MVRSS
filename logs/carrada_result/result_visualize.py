import numpy as np
import json
import matplotlib.pyplot as plt
from utils_for_investigate import *
from sklearn import metrics

with open("result.json", "r") as f:
    result = json.load(f)

range_res = (RANGE_MAX-RANGE_MIN)/GRID_SIZE
doppler_res = (DOPPLER_MAX - DOPPLER_MIN)/GRID_SIZE
angle_res = (ANGLE_MAX - ANGLE_MIN)/GRID_SIZE

axes_grid = {
    "range": [range_res*i + RANGE_MIN for i in range(GRID_SIZE,-1,-1)],
    "doppler": [doppler_res*i + DOPPLER_MIN for i in range(GRID_SIZE+1)],
    "angle": [angle_res*i + ANGLE_MIN for i in range(GRID_SIZE+1)]
}

units = {"range": "m", "angle": "degree", "doppler": "m/s"}
classes = ["pedestrian", "cyclist", "car"]
views = ["range_angle", "range_doppler"]
metrics = ["iou", "centroid_distance", "FN", "FP", "miss_rate"]

fig_ind = 1
for cls in classes:
    for view in views:
        view_list = view.split("_")
        for metric in metrics:
            figure = plt.figure(fig_ind, figsize=(11, 5))
            figure.tight_layout()
            fig_ind += 1

            if (metric == "iou" or metric == "centroid_distance"):
                mats = [None, None]
                mats[0] = np.nanmean(result[cls][view][metric], axis=0)
                mats[1] = np.nanstd(result[cls][view][metric], axis=0)
                
                axes = figure.add_subplot(121)
                axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_yticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_xticklabels(axes_grid[view_list[1]])
                axes.set_yticklabels(axes_grid[view_list[0]])
                axes.set_xlabel(view_list[1] + "({})".format(units[view_list[1]]))
                axes.set_ylabel("range (m)")
                axes.set_title(cls+"_"+view+"_mean_"+metric)
                im = None
                if (metric == "centroid_distance"):
                    im = axes.imshow(mats[0], vmax=50)
                else:
                    im = axes.imshow(mats[0], vmin=0.0, vmax=1.0)
                figure.colorbar(im)

                axes = figure.add_subplot(122)
                axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_yticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_xticklabels(axes_grid[view_list[1]])
                axes.set_yticklabels(axes_grid[view_list[0]])
                axes.set_xlabel(view_list[1] + "({})".format(units[view_list[1]]))
                axes.set_ylabel("range (m)")
                axes.set_title(cls+"_"+view+"_std_"+metric)
                im = None
                if (metric == "centroid_distance"):
                    im = axes.imshow(mats[1], vmax=50)
                else:
                    im = axes.imshow(mats[1], vmin=0.0, vmax=1.0)
                figure.colorbar(im)
                figure.savefig("res_images/"+cls+"/"+cls+"_"+view+"_"+metric+".png", dpi=200)
            elif (metric == "miss_rate"):
                appearance = result[cls][view]["appearance"]
                mat = np.asarray(result[cls][view]["FN"]) / np.asarray(appearance)
                axes = figure.add_subplot(111)
                axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_yticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_xticklabels(axes_grid[view_list[1]])
                axes.set_yticklabels(axes_grid[view_list[0]])
                axes.set_xlabel(view_list[1] + "({})".format(units[view_list[1]]))
                axes.set_ylabel("range (m)")
                axes.set_title(cls+"_"+view+"_miss_rate")
                im = None
                im = axes.imshow(mat)
                figure.colorbar(im)
                figure.savefig("res_images/"+cls+"/"+cls+"_"+view+"_miss_rate.png", dpi=200)
            else:
                mat = result[cls][view][metric]
                axes = figure.add_subplot(111)
                axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_yticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_xticklabels(axes_grid[view_list[1]])
                axes.set_yticklabels(axes_grid[view_list[0]])
                axes.set_xlabel(view_list[1] + "({})".format(units[view_list[1]]))
                axes.set_ylabel("range (m)")
                axes.set_title(cls+"_"+view+"_"+metric)
                im = None
                im = axes.imshow(mat)
                figure.colorbar(im)
                figure.savefig("res_images/"+cls+"/"+cls+"_"+view+"_"+metric+"_miss_rate.png", dpi=200)
# plt.show()
