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
metrices = ["iou", "centroid_distance", "FN", "FP"]

fig_ind = 1
for cls in classes:
    for view in views:
        view_list = view.split("_")
        for metrice in metrices:
            figure = plt.figure(fig_ind)
            fig_ind += 1

            if (metrice == "iou" or metrice == "centroid_distance"):
                mats = [None, None]
                mats[0] = np.nanmean(result[cls][view][metrice], axis=0)
                mats[1] = np.nanstd(result[cls][view][metrice], axis=0)
                
                axes = figure.add_subplot(121)
                axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_yticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_xticklabels(axes_grid[view_list[1]])
                axes.set_yticklabels(axes_grid[view_list[0]])
                axes.set_xlabel(view_list[1] + "({})".format(units[view_list[1]]))
                axes.set_ylabel("range (m)")
                axes.set_title(cls+" "+view+" mean "+metrice)
                im = None
                if (metrice == "centroid_distance"):
                    im = axes.imshow(mats[0], vmax=50, cmap=plt.cm.get_cmap('viridis').reversed())
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
                axes.set_title(cls+" "+view+" std "+metrice)
                im = None
                if (metrice == "centroid_distance"):
                    im = axes.imshow(mats[1], vmax=50, cmap=plt.cm.get_cmap('viridis').reversed())
                else:
                    im = axes.imshow(mats[1], vmin=0.0, vmax=1.0)
                figure.colorbar(im)
            else:
                mat = result[cls][view][metrice]
                axes = figure.add_subplot(111)
                axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_yticks(np.arange(-.5, GRID_SIZE, 1))
                axes.set_xticklabels(axes_grid[view_list[1]])
                axes.set_yticklabels(axes_grid[view_list[0]])
                axes.set_xlabel(view_list[1] + "({})".format(units[view_list[1]]))
                axes.set_ylabel("range (m)")
                axes.set_title(cls+" "+view+" "+metrice)
                im = None
                im = axes.imshow(mat)
                figure.colorbar(im)
plt.show()
