import numpy as np
import json
import matplotlib.pyplot as plt
from utils_for_investigate import *
from sklearn import metrics

with open("data_stat.json", "r") as f:
    data_stat = json.load(f)

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

fig_ind = 1
for cls in classes:
    for view in views:
        for metric in metrics:
            figure = plt.figure(fig_ind, figsize=(11, 5))
            fig_ind += 1

            power_data = data_stat[cls][view]["power"]
            area_data = data_stat[cls][view]["area"]
            mean_power = [0 for i in range(GRID_SIZE)]  
            std_power = [0 for i in range(GRID_SIZE)]
            for i, grid in enumerate(power_data):
                if (len(grid) != 0):
                    mean_power[i] = np.mean(grid)
                    std_power[i] = np.std(grid)
            # print(mean_power)
            axes = figure.add_subplot(121)
            axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
            axes.set_xticklabels(axes_grid[view])
            axes.set_ylim(0, 80)
            axes.set_xlabel(view + " (" + units[view] + ")")
            axes.set_ylabel("log power mean")
            axes.set_title(cls+"_"+view+"_mean_log_power")
            axes.bar(np.arange(GRID_SIZE), mean_power, color = 'b', 
                    width = 1.0, edgecolor = 'black', 
                    label='log power')

            axes = figure.add_subplot(122)
            axes.set_xticks(np.arange(-.5, GRID_SIZE, 1))
            axes.set_xticklabels(axes_grid[view])
            axes.set_ylim(0, 80)
            axes.set_xlabel(view + " (" + units[view] + ")")
            axes.set_ylabel("log power std")
            axes.set_title(cls+"_"+view+"_std_log_power")
            axes.bar(np.arange(GRID_SIZE), std_power, color = 'b', 
                    width = 1.0, edgecolor = 'black', 
                    label='std')
            figure.savefig("data_stat_images/"+cls+"/"+cls+"_"+view+"_"+metric+".png")
plt.show()
