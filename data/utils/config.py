# directory settings
data_sets = {
    'root_dir': "./data/Automotive/",
    'dates': ['2019_04_09', '2019_05_28'],
    'cam_anno': [False, False],
}

train_sets = {
    'root_dir': "./data/Automotive/",
    'dates': ['2019_04_09'],
    'seqs': [
        ['2019_04_09_bms1000'],
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
    'cam_anno': [False],
}

valid_sets = {
    'root_dir': "./data/Automotive/",
    'dates': ['2019_05_28'],
    'seqs': [
        ['2019_05_28_pm2s012']
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
}

test_sets = {
    'root_dir': "./data/Automotive/",
    'dates': ['2019_05_28'],
    'seqs': [
        ['2019_05_28_pm2s012']
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
}

# class settings
n_class = 3
class_table = {
    0: 'pedestrian',
    1: 'cyclist',
    2: 'car',
}

class_ids = {
    'pedestrian': 0,
    'cyclist': 1,
    'car': 2,
    'noise': -1000,
}

confmap_sigmas = {
    'pedestrian': 15,
    'cyclist': 20,
    'car': 30,
}

confmap_sigmas_interval = {
    'pedestrian': [5, 15],
    'cyclist': [8, 20],
    'car': [10, 30],
    # 'van': 12,
    # 'truck': 20,
}

confmap_length = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    # 'van': 12,
    # 'truck': 20,
}

object_sizes = {
    'pedestrian': 0.5,
    'cyclist': 1.0,
    'car': 3.0,
}

# calibration
t_cl2cr = [0.35, 0, 0]
t_cl2rh = [0.11, -0.05, 0.06]
t_cl2rv = [0.21, -0.05, 0.06]

# parameter settings
camera_configs = {
    'image_width': 1232,
    'image_height': 1028,
    'frame_rate': 10,
    # 'image_folder': 'images_0',
    # 'image_folder': 'images_hist_0',
    'image_folder': 'images',
    'time_stamp_name': 'timestamps.txt',
    # 'time_stamp_name': 'timestamps_0.txt',
    'frame_expo': 0,
    # 'frame_expo': 40,
    'start_time_name': 'start_time.txt',
}

Fc = 77*1e9         # Carrier frequency
B = 4*1e9           # Sweep Bandwidth
MAX_RANGE = 50      # Max range
RANGE_RES = 0.2     # Range resolution
N_CHIRPS = 64       # Number of chirps per frame
N_SAMPLES = 256     # Number of samples per frame

radar_configs = {
    'ramap_rsize': 256,             # RAMap range size
    'ramap_asize': 256,             # RAMap angle size
    'ramap_vsize': 64,             # RAMap angle size
    'frame_rate': 10,
    'crop_num': 0,                  # crop some indices in range domain
    'n_chirps': 64,                # number of chirps in one frame
    'sample_freq': 18.38417777e6,
    'sweep_slope': 55.1525333e12,
    'data_type': 'RISEP',           # 'RI': real + imaginary, 'AP': amplitude + phase
    'ramap_rsize_label': 250,       # TODO: to be updated. rsize - crop_num*2
    'ramap_asize_label': 249,       # TODO: to be updated. asize - crop_num*2 - 1
    'ra_min_label': -60,            # min radar angle
    'ra_max_label': 60,             # max radar angle
    'rr_min': 0.0,                  # min radar range (fixed)
    'rr_max': 50.0,                 # max radar range (fixed)
    'ra_min': -90,                  # min radar angle (fixed)
    'ra_max': 90,                   # max radar angle (fixed)
    'ramap_folder': 'WIN_HEATMAP',
}

semi_loss_err_reg = {
    # index unit
    'level1': 30,
    'level2': 60,
    'level3': 80,
}
# correct error region for level 1
err_cor_reg_l1 = {
    'top': 3,
    'bot': 3,
}
# correct error region for level 2
err_cor_reg_l2 = {
    'top': 3,
    'bot': 25,
}
# correct error region for level 3
err_cor_reg_l3 = {
    'top': 3,
    'bot': 35,
}

# for the old data
mean1 = -5.052344347731883e-05
std1 = 0.029227407344111892
# for the new data
mean2 = -5.087775336050677e-05
std2 = 0.03159186371634542

# for the new data
mean1_rv = 0.038192792357185736
std1_rv = 0.16754211919064926
# for the new data
mean2_rv = 0.05788228918531949
std2_rv = 0.18304037677587492

# for the new data
mean1_va = 0.06261547148605366
std1_va = 0.08709724872133341
# for the new data
mean2_va = 0.09291138048788067
std2_va = 0.11100713809079792
