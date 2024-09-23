"""
This script is for generating confidence map from Carrada range_angle frame to fit RAMP_CNN format
"""
import os
import numpy as np
import pickle
import argparse

from confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from config import train_sets, test_sets, valid_sets
from config import n_class, radar_configs

from read_annotations import read_ra_labels_csv

def prepare_data():
    root_dir = "../Carrada/"
    seqs = []
    for seq in os.listdir(root_dir):
        if "20" in seq:
            seqs.append(seq)
    seqs.sort()
    for seq in seqs[:1]:
        # seq = "2020-02-28-12-12-16"
        detail_list = [[], 0]
        confmap_list = [[], []]

        # use labelled RAMap
        seq_path = os.path.join(root_dir, seq)
        try:
            obj_info_list = read_ra_labels_csv(seq_path)
        except Exception as e:
            print("Load sequence %s failed!" % seq_path)
            print(e)
            continue

        n_data = len(obj_info_list)
        # create paths for data
        for fid in range(n_data):
            if radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                # path = os.path.join(root_dir, seq, chirp_folder_name, "%04d", "%06d.npy")
                path = os.path.join(root_dir, seq, "range_angle_raw/%06d.npy")
            else:
                raise ValueError
            detail_list[0].append(path)
        # print(detail_list)

        print(obj_info_list)
        for obj_info in obj_info_list:
            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                  dtype=float)
            confmap_gt[-1, :, :] = 1.0
            if len(obj_info) != 0:
                confmap_gt = generate_confmap(obj_info)
                confmap_gt = normalize_confmap(confmap_gt)
                confmap_gt = add_noise_channel(confmap_gt)
            assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            confmap_list[0].append(confmap_gt)
            confmap_list[1].append(obj_info)
            # end objects loop

        confmap_dir = os.path.join(root_dir, seq, 'confmaps_gt')
        detail_dir = os.path.join(root_dir, seq, 'data_details')

        confmap_list[0] = np.array(confmap_list[0])
        dir2 = os.path.join(confmap_dir, "train")
        dir3 = os.path.join(detail_dir, "train")
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        if not os.path.exists(dir3):
            os.makedirs(dir3)

        # save pkl files
        # pickle.dump(confmap_list, open(os.path.join(dir2, seq + '.pkl'), 'wb'))
        # save pkl files
        # pickle.dump(detail_list, open(os.path.join(dir3, seq + '.pkl'), 'wb'))

        # end frames loop
    # end seqs loop

if __name__ == "__main__":
    """
    Example:
        python prepare_data.py -m train -dd './data/'
    """
    prepare_data()