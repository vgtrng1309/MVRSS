import os
import torch
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

from random import randint, random
from mappings import confmap2ra
from __init__ import find_nearest, convert_data_2Dmat_to_histogram
from config import radar_configs


Max_trans_agl = 20
Max_trans_rng = 40
range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle') # middle number index 63, 64


def resamp_shiftrange(data, shift_range):
    assert len(data.shape) == 5
    data_new = torch.zeros_like(data)
    if shift_range > 0:
        for ir in range(data.shape[3]):
            data_new[:, :, :, ir, :] = resample_range(data[:, :, :, ir, :], ir, shift_range)
    else:
        for ir in range(data.shape[3]):
            data_new[:, :, :, ir, :] = resample_range(data[:, :, :, ir, :], ir - shift_range, shift_range)
    return data_new


def resample_range(data, rangeid, shift_range):
    # rangeid, shift_range can all be id
    # the format of data [batch, C=2, window, angle]
    assert len(data.shape) == 4
    Is_Tensor = False
    if torch.is_tensor(data):
        Is_Tensor = True
        # convert to numpy
        data = data.cpu().detach().numpy()

    # rangeid, shift_range can all be id
    angle_len = len(angle_grid)
    data_new = np.zeros_like(data)
    interval = range_grid[rangeid] * abs(angle_grid[angle_len//2:])
    interval_new = range_grid[rangeid + shift_range] * abs(angle_grid[angle_len//2:])
    start_id = angle_len//2
    # move upwards
    if shift_range > 0:
        for id in range(angle_len//2, angle_len):
            need_id, _ = find_nearest(interval, interval_new[id - angle_len//2])
            start_id = min(start_id, need_id + angle_len//2)
            data_new[:, :, :, id] = np.mean(data[:, :, :, start_id:need_id + 1 + angle_len//2], axis=-1)
            data_new[:, :, :, angle_len-1-id] = np.mean(data[:, :, :, angle_len//2 - need_id - 1:angle_len - start_id],
                                                        axis=-1)
            start_id = need_id + 1 + angle_len//2
            if interval_new[id - angle_len//2] >= interval[-1]:
                break
     # move downwards
    else:
        for id in range(angle_len//2, angle_len):
            need_id, _ = find_nearest(interval_new, interval[id - angle_len//2])
            start_id = min(start_id, need_id + angle_len//2 + 1)
            if id == angle_len//2:
                # interpolate the angle between index 63~64
                data_new[:, :, :, angle_len//2-need_id-1:need_id+angle_len//2+1] = \
                    np.linspace(data[:, :, :, angle_len//2-1], data[:, :, :, angle_len//2], num=2*need_id+2, axis=-1)
                start_id = need_id + 1 + angle_len//2
            else:
                data_new[:, :, :, start_id-1:need_id+angle_len//2+1] = \
                    np.linspace(data[:, :, :, id-1], data[:, :, :, id], num=need_id+angle_len//2-start_id+2, axis=-1)
                data_new[:, :, :, angle_len//2-need_id-1:angle_len-start_id+1] = \
                    np.linspace(data[:, :, :, angle_len-id-1], data[:, :, :, angle_len-id],
                                num=need_id+angle_len//2-start_id+2, axis=-1)
                start_id = need_id + 1 + angle_len//2

            if interval[id - angle_len//2] >= interval_new[-1]:
                break
    if Is_Tensor:
        data_new = torch.from_numpy(data_new)

    return data_new


def resample(data, interval_cur, interval_new):
    Is_Tensor = False
    if torch.is_tensor(data):
        Is_Tensor = True
        # convert to numpy
        data = data.cpu().detach().numpy()
    data_new = np.zeros_like(data)
    interv_len = len(interval_cur)
    max_interv = max(abs(interval_cur))
    start_id_new = 0  # for interval_new
    start_id_cur = 0  # for interval

    for i in range(interv_len):
        if i < start_id_new:
            continue
        # the last element
        if i == interv_len - 1:
            data_new[:, :, :, :, i] = np.mean(data[:, :, :, :, start_id_cur:], axis=-1)
            break

        len_cell_new = interval_new[i+1] - interval_new[i]
        len_cell_cur = interval_cur[start_id_cur+1] - interval_cur[start_id_cur]
        # deal with negative number: 90 degrees, and -90 degrees
        if len_cell_cur < 0:
            len_cell_cur = len_cell_cur + max_interv * 2

        if len_cell_new < len_cell_cur:
            # the interval of new is less than the old interval
            # so we need to find interpolate the interval new
            idn, _ = find_nearest(interval_new - interval_new[start_id_new], len_cell_cur)
            # interpolate the cell between i and idn
            data_new[:, :, :, :, i:idn+1] = np.linspace(data[:, :, :, :, start_id_cur], data[:, :, :, :, start_id_cur+1],
                                                     num=idn+1-i, axis=-1)
            start_id_new = min(idn + 1, interv_len - 1)
            start_id_cur = min(start_id_cur + 1, interv_len - 1)
        else:
            # the interval of new is larger than the old interval
            # so we need to sum multiple cells in old interval
            idn, _ = find_nearest(interval_cur[start_id_cur:] - interval_cur[start_id_cur], len_cell_new)
            # sum the cell between start_id_cur and start_id_cur+idn to cell i
            data_new[:, :, :, :, i] = np.mean(data[:, :, :, :, start_id_cur:start_id_cur+idn+1], axis=-1)

            start_id_cur = min(start_id_cur + idn + 1, interv_len - 1)
            start_id_new = min(start_id_new + 1, interv_len - 1)

    if Is_Tensor:
        data_new = torch.from_numpy(data_new)

    return data_new


def resamp_shiftangle(data, shift_angle, axis=-1):
    # rangeid, shift_range can all be id
    # the format of data [batch, C=2, window, range, angle]
    assert len(data.shape) == 5
    interval_new = angle_grid
    interval = np.roll(interval_new, shift_angle, axis=0)
    if axis == -1 or axis == 4:
        data_new = resample(data, interval, interval_new)
    else:
        data_new = resample(torch.transpose(data, axis, -1), interval, interval_new)
        data_new = torch.transpose(data_new, axis, -1)

    return data_new


def Flip(data, data_va, confmap):
    # flip the angle dimension
    shape = data.shape
    assert len(shape) == 5
    data = torch.flip(data, [4])
    confmap = torch.flip(confmap, [4])
    if data_va is not None:
        data_va = torch.flip(data_va, [3])
        return data, data_va, confmap
    else:
        return data, None, confmap


def transition_angle(data, data_va, confmap=None, trans_angle=None):
    # shift_angle > 0, move rightward
    # shift_range < 0, move leftward
    if trans_angle is None:
        shift_angle = randint(-Max_trans_agl, Max_trans_agl)
    else:
        shift_angle = trans_angle
    shape = data.shape
    assert len(shape) == 5
    if data_va is not None:
        if shift_angle != 0:
            data_new = torch.roll(data, shift_angle, 4)
            # data_new = resamp_shiftangle(data_new, shift_angle, axis=4)
            data_va_new = torch.roll(data_va, shift_angle, 3)
            # data_va_new = resamp_shiftangle(data_va_new, shift_angle, axis=3)
            if (confmap is not None):
                confmap_new = torch.roll(confmap, shift_angle, 4)
                # confmap_new = resamp_shiftangle(confmap_new, shift_angle, axis=4)
                return data_new, data_va_new, confmap_new
            else:
                return data_new, data_va_new, None
        else:
            if (confmap is not None):
                return data, data_va, confmap
            else:
                return data, data_va, None
    else:
        if shift_angle != 0:
            data_new = torch.roll(data, shift_angle, 4)
            # data_new = resamp_shiftangle(data_new, shift_angle, axis=4)
            if (confmap is not None):
                confmap_new = torch.roll(confmap, shift_angle, 4)
                # confmap_new = resamp_shiftangle(confmap_new, shift_angle, axis=4)
                return data_new, None, confmap_new
            else:
                return data_new, None, None
        else:
            if (confmap is not None):
                return data, None, confmap
            else:
                return data, None, None

def interpolation(data, size=None):
    num_noise_cand = 100
    shape = data.shape
    # print(shape)
    assert len(shape) == 5
    if shape[1] == 2:
        data1 = torch.flatten(data[0, 0, 0:5, :, :])
        data2 = torch.flatten(data[0, 1, 0:5, :, :])
        data_amp = data1 ** 2 + data2 ** 2
        _, indices = torch.sort(data_amp)
        noise_cand1 = np.zeros(num_noise_cand)
        noise_cand2 = np.zeros(num_noise_cand)
        for i, index in enumerate(indices[0:num_noise_cand]):
            noise_cand1[i] = data1[index]
            noise_cand2[i] = data2[index]

        if size is not None:
            need_size = shape[0] * shape[2] * size[0] * size[1]
            noise1 = np.zeros(need_size)
            noise2 = np.zeros(need_size)
            for i in range(need_size):
                noise_id = randint(0, num_noise_cand-1)
                noise1[i] = noise_cand1[noise_id]
                noise2[i] = noise_cand2[noise_id]

            noise1 = np.reshape(noise1, (shape[0], 1, shape[2], size[0], size[1]))
            noise2 = np.reshape(noise2, (shape[0], 1, shape[2], size[0], size[1]))
            noise = np.concatenate((noise1, noise2), axis=1)
        else:
            zero_inds = (data[0, 0, 0, :, :] == 0).nonzero()
            # print(zero_inds)
            need_size = shape[0] * shape[2]
            noise1 = np.zeros(need_size)
            noise2 = np.zeros(need_size)

            for ind in zero_inds:
                for i in range(need_size):
                    noise_id = randint(0, num_noise_cand - 1)
                    noise1[i] = noise_cand1[noise_id]
                    noise2[i] = noise_cand2[noise_id]
                noise1 = np.reshape(noise1, (shape[0], 1, shape[2]))
                noise2 = np.reshape(noise2, (shape[0], 1, shape[2]))
                noise = np.concatenate((noise1, noise2), axis=1)
                data[:, :, :, ind[0], ind[1]] = torch.from_numpy(noise)

    elif shape[1] == 1:
        print("test", data.shape)
        data1 = torch.flatten(data[0, 0, 0:5, :, :])
        _, indices = torch.sort(data1)
        noise_cand = np.zeros(num_noise_cand)
        for i, index in enumerate(indices[0:num_noise_cand]):
            noise_cand[i] = data1[index]

        if size is not None:
            need_size = shape[0] * shape[2] * size[0] * size[1]
            noise = np.zeros(need_size)
            for i in range(need_size):
                noise[i] = noise_cand[randint(0, num_noise_cand-1)]
            noise = np.reshape(noise, (shape[0], 1, shape[2], size[0], size[1]))
        else:
            zero_inds = (data[0, 0, 0, :, :] == 0).nonzero()
            need_size = shape[0] * shape[2]
            noise = np.zeros(need_size)
            for ind in zero_inds:
                for i in range(need_size):
                    noise_id = randint(0, num_noise_cand - 1)
                    noise[i] = noise_cand[noise_id]
                noise = np.reshape(noise, (shape[0], 1, shape[2]))
                data[:, :, :, ind[0], ind[1]] = torch.from_numpy(noise)

    else:
        print('error')

    if size is not None:
        # print("Noise: ", np.sqrt(noise))
        return noise
    else:
        return data


def transition_range(data, data_rv, confmap=None, trans_range=None):
    # shift_range > 0, move upward
    # shift_range < 0, move downward
    if trans_range is None:
        shift_range = randint(-Max_trans_rng, Max_trans_rng)
    else:
        shift_range = trans_range
    shape = data.shape
    assert len(shape) == 5
    if data_rv is not None:
        if shift_range != 0:
            data_new = torch.zeros_like(data)
            data_rv_new = torch.zeros_like(data_rv)
            if (confmap is not None):
                confmap_new = torch.zeros_like(confmap)
            gene_noise_data = torch.from_numpy(interpolation(data, [abs(shift_range), 256]))
            gene_noise_data_rv = torch.from_numpy(interpolation(data_rv, [abs(shift_range), 256]))

            if shift_range > 0:
                compen_mag = np.divide(range_grid[0:shape[3]-shift_range], range_grid[shift_range:shape[3]]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(data[:, :, :, 0:shape[3]-shift_range, :],
                #                                                                shift_range) * compen_mag
                data_new[:, :, :, shift_range:shape[3], :] = data[:, :, :, 0:shape[3] - shift_range, :] * compen_mag
                # data_new[:, :, :, shift_range:shape[3], :] = interpolation(data_new[:, :, :, shift_range:shape[3], :])
                data_new[:, :, :, 0:shift_range, :] = gene_noise_data
                data_rv_new[:, :, :, shift_range:shape[3], :] = data_rv[:, :, :, 0:shape[3] - shift_range, :] * compen_mag
                data_rv_new[:, :, :, 0:shift_range, :] = gene_noise_data_rv
                # confmap_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(confmap[:, :, :, 0:shape[3]-shift_range, :],
                #                                                                   shift_range)
                if (confmap is not None):
                    confmap_new[:, :, :, shift_range:shape[3], :] = confmap[:, :, :, 0:shape[3] - shift_range, :]
            else:
                shift_range = abs(shift_range)
                compen_mag = np.divide(range_grid[shift_range:shape[3]], range_grid[0:shape[3]-shift_range]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, 0:shape[3]-shift_range, :] = resamp_shiftrange(data[:, :, :, shift_range:shape[3], :],
                #                                                                  -shift_range) * compen_mag
                data_new[:, :, :, 0:shape[3] - shift_range, :] = data[:, :, :, shift_range:shape[3], :] * compen_mag
                # data_new[:, :, :, 0:shape[3]-shift_range, :] = interpolation(data_new[:, :, :, 0:shape[3]-shift_range, :])
                data_new[:, :, :, shape[3]-shift_range:shape[3], :] = gene_noise_data
                data_rv_new[:, :, :, 0:shape[3]-shift_range, :] = data_rv[:, :, :, shift_range:shape[3], :] * compen_mag
                data_rv_new[:, :, :, shape[3]-shift_range:shape[3], :] = gene_noise_data_rv
                # confmap_new[:, :, :, 0:shape[3]-shift_range, :] = resamp_shiftrange(confmap[:, :, :, shift_range:shape[3], :],
                #                                                                     -shift_range)
                if (confmap is not None):
                    confmap_new[:, :, :, 0:shape[3] - shift_range, :] = confmap[:, :, :, shift_range:shape[3], :]
            
            if (confmap is not None):
                return data_new, data_rv_new, confmap_new
            return data_new, data_rv_new, None
        else:
            if (confmap is not None):
                return data, data_rv, confmap
            else:
                return data, data_rv, None
    else:
        if shift_range != 0:
            data_new = torch.zeros_like(data)
            if (confmap is not None):
                confmap_new = torch.zeros_like(confmap)
            gene_noise_data = torch.from_numpy(interpolation(data, [abs(shift_range), 256]))

            if shift_range > 0:
                compen_mag = np.divide(range_grid[0:shape[3] - shift_range], range_grid[shift_range:shape[3]]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(data[:, :, :, 0:shape[3] - shift_range, :],
                #                                                                shift_range) * compen_mag
                data_new[:, :, :, shift_range:shape[3], :] = data[:, :, :, 0:shape[3] - shift_range, :] * compen_mag
                # data_new[:, :, :, shift_range:shape[3], :] = interpolation(data_new[:, :, :, shift_range:shape[3], :])
                data_new[:, :, :, 0:shift_range, :] = gene_noise_data
                # confmap_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(confmap[:, :, :, 0:shape[3] - shift_range, :],
                #                                                                   shift_range)
                if (confmap is not None):
                    confmap_new[:, :, :, shift_range:shape[3], :] = confmap[:, :, :, 0:shape[3] - shift_range, :]
            else:
                shift_range = abs(shift_range)
                compen_mag = np.divide(range_grid[shift_range:shape[3]], range_grid[0:shape[3] - shift_range]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, 0:shape[3] - shift_range, :] = resamp_shiftrange(data[:, :, :, shift_range:shape[3], :],
                #                                                                    -shift_range) * compen_mag
                data_new[:, :, :, 0:shape[3] - shift_range, :] = data[:, :, :, shift_range:shape[3], :] * compen_mag
                # data_new[:, :, :, 0:shape[3] - shift_range, :] = interpolation(data_new[:, :, :, 0:shape[3] - shift_range, :])
                data_new[:, :, :, shape[3] - shift_range:shape[3], :] = gene_noise_data
                # confmap_new[:, :, :, 0:shape[3] - shift_range, :] = resamp_shiftrange(confmap[:, :, :, shift_range:shape[3], :],
                #                                                                       -shift_range)
                if (confmap is not None):
                    confmap_new[:, :, :, 0:shape[3] - shift_range, :] = confmap[:, :, :, shift_range:shape[3], :]
            if (confmap is not None):
                return data_new, None, confmap_new
            else:
                return data_new, None, None
        else:
            if (confmap is not None):
                return data, None, confmap
            else:
                return data, None, None

def Aug_data(data, data_rv, data_va, confmap, type=None):
    if type == 'mix':
        prob = random()
        if prob < 0.3:
            data, data_va, confmap = Flip(data, data_va, confmap)
        prob = random()
        if prob < 0.4:
            data, data_va, confmap = transition_angle(data, data_va, confmap)
        prob = random()
        if prob < 0.4:
            data, data_rv, confmap = transition_range(data, data_rv, confmap)
    else:
        prob = random()
        if prob < 0.2:
            data, data_va, confmap = Flip(data, data_va, confmap)
        elif prob < 0.5:
            data, data_va, confmap = transition_angle(data, data_va, confmap)
        elif prob < 0.8:
            data, data_rv, confmap = transition_range(data, data_rv, confmap)
        else:
            pass

    return data, data_rv, data_va, confmap

if __name__ == '__main__':

    root_dir = "../Carrada"
    seqs = []
    for seq in os.listdir(root_dir):
        if "20" in seq:
            seqs.append(seq)
    seqs.sort()
    
    seq = "2019-09-16-12-52-12"
    index = 185
    box_anno_dir = "annotations/box"
    ra_box_anno_file = "range_angle_light.json"

    # open a file, where you stored the pickled data
    file = open(os.path.join(root_dir, seq, "confmaps_gt/train/" + seq + ".pkl"), 'rb')
    all_confmap = pickle.load(file)   # dump information to that file
    file.close()    # close the file

    # open a file, where you stored the data details
    file = open(os.path.join(root_dir, seq, "data_details/train/" + seq + ".pkl"), 'rb')
    all_details = pickle.load(file)   # dump information to that file
    file.close()    # close the file

    # confidence map of first frame
    print(len(all_confmap[0]))
    print(all_confmap[1][1:10])
    confmap = all_confmap[0][0][0:3]
    confmap = np.reshape(confmap, (1, 3, 1, 256, 256))

    plt.figure()
    plt.imshow(confmap[0,2,0,:,:])
    
    data = np.load(os.path.join(root_dir, seq, "range_angle_raw/" + str(index).zfill(6) + ".npy"))
    # print(data.shape)
    data_amp = data.copy()
    boxes = None
    with open(os.path.join(root_dir, seq, box_anno_dir, ra_box_anno_file), "r") as f:
        boxes = json.load(f)
        boxes = boxes[str(index).zfill(6)]["boxes"]
        boxes = boxes[0]

    # data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=2)
    data = np.reshape(data, (1, 1, 1, 256, 256))

    data = torch.from_numpy(data)
    confmap = torch.from_numpy(confmap)

    trans_range = 50
    trans_angle = 25
    # range translation 10 (move up 10 bins)
    data_shift_range, _, confmap_shift_range = transition_range(data, None, confmap, trans_range=trans_range)
    # angle translation 25 (move right 25 degrees)
    data_shift_angle, _, confmap_shift_angle = transition_angle(data, None, confmap, trans_angle=trans_angle)
    # flip angle
    data_flip, _, confmap_flip = Flip(data, None, confmap)
    # combination of range and angle translation
    data_combine, _, confmap_combine = transition_angle(data_shift_range, None, 
                                                        confmap_shift_range, trans_angle=trans_angle)

    print(data_shift_range.shape)
    print(data_shift_angle.shape)
    print(data_flip.shape)
    print(data_combine.shape)

    data_amp = data_amp ** 2
    data_amp1 = data_shift_range[0, 0, 0, :, :].numpy() ** 2 #+ data_shift_range[0, 1, 0, :, :] ** 2
    data_amp2 = data_shift_angle[0, 0, 0, :, :].numpy() ** 2 #+ data_shift_angle[0, 1, 0, :, :] ** 2
    data_amp3 = data_flip[0, 0, 0, :, :].numpy() ** 2 #+ data_flip[0, 1, 0, :, :] ** 2
    data_amp4 = data_combine[0, 0, 0, :, :].numpy() ** 2 #+ data_flip[0, 1, 0, :, :] ** 2

    data_amp.tofile("orginal.bin")
    data_amp1.tofile("range_shift.bin")
    data_amp2.tofile("angle_shift.bin")
    data_amp3.tofile("flipping.bin")
    data_amp4.tofile("combination.bin")

    sz_x = 8
    sz_y = 1
    plt.figure()
    plt.hist(data_amp[boxes[1]-sz_y:boxes[3]+sz_y, boxes[0]-sz_x:boxes[2]+sz_x].flatten(), 
             alpha=0.5, bins=100, label="org")
    plt.hist(data_amp1[boxes[1]+trans_range-sz_y:boxes[3]+trans_range+sz_y, boxes[0]-sz_x:boxes[2]+sz_x].flatten(), 
             alpha=0.5, bins=100, label="range_shift")
    plt.legend(loc='upper right')

    plt.figure()
    plt.hist(data_amp[boxes[1]-sz_y:boxes[3]+sz_y, boxes[0]-sz_x:boxes[2]+sz_x].flatten(), 
             alpha=0.5, bins=100, label="org")
    plt.hist(data_amp2[boxes[1]-sz_y:boxes[3]+sz_y, boxes[0]-sz_x+trans_angle:boxes[2]+sz_x+trans_angle].flatten(), 
             alpha=0.5, bins=100, label="angle_shift")
    plt.legend(loc='upper right')

    plt.figure()
    plt.hist(data_amp[boxes[1]-sz_y:boxes[3]+sz_y, boxes[0]-sz_x:boxes[2]+sz_x].flatten(), 
             alpha=0.5, bins=100, label="org")
    plt.hist(data_amp4[boxes[1]+trans_range-sz_y:boxes[3]+trans_range+sz_y, boxes[0]+trans_angle-sz_x:boxes[2]+trans_angle+sz_x].flatten(), 
             alpha=0.5, bins=100, label="comb")
    plt.legend(loc='upper right')

    # convert_data_2Dmat_to_histogram(data_amp, boxes, translation=[0,0])
    # convert_data_2Dmat_to_histogram(data_amp1.numpy(), boxes, translation=[10,0])
    # convert_data_2Dmat_to_histogram(data_amp2.numpy(), boxes, translation=[0,25])
    # convert_data_2Dmat_to_histogram(data_amp4.numpy(), boxes, translation=[10,25])
    # convert_data_2Dmat_to_histogram(np.asarray([data_amp, data_amp4.numpy()]),
    #                                 boxes, translation=[[0,0],[10,25]])

    plt.figure()
    # Create 1x3 sub plots
    gs = gridspec.GridSpec(2, 2)
    # visualize original image and augmentated images
    plt.figure(tight_layout=True)
    ax = plt.subplot(gs[0, 0])  # row 0, col 0
    plt.imshow(data_amp)
    # plt.imshow(data_amp[boxes[1]-sz_y:boxes[3]+sz_y, boxes[0]-sz_x:boxes[2]+sz_x])
    ax.set_title("Original RA image")

    ax = plt.subplot(gs[0, 1])  # row 0, col 1
    plt.imshow(data_amp1)
    # plt.imshow(data_amp1[boxes[1]+trans_range-sz_y:boxes[3]+trans_range+sz_y, boxes[0]-sz_x:boxes[2]+sz_x])
    ax.set_title("Range-shift RA image")

    ax = plt.subplot(gs[1, 0])  # row 0, col 2
    plt.imshow(data_amp2)
    # plt.imshow(data_amp2[boxes[1]-sz_y:boxes[3]+sz_y, boxes[0]-sz_x+trans_angle:boxes[2]+sz_x+trans_angle])
    ax.set_title("Angle-shift RA image")

    # ax = plt.subplot(gs[1, 0])  # row 0, col 3
    # plt.imshow(data_amp3[boxes[0]:boxes[2], boxes[1]:boxes[3]])
    # ax.set_title("Angle-flip RA image")

    ax = plt.subplot(gs[1, 1])  # row 0, col 3
    plt.imshow(data_amp4)
    # plt.imshow(data_amp4[boxes[1]+trans_range-sz_y:boxes[3]+trans_range+sz_y, boxes[0]+trans_angle-sz_x:boxes[2]+trans_angle+sz_x])
    ax.set_title("Range-Angle shift RA image")

    plt.show()
