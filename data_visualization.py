import numpy as np
import matplotlib.pyplot as plt
import os
import glob

data_dir = "./data/Carrada/"
sequences = os.listdir(data_dir)
tmp = []
for seq in sequences:
    if "20" in seq:
        tmp.append(seq)
sequences = tmp
sequences.sort()

DOPPLER_RES = 0.41968030701528203
RANGE_RES = 0.1953125
ANGLE_RES = 0.01227184630308513
WORLD_RADAR = (0.0, 0.0)
MAX_RANGE_PIX = 255
MAX_ANGLE_PIX = 255
MAX_DOPPLE_PIX = 63
WORLD_RADAR = [0,0]

use_raw = False
if (use_raw):
    ra_folder = "range_angle_raw"
    rd_folder = "range_doppler_raw"
    ad_folder = "angle_doppler_raw"
else:
    ra_folder = "range_angle_processed"
    rd_folder = "range_doppler_processed"
    ad_folder = "angle_doppler_processed"
img_folder = "camera_images"
anno_folder = "annotations/dense"

seq_idx = 20
ra_list = os.listdir(os.path.join(data_dir,sequences[seq_idx],ra_folder))
ra_list.sort()

ad_list = os.listdir(os.path.join(data_dir,sequences[seq_idx],ad_folder))
ad_list.sort()

img_list = os.listdir(os.path.join(data_dir,sequences[seq_idx],img_folder))
img_list.sort()
img_list.remove("camera_timestamps.txt")

pred_list = glob.glob(os.path.join("weight/range_angle", sequences[seq_idx], "*.npy"))
pred_list.sort()

print("Load data from sequence ", sequences[seq_idx])
print("Number of frames ", len(ra_list))

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

# Load frame
fig = plt.figure()
ax = fig.subplots(1,3)
show_ra = None
show_rd = None
show_img = None
txt_ra = None
txt_rd = None

fig2 = plt.figure()
ax2 = fig2.subplots(2,3)
show_masks = [None] * 6

j=0
i=0
while j < 10:
    for i in range(len(ra_list)):
        ran, rdn, imn, rapn, rdpn = ra_list[i], ad_list[i], img_list[i], \
                                    pred_list[i], pred_list[i]
        rdpn = rdpn.replace("angle", "doppler")

        # Load
        ran = os.path.join(data_dir,sequences[seq_idx],ra_folder,ran)
        rdn = os.path.join(data_dir,sequences[seq_idx],ad_folder,rdn)
        imn = os.path.join(data_dir,sequences[seq_idx],img_folder,imn)

        raf = np.load(ran)
        rdf = np.load(rdn)
        img = plt.imread(imn)

        # Pre-process
        rdf = np.rot90(rdf,2)

        # Remove background class
        ram = np.load(rapn)[1:,:,:]
        rdm = np.load(rdpn)[1:,:,:]
        for l in range(rdm.shape[0]):
            rdm[l] = np.flip(rdm[l], 0)

        # plot contour
        if (i == 0 and j == 0):
            ax[0].set_title("Cam view", y=-0.2)
            ax[1].set_title("Range-angle view", y=-0.2)
            ax[2].set_title("Range-Doppler view", y=-0.2)
            
            txt_ra = ax[1].text(-90.0, 52.0, "max: {}\nmin: {}".format(np.max(raf), np.min(raf)))
            txt_rd = ax[2].text(-13.0, 52.0, "max: {}\nmin: {}".format(np.max(rdf), np.min(rdf)))

            show_img = ax[0].imshow(img)
            if (not use_raw):
                show_ra = ax[1].pcolormesh(RA_x, RA_y, raf, vmin=40.0, vmax=90.0)
                show_rd = ax[2].pcolormesh(RD_x, RD_y, rdf, vmin=50.0, vmax=100.0)
            else:
                show_ra = ax[1].pcolormesh(RA_x, RA_y, raf)
                show_rd = ax[2].pcolormesh(RD_x, RD_y, rdf)
            fig.colorbar(show_ra, ax=ax[1])
            fig.colorbar(show_rd, ax=ax[2])

            for k in range(3):
                show_masks[k] = ax2[0][k].pcolormesh(RA_x, RA_y, ram[k])
                show_masks[k+3] = ax2[1][k].pcolormesh(RD_x, RD_y, rdm[k])
                fig2.colorbar(show_masks[k], ax=ax2[0][k])
                fig2.colorbar(show_masks[k+3], ax=ax2[1][k])

        else:
            show_img.set_data(img)
            show_ra.set_array(raf.ravel())
            show_rd.set_array(rdf.ravel())

            txt_ra.set_text("max: {}\nmin: {}".format(np.max(raf), np.min(raf)))
            txt_rd.set_text("max: {}\nmin: {}".format(np.max(rdf), np.min(rdf)))

            for k in range(3):
                show_masks[k].set_array(ram[k].ravel())
                show_masks[k+3].set_array(rdm[k].ravel())

        plt.draw()
        plt.pause(0.001)
    
    j += 1