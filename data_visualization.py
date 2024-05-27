import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from carrada_dataset.utils.camera import Camera

data_dir = "./data/Carrada/"
intrinsic_path = data_dir + "cam_params/intrinsics.xml"
extrinsic_path = data_dir + "cam_params/extrinsics_2020.xml"

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
IMG_SIZE = (1232, 1028)
COLOR_CODE = np.array([[0, 0, 0],
                       [255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255]])

# Load camera object
cam = Camera(intrinsic_path, extrinsic_path)

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

seq_idx = 0 #20
sequences[seq_idx] = "2020-02-28-13-10-51" #"2019-09-16-13-18-33"
ra_list = os.listdir(os.path.join(data_dir,sequences[seq_idx],ra_folder))
ra_list.sort()

ad_list = os.listdir(os.path.join(data_dir,sequences[seq_idx],ad_folder))
ad_list.sort()

img_list = os.listdir(os.path.join(data_dir,sequences[seq_idx],img_folder))
img_list.sort()
img_list.remove("camera_timestamps.txt")
print(len(img_list))

pred_list = glob.glob(os.path.join("weight/range_angle", sequences[seq_idx], "*.npy"))
pred_list.sort()
print(len(pred_list))


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
ax = fig.subplots(1,2)
show_ra = None
show_rd = None
txt_ra = None
txt_rd = None

is_show_mask = True
if is_show_mask:
    fig2_a = plt.figure()
    fig2_a.tight_layout()
    ax2_a = fig2_a.subplots(1,3)
    show_masks_a = [None] * 3
    fig2_d = plt.figure()
    fig2_d.tight_layout()
    ax2_d = fig2_d.subplots(1,3)
    show_masks_d = [None] * 3

fig3 = plt.figure()
ax3 = fig3.subplots()
show_img3 = None

j=0
i=0
while j < 10:
    for i in range(len(ra_list)):
        try:
            ran, rdn, imn, rapn, rdpn = ra_list[i], ad_list[i], img_list[i], \
                                        pred_list[i], pred_list[i]
            rdpn = rdpn.replace("angle", "doppler")

            # Load
            ran = os.path.join(data_dir,sequences[seq_idx],ra_folder,ran)
            rdn = os.path.join(data_dir,sequences[seq_idx],ad_folder,rdn)
            imn = os.path.join(data_dir,sequences[seq_idx],img_folder,imn)

            raf = np.load(ran)
            rdf = np.load(rdn)
            img = np.array(plt.imread(imn))
            # Pre-process
            rdf = np.rot90(rdf,2)

            # Load range-angle output mask
            ram = np.load(rapn)
            ram_img = np.argmax(ram, axis=0)
            ram = ram[1:,:,:]

            if is_show_mask:
                for l in range(ram.shape[0]):
                    print(np.min(ram[l]), end=" ")
                    print(np.max(ram[l]), end=" ")
                print()

                # Load range-doppler output mask
                rdm = np.load(rdpn)[1:,:,:]
                for l in range(rdm.shape[0]):
                    rdm[l] = np.flip(rdm[l], 0)
                    rdm[l] = np.flip(rdm[l], 1)
                    print(np.min(rdm[l]), end=" ")
                    print(np.max(rdm[l]), end=" ")
                print("\n")

            obj_points = np.transpose(np.array(np.nonzero(ram_img)))
            for obj_point in obj_points:
                # print(obj_point)
                ra = R_axis[obj_point[0]]
                an = A_axis[obj_point[1]] * np.pi / 180.0
                # print(ra, an)
                xm = ra * np.sin(an)
                ym = ra * np.cos(an)
                zm = 0.0
                xc, yc, zc = cam.worldToCam(xm, ym, zm)
                # print(xc, yc, zc, xm, ym, zm)
                # print(ram_img[obj_point[0], obj_point[1]])
                # img[v, u] = COLOR_CODE[ram_img[obj_point[0], obj_point[1]]]
            # exit(0)

            # plot contour
            if (i == 0 and j == 0):
                ax[0].set_title("Range-angle view", y=-0.2)
                ax[1].set_title("Range-Doppler view", y=-0.2)
                
                txt_ra = ax[0].text(-90.0, 50.5, "max: {}\nmin: {}".format(np.max(raf), np.min(raf)))
                txt_rd = ax[1].text(-13.0, 50.5, "max: {}\nmin: {}".format(np.max(rdf), np.min(rdf)))

                if (not use_raw):
                    show_ra = ax[0].pcolormesh(RA_x, RA_y, raf, vmin=40.0, vmax=90.0)
                    show_rd = ax[1].pcolormesh(RD_x, RD_y, rdf, vmin=50.0, vmax=100.0)
                else:
                    show_ra = ax[0].pcolormesh(RA_x, RA_y, raf)
                    show_rd = ax[1].pcolormesh(RD_x, RD_y, rdf)
                fig.colorbar(show_ra, ax=ax[0])
                fig.colorbar(show_rd, ax=ax[1])
                fig.suptitle("Range-Angle and Range-Doppler input")
                ax[0].set_xlabel("Angle (degree)")
                ax[0].set_ylabel("Range (m)")
                ax[1].set_xlabel("Doppler (m/s)")
                ax[1].set_ylabel("Range (m)")

                show_img3 = ax3.imshow(img)

                if is_show_mask:
                    fig2_a.suptitle("Rangle-Angle model output")
                    fig2_d.suptitle("Rangle-Doppler model output")
                    for k in range(3):
                        show_masks_a[k] = ax2_a[k].pcolormesh(RA_x, RA_y, ram[k], vmin=-12.0, vmax=12.0)
                        show_masks_d[k] = ax2_d[k].pcolormesh(RD_x, RD_y, rdm[k], vmin=-12.0, vmax=12.0)
                        fig2_a.colorbar(show_masks_a[k], ax=ax2_a[k])
                        fig2_d.colorbar(show_masks_d[k], ax=ax2_d[k])
                        ax2_a[k].set_xlabel("Angle (degree)")
                        ax2_a[k].set_ylabel("Range (m)")
                        ax2_d[k].set_xlabel("Doppler (m/s)")
                        ax2_d[k].set_ylabel("Range (m)")
            else:
                show_ra.set_array(raf.ravel())
                show_rd.set_array(rdf.ravel())

                txt_ra.set_text("max: {}\nmin: {}".format(np.max(raf), np.min(raf)))
                txt_rd.set_text("max: {}\nmin: {}".format(np.max(rdf), np.min(rdf)))

                show_img3.set_data(img)

                if is_show_mask:
                    for k in range(3):
                        show_masks_a[k].set_array(ram[k].ravel())
                        show_masks_d[k].set_array(rdm[k].ravel())
        except:
            continue

        plt.draw()
        plt.pause(0.001)
    
    j += 1