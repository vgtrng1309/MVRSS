"""Class to test a model"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mvrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives, mask_to_img, get_non_img_qualitatives
from mvrss.utils.paths import Paths
from mvrss.utils.metrics import Evaluator
from mvrss.loaders.dataloaders import CarradaDataset


class Tester:
    """
    Class to test a model

    PARAMETERS
    ----------
    cfg: dict
        Configuration parameters used for train/test
    visualizer: object or None
        Add a visulization during testing
        Default: None
    """

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.visualizer = visualizer
        self.model = self.cfg['model']
        self.nb_classes = self.cfg['nb_classes']
        self.use_ad = self.cfg['use_ad']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        rd_criterion = define_loss('range_doppler', self.custom_loss, self.device)
        ra_criterion = define_loss('range_angle', self.custom_loss, self.device)
        nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        coherence_running_losses = list()
        rd_metrics = Evaluator(num_class=self.nb_classes)
        ra_metrics = Evaluator(num_class=self.nb_classes)
        total_cumulative_time = 0
        total_num_frames = 0
        timer_flag = True
        
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths['carrada'] / seq_name[0]
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.norm_type,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_rd = self.n_frames-1
                    quali_iter_ra = self.n_frames-1
                j = 0
                for frame, _, _, _ in frame_dataloader:
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    if self.use_ad is True:
                        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)

                    start_time_dummy = time.time()

                    if self.model == 'mvnet' or self.model == 'radarformer2':
                        rd_outputs, ra_outputs = net(rd_data, ra_data)
                    else:
                        rd_outputs, ra_outputs = net(rd_data, ra_data, ad_data) 
                    end_time_dummy = time.time() - start_time_dummy
                    total_cumulative_time += end_time_dummy
                    total_num_frames += 1
                    
                    if timer_flag == True:
                        total_cumulative_time = 0
                        total_num_frames = 0
                        timer_flag = False

                    rd_outputs = rd_outputs.to(self.device)
                    ra_outputs = ra_outputs.to(self.device)

                    if get_quali:
                        quali_iter_rd = get_non_img_qualitatives(rd_outputs, rd_mask, self.paths,
                                                         seq_name, quali_iter_rd, 'range_doppler')
                        quali_iter_ra = get_non_img_qualitatives(ra_outputs, ra_mask, self.paths,
                                                         seq_name, quali_iter_ra, 'range_angle')

                    rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                         torch.argmax(rd_outputs, axis=1).cpu())
                    ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                         torch.argmax(ra_outputs, axis=1).cpu())

                    if nb_losses < 3:
                        # Case without the CoL
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                     for c in ra_criterion]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        loss = torch.mean(rd_loss + ra_loss)
                    else:
                        # Case with the CoL
                        # Select the wCE and wSDice
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion[:2]]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                     for c in ra_criterion[:2]]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        # Coherence loss
                        coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                        loss = torch.mean(rd_loss + ra_loss + coherence_loss)

                    running_losses.append(loss.data.cpu().numpy()[()])
                    rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                    rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                    rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                    ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                    ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                    ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                    if nb_losses > 2:
                        coherence_running_losses.append(coherence_loss.data.cpu().numpy()[()])

                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                            ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                            rd_gt_masks = torch.argmax(rd_mask, axis=1)[:5]
                            ra_gt_masks = torch.argmax(ra_mask, axis=1)[:5]
                            rd_pred_grid = make_grid(transform_masks_viz(rd_pred_masks,
                                                                         self.nb_classes))
                            ra_pred_grid = make_grid(transform_masks_viz(ra_pred_masks,
                                                                         self.nb_classes))
                            rd_gt_grid = make_grid(transform_masks_viz(rd_gt_masks,
                                                                       self.nb_classes))
                            ra_gt_grid = make_grid(transform_masks_viz(ra_gt_masks,
                                                                       self.nb_classes))
                            self.visualizer.update_multi_img_masks(rd_pred_grid, rd_gt_grid,
                                                                   ra_pred_grid, ra_gt_grid,
                                                                   iteration)
                    j += 1
                print('Total time: ', total_cumulative_time, ' on ', total_num_frames, ' frames')

            self.test_results = dict()
            self.test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                             [np.mean(sub_loss) for sub_loss
                                                              in rd_running_global_losses])
            self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
                                                           [np.mean(sub_loss) for sub_loss
                                                            in ra_running_global_losses])
            if nb_losses > 2:
                self.test_results['coherence_loss'] = np.mean(coherence_running_losses).item()
            self.test_results['global_acc'] = (1/2)*(self.test_results['range_doppler']['acc']+
                                                     self.test_results['range_angle']['acc'])
            self.test_results['global_prec'] = (1/2)*(self.test_results['range_doppler']['prec']+
                                                      self.test_results['range_angle']['prec'])
            self.test_results['global_dice'] = (1/2)*(self.test_results['range_doppler']['dice']+
                                                      self.test_results['range_angle']['dice'])

            rd_metrics.reset()
            ra_metrics.reset()
            print('Total testing time is: ', total_cumulative_time)
            print('Total number of frames is: ', total_num_frames)
            print('Per frame time is: ', total_cumulative_time/total_num_frames)
        return self.test_results

    def predict_nosave(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))

        fig_sum, ax_sum = plt.subplots(3, 3)
        fig_sum.tight_layout()
        show_plot_sum = [[None, None, None],
                         [None, None, None],
                         [None, None, None]]

        # fig = plt.figure()
        # fig.tight_layout()
        # ax = fig.subplots(2,4)
        # fig2, ax2 = plt.subplots()
        # fig2.tight_layout()
        show_ragt = None
        show_rdgt = None
        show_rapr = None
        show_rdpr = None
        i = 0

        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths['carrada'] / seq_name[0]
                viz_dir = path_to_frames / "viz"
                try:
                    os.makedirs(viz_dir / "compare")
                except:
                    print("folder exists")
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.norm_type,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                start_frame = True
                k = 0
                for frame, rd, ra, ad in frame_dataloader:
                    k += 1
                    if (k < 30):
                        continue
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    rd_data_non_norm = rd_data.clone()
                    ra_data_non_norm = ra_data.clone()
                    ad_data_non_norm = ad_data.clone()
                    
                    # TODO: Change normalization here
                    ra_norm = normalize(ra, 'range_angle', norm_type=self.norm_type)
                    print(torch.max(ra_norm), torch.min(ra_norm))
                    rd_norm = normalize(rd, 'range_doppler', norm_type=self.norm_type)
                    ad_norm = normalize(ad, 'angle_doppler', norm_type=self.norm_type)
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    print(torch.max(ra_data), torch.min(ra_data))
                    ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                    rd_mask = mask_to_img(torch.argmax(rd_mask, axis=1).cpu().numpy()[0])
                    ra_mask = mask_to_img(torch.argmax(ra_mask, axis=1).cpu().numpy()[0])

                    if (start_frame):
                        show_plot_sum[0][0] = ax_sum[0][0].imshow(ra_norm[0,0,:,:], vmin=0.0, vmax=1.0)
                        show_plot_sum[1][0] = ax_sum[1][0].imshow(rd_norm[0,0,:,:], vmin=0.0, vmax=1.0)
                        show_plot_sum[2][0] = ax_sum[2][0].imshow(ad_norm[0,0,:,:], vmin=0.0, vmax=1.0)
                        show_plot_sum[0][1] = ax_sum[0][1].imshow(ra_data[0,0,0,:,:], vmin=0.0, vmax=1.0)
                        show_plot_sum[1][1] = ax_sum[1][1].imshow(rd_data[0,0,0,:,:], vmin=0.0, vmax=1.0)
                        show_plot_sum[2][1] = ax_sum[2][1].imshow(ad_data[0,0,0,:,:], vmin=0.0, vmax=1.0)
                        show_plot_sum[0][2] = ax_sum[0][2].imshow(ra_mask)
                        show_plot_sum[1][2] = ax_sum[1][2].imshow(rd_mask)
                        ax_sum[0][0].set_title("RA org data")
                        ax_sum[1][0].set_title("RD org data")
                        ax_sum[2][0].set_title("AD org data")
                        ax_sum[0][1].set_title("RA shifted data")
                        ax_sum[1][1].set_title("RD shifted data")
                        ax_sum[2][1].set_title("AD shifted data")
                        ax_sum[0][2].set_title("RA shifted mask")
                        ax_sum[1][2].set_title("RD shifted mask")

                        # show_plot = ax[0][3].plot(np.arange(0,256,1),
                        #                      torch.mean(ra[0,0,:,:],axis=0))
                        # show_plot1 = ax[1][3].plot(np.arange(0,256,1),
                        #                      torch.mean(ra_data_non_norm[0,0,0,:,:],axis=0))

                        # show_plot.ylim(0, 100000)
                        # show_raorg = ax[0][0].imshow(ra[0,0,:,:], vmin=0.0, vmax=50000.0)
                        # show_rdorg = ax[1][0].imshow(rd[0,0,:,:], vmin=0.0, vmax=100000.0)
                        # show_rashift = ax[0][1].imshow(ra_data_non_norm[0,0,0,:,:], vmin=0.0, vmax=100000.0)
                        # show_rdshift = ax[1][1].imshow(rd_data_non_norm[0,0,0,:,:], vmin=0.0, vmax=100000.0)
                        # show_ragt = ax[0][2].imshow(ra_mask)
                        # show_rdgt = ax[1][2].imshow(rd_mask)
                        # show_ad = ax[0][4].imshow(ad_data[0,0,0,:,:])
                        # fig.colorbar(show_raorg, ax=ax[0][0], orientation='vertical')
                        # fig.colorbar(show_rdorg, ax=ax[1][0], orientation='vertical')
                        # fig.colorbar(show_rashift, ax=ax[0][1], orientation='vertical')
                        # fig.colorbar(show_rdshift, ax=ax[1][1], orientation='vertical')
                    else:
                        show_plot_sum[0][0].set_data(ra_norm[0,0,:,:])
                        show_plot_sum[1][0].set_data(rd_norm[0,0,:,:])
                        show_plot_sum[2][0].set_data(ad_norm[0,0,:,:])
                        show_plot_sum[0][1].set_data(ra_data[0,0,0,:,:])
                        show_plot_sum[1][1].set_data(rd_data[0,0,0,:,:])
                        show_plot_sum[2][1].set_data(ad_data[0,0,0,:,:])
                        show_plot_sum[0][2].set_data(ra_mask)
                        show_plot_sum[1][2].set_data(rd_mask)

                        # show_plot[-1].set_data(np.arange(0,256,1),
                        #                    torch.mean(ra[0,0,:,:],axis=0))
                        # show_plot1[-1].set_data(np.arange(0,256,1),
                        #                    torch.mean(ra_data_non_norm[0,0,0,:,:],axis=0))
                        # show_raorg.set_data(ra[0,0,:,:])
                        # show_rdorg.set_data(rd[0,0,:,:])
                        # show_rashift.set_data(ra_data_non_norm[0,0,0,:,:])
                        # show_rdshift.set_data(rd_data_non_norm[0,0,0,:,:])
                        # show_ragt.set_data(ra_mask)
                        # show_rdgt.set_data(rd_mask)
                        # show_ad.set_data(ad_data[0,0,0,:,:])

                    # if (k % 25 == 0):
                    #     fig_sum.savefig(viz_dir / "compare" / (str(k)+".png"), dpi=750)
                    show_model_result = True
                    if (show_model_result):
                        start = time.time()
                        if self.model == 'tmvanet':
                            ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                            rd_outputs, ra_outputs = net(rd_data, ra_data, ad_data)
                        else:
                            rd_outputs, ra_outputs = net(rd_data, ra_data)
                        end = time.time()
                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)

                        rd_outputs = mask_to_img(torch.argmax(rd_outputs, axis=1).cpu().numpy()[0])
                        ra_outputs = mask_to_img(torch.argmax(ra_outputs, axis=1).cpu().numpy()[0])

                        print("Model processing time: ", end - start)
                        #"""
                        if (start_frame):
                            show_rapr = ax[0][3].imshow(ra_outputs)
                            show_rdpr = ax[1][3].imshow(rd_outputs)
                        else:
                            show_rapr.set_data(ra_outputs)
                            show_rdpr.set_data(rd_outputs)
                    start_frame = False
                    i += 1
                    plt.draw()
                    plt.pause(0.0001)
                    #"""
        return
    def write_params(self, path):
        """Write quantitative results of the Test"""
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        """Set device used for test (supported: 'cuda', 'cpu')"""
        self.device = device

    def set_annot_type(self, annot_type):
        """Set annotation type to test on (specific to CARRADA)"""
        self.annot_type = annot_type
