"""Class to test a model"""
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


class StatisticModel:
    """
    Class to form statistical data 

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

    def predict(self, seq_loader):
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
        rd_ped, rd_cyc, rd_car, ra_ped, ra_cyc, ra_car = [], [], [], [], [], [] 
        for i, sequence_data in enumerate(seq_loader):
            seq_name, seq = sequence_data
            path_to_frames = self.paths['carrada'] / seq_name[0]
            frame_dataloader = DataLoader(CarradaDataset(seq,
                                                        self.annot_type,
                                                        path_to_frames,
                                                        self.process_signal,
                                                        1),
                                        shuffle=False,
                                        batch_size=1,
                                        num_workers=6)
            for j, frame in enumerate(frame_dataloader):
                rd_data = frame['rd_matrix'][0,0,:,:]
                ra_data = frame['ra_matrix'][0,0,:,:]
                # rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                # ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                
                rd_mask = frame['rd_mask']
                ra_mask = frame['ra_mask']
                rd_cls = torch.argmax(rd_mask, axis=1)[0]
                ra_cls = torch.argmax(ra_mask, axis=1)[0]
                
                rd_ped = rd_ped + list(rd_data[rd_cls == 1].tolist())
                rd_cyc = rd_cyc + list(rd_data[rd_cls == 2].tolist())
                rd_car = rd_car + list(rd_data[rd_cls == 3].tolist())
                ra_ped = ra_ped + list(ra_data[ra_cls == 1].tolist())
                ra_cyc = ra_cyc + list(ra_data[ra_cls == 2].tolist())
                ra_car = ra_car + list(ra_data[ra_cls == 3].tolist())

        np.savetxt("./logs/rd_ped.txt", rd_ped, fmt='%1.5f')
        np.savetxt("./logs/rd_cyc.txt", rd_cyc, fmt='%1.5f')
        np.savetxt("./logs/rd_car.txt", rd_car, fmt='%1.5f')
        np.savetxt("./logs/ra_ped.txt", ra_ped, fmt='%1.5f')
        np.savetxt("./logs/ra_cyc.txt", ra_cyc, fmt='%1.5f')
        np.savetxt("./logs/ra_car.txt", ra_car, fmt='%1.5f')

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
