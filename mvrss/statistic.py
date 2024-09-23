"""Main script to test a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from mvrss.utils.paths import Paths
from mvrss.utils.functions import count_params
from mvrss.learners.statistic_model import StatisticModel
from mvrss.models import TMVANet, MVNet
from mvrss.loaders.dataset import Carrada
from mvrss.loaders.dataloaders import SequenceCarradaDataset

def get_stat_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    paths = Paths().get()
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name
    model_path = path / 'results' / 'model.pt'
    test_results_path = path / 'results' / 'test_results.json'

    stat_model = StatisticModel(cfg)
    data = Carrada()
    # seq_name = "2019-09-16-13-18-33" # Two cars (summer - forward)
    # seq_name = "2019-09-16-13-20-20" # Two cars (summer - backward)
    # seq_name = "2020-02-28-13-06-53" # Cyclist and Car (winter - forward)
    test = data.get('Train')
    testset = SequenceCarradaDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    stat_model.set_annot_type(cfg["annot_type"])
    stat_model.predict(seq_testloader)

if __name__ == '__main__':
    get_stat_model()
