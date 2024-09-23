import logging
import torch

from models.get_model import get_arch
from utils.model_saving_loading import load_model

#======================================================================#

def load_models():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Check if CUDA is available
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using CUDA...")
        device = torch.device("cuda:0")
    else:
        logging.info("CUDA is not available. Using CPU...")
        device = torch.device("cpu")
    logging.info(f'Using device {device}')

    model_name = "wnet"

    model_1 = get_arch(model_name, n_classes=3).to(device)
    model_2 = get_arch(model_name, n_classes=3).to(device)
    model_3 = get_arch(model_name, n_classes=3).to(device)
    model_4 = get_arch(model_name, n_classes=3).to(device)
    model_5 = get_arch(model_name, n_classes=3).to(device)
    model_6 = get_arch(model_name, n_classes=3).to(device)
    model_7 = get_arch(model_name, n_classes=3).to(device)
    model_8 = get_arch(model_name, n_classes=3).to(device)
    
    experiment_path_1 = './experiments/wnet_All_three_1024_disc_cup/28/'
    experiment_path_2 = './experiments/wnet_All_three_1024_disc_cup/30/'
    experiment_path_3 = './experiments/wnet_All_three_1024_disc_cup/32/'
    experiment_path_4 = './experiments/wnet_All_three_1024_disc_cup/34/'
    experiment_path_5 = './experiments/wnet_All_three_1024_disc_cup/36/'
    experiment_path_6 = './experiments/wnet_All_three_1024_disc_cup/38/'
    experiment_path_7 = './experiments/wnet_All_three_1024_disc_cup/40/'
    experiment_path_8 = './experiments/wnet_All_three_1024_disc_cup/42/'

    model_1, stats = load_model(model_1, experiment_path_1, device)
    model_1.eval()

    model_2, stats = load_model(model_2, experiment_path_2, device)
    model_2.eval()
    
    model_3, stats = load_model(model_3, experiment_path_3, device)
    model_3.eval()
    
    model_4, stats = load_model(model_4, experiment_path_4, device)
    model_4.eval()
    
    model_5, stats = load_model(model_5, experiment_path_5, device)
    model_5.eval()
    
    model_6, stats = load_model(model_6, experiment_path_6, device)
    model_6.eval()
    
    model_7, stats = load_model(model_7, experiment_path_7, device)
    model_7.eval()
    
    model_8, stats = load_model(model_8, experiment_path_8, device)
    model_8.eval()

    return model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8

#======================================================================#
