import pandas as pd 
from ..utils.config import ConfigLoaded
from pathlib import Path


def generate_csv():

    cfg = ConfigLoaded().get_config()
    path = Path(cfg.path)
    img_path_list = sorted(path.glob("**/*.tiff"))
    mask_path_list = sorted(path.glob("**/*.png"))
   
    train_list = []
    val_list = []
    test_list = []
    for i, m in zip(img_path_list, mask_path_list):
        if i.parent.stem in 'train':
            train_list.append((i, m))
        elif i.parent.stem in 'test':
            test_list.append((i, m))
        else:
            val_list.append((i, m))
  
    
