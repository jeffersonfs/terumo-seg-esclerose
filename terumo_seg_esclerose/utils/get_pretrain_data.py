import logging
import subprocess
from pathlib import Path



def get_pretrain_data():

    logging.info('Get dataset in Kaggle competitions in HubMap')
    path_base = Path("./dist/datasets/hubmap/")
    path_base.mkdir(parents=True,exist_ok=True)
    subprocess.run(["kaggle","competitions", "download", "-p", "./dist/datasets/hubmap", "-c", "hubmap-kidney-segmentation"])

