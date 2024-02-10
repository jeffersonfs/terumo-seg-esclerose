"""
CLI interface for terumo_seg_esclerose project.
"""
import typer
from .utils.get_original_data import get_datasets
from .utils.get_pretrain_data import get_pretrain_data
from .utils.config import ConfigLoaded
from .datasets.hubmap import pre_processing
from .datasets.sclerosis import generate_csv
from .train.pre_training import run_pretraining
from .train.train import run_train
from .predict.seg_glomerulus import predict as predict_glo
from .predict.seg_sclerosis import predict as predict_sle
from mmengine.config import Config
import numpy as np
app = typer.Typer()

@app.command()
def make_datasets():
    get_datasets()
    get_pretrain_data()


@app.command()
def prepare_datasets(path: str):
    cfg = Config.fromfile(path)

    ConfigLoaded().load_config(path)
    # pre_processing(cfg) 
    generate_csv()


@app.command()
def pretraining(path: str):
    cfg = Config.fromfile(path)
    run_pretraining(cfg)


@app.command()
def run_predict(image_path, path: str):
    ConfigLoaded().load_config(path)
    mask_glo = predict_glo(image_path)
    mask_sle = predict_sle(image_path)

    inter = np.logical_and(mask_glo > 0.5, mask_sle > 0.5)
    p = np.sum(inter) / (np.sum(mask_glo > 0.5) + 0.00001)

    print(f"Glomerulu with {p} sclerosis")
    return p

@app.command()
def train(path: str):
    ConfigLoaded().load_config(path)
    run_train()



@app.command()
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m terumo_seg_esclerose` and `$ terumo_seg_esclerose `.
    """
    print("Command principal")
