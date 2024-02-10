import gc
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
from mmengine.model import BaseModel
from albumentations import (Compose,
                            Normalize,) 
                            
from albumentations.pytorch import ToTensorV2
from rasterio.windows import Window
#
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from mmengine.runner.checkpoint import load_checkpoint
from ..utils.config import ConfigLoaded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(path: str):
    
    cfg = ConfigLoaded().get_config()
    model_list = get_model_list()
    pred_mask, h, w  = get_pred_mask(img_path= path, model_list=model_list)

    output_exp_path = Path(cfg.test_param.model_param.output_exp.sclerosis) / "masks"
    output_exp_path.mkdir(parents=False, exist_ok=True)
    id = Path(path).name.split(".")[0]
    cv2.imwrite(
        (output_exp_path / f"{id}-sclerosis.png").as_posix(),
        (pred_mask > 0.5).astype(np.uint8) * 255,
    )

    return pred_mask


def get_model_list():

    cfg = ConfigLoaded().get_config()
    model_param = cfg.test_param.model_param
    output_exp = model_param.output_exp.sclerosis

    filename_checkpoint = model_param.filename_checkpoint
    encoder_name = model_param.encoder_name
    encoder_weights = model_param.encoder_weights
    classes = model_param.classes
    activation = model_param.activation
    network_name = model_param.network_name

    if network_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )
    elif network_name == "manet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )

    elif network_name == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )
    else:
        raise Exception(f"Network {network_name} not found ")

    output_path = Path(output_exp) / filename_checkpoint

    logging.info(f"Load model in {output_path.as_posix()}")

    class _Model(BaseModel):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, imgs, labels, mode):
            x = self.model(imgs)
            if mode == 'loss':
                return 0 #{'loss': criterion(x, labels)}

            elif mode == 'predict':
                return x, labels

    model_mm = _Model()
    load_checkpoint(model_mm, output_path.as_posix())
    model_final = model_mm.model
    model_final.to(device)

    return [model_final]


def get_transforms_test():

    cfg = ConfigLoaded().get_config().train_param
    mean = cfg.mean
    std = cfg.std


    transforms = Compose(
        [
            Normalize(mean=(mean[0], mean[1], mean[2]), std=(std[0], std[1], std[2])),
            ToTensorV2(),
        ]
    )
    return transforms


def denormalize(z):
    cfg = ConfigLoaded().get_config().train_param
    mean = np.asarray(cfg["mean"])
    std = np.asarray(cfg["std"])
    mean_t = mean.reshape(-1, 1, 1)
    std_t = std.reshape(-1, 1, 1)
    return std_t * z + mean_t


class WSITestDataset(Dataset):
    def __init__(self, img_path, input_resolution, resolution, pad_size):
        super().__init__()
        self.data = rasterio.open(img_path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = input_resolution
        self.sz = resolution
        self.pad_sz = pad_size  # add to each input tile
        self.pred_sz = self.sz - 2 * self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz  # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz  # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        self.transforms = get_transforms_test()

    def __len__(self):
        return self.num_h * self.num_w

    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.pred_sz
        x = i_w * self.pred_sz
        py0, py1 = max(0, y), min(y + self.pred_sz, self.h)
        px0, px1 = max(0, x), min(x + self.pred_sz, self.w)

        # padding coordinate for rasterio
        qy0, qy1 = max(0, y - self.pad_sz), min(y + self.pred_sz + self.pad_sz, self.h)
        qx0, qx1 = max(0, x - self.pad_sz), min(x + self.pred_sz + self.pad_sz, self.w)

        # placeholder for input tile (before resize)
        img = np.zeros((self.sz, self.sz, 3), np.uint8)

        # replace the value
        if self.data.count == 3:
            img[0 : qy1 - qy0, 0 : qx1 - qx0] = np.moveaxis(
                self.data.read(
                    [1, 2, 3], window=Window.from_slices((qy0, qy1), (qx0, qx1))
                ),
                0,
                -1,
            )
        else:
            for i, layer in enumerate(self.layers):
                img[0 : qy1 - qy0, 0 : qx1 - qx0, i] = layer.read(
                    1, window=Window.from_slices((qy0, qy1), (qx0, qx1))
                )
        if self.sz != self.input_sz:
            img = cv2.resize(
                img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA
            )
        std = np.std(img)
        img = self.transforms(image=img)["image"]  # to normalized tensor
        return {
            "img": img,
            "p": [py0, py1, px0, px1],
            "q": [qy0, qy1, qx0, qx1],
            "std": std,
        }


def my_collate_fn(batch):
    img = []
    p = []
    q = []
    std = []
    for sample in batch:
        img.append(sample["img"])
        p.append(sample["p"])
        q.append(sample["q"])
        std.append(sample["std"])
    img = torch.stack(img)
    return {"img": img, "p": p, "q": q, "std": std}


def get_pred_mask(
    img_path,
    model_list,
):
    cfg = ConfigLoaded().get_config().test_param
    input_resolution = cfg.input_resolution 
    resolution =cfg.resolution
    test_batch_size  = cfg.test_batch_size
    pad_size = cfg.pad_size
    tta = cfg.tta
    mask_threshold = cfg.mask_threshold

    logging.info(f"Inference of the {img_path}")
    ds = WSITestDataset(
        img_path,
        input_resolution=input_resolution,
        resolution=resolution,
        pad_size=pad_size,
    )

    # rasterio cannot be used with multiple workers
    dl = DataLoader(
        ds,
        batch_size=test_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=my_collate_fn,
    )

    pred_mask = np.zeros((len(ds), ds.pred_sz, ds.pred_sz), dtype=np.uint8)

    i_data = 0
    for data in tqdm(dl):
        bs = data["img"].shape[0]
        img_patch = data["img"]  # (bs,3,input_res,input_res)
        pred_mask_float = 0
        for model in model_list:
            with torch.no_grad():
                if tta > 0:
                    pred_mask_float += (
                        torch.sigmoid(
                            model(
                                img_patch.to(device, torch.float32, non_blocking=True),
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()[:, 0, :, :]
                    )  # .squeeze()
                if tta > 1:
                    # h-flip
                    _pred_mask_float = (
                        torch.sigmoid(
                            model(
                                img_patch.flip([-1]).to(
                                    device, torch.float32, non_blocking=True,
                                ),

                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()[:, 0, :, :]
                    )  # .squeeze()
                    pred_mask_float += _pred_mask_float[:, :, ::-1]
                if tta > 2:
                    # v-flip
                    _pred_mask_float = (
                        torch.sigmoid(
                            model(
                                img_patch.flip([-2]).to(
                                    device, torch.float32, non_blocking=True
                                ),

                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()[:, 0, :, :]
                    )  # .squeeze()
                    pred_mask_float += _pred_mask_float[:, ::-1, :]
                if tta > 3:
                    # h-v-flip
                    _pred_mask_float = (
                        torch.sigmoid(
                            model(
                                img_patch.flip([-1, -2]).to(
                                    device, torch.float32, non_blocking=True
                                ),

                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()[:, 0, :, :]
                    )  # .squeeze()
                    pred_mask_float += _pred_mask_float[:, ::-1, ::-1]
        pred_mask_float = (
            pred_mask_float / min(tta, 4) / len(model_list)
        )  # (bs,input_res,input_res)

        # resize
        pred_mask_float = np.vstack(
            [
                cv2.resize(_mask.astype(np.float32), (ds.sz, ds.sz))[None]
                for _mask in pred_mask_float
            ]
        )

        # float to uint8
        pred_mask_int = (pred_mask_float > mask_threshold).astype(np.uint8)

        # replace the values
        for j in range(bs):
            py0, py1, px0, px1 = data["p"][j]
            qy0, qy1, qx0, qx1 = data["q"][j]
            std = data["std"][j]
            pred_mask_current = pred_mask_int[
                j, py0 - qy0 : py1 - qy0, px0 - qx0 : px1 - qx0
            ]  # (pred_sz,pred_sz)

            # check for empty images
            img = (np.transpose(denormalize(data["img"][j].numpy()), (1,2,0)) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            s_th = 40
            p_th = 1001*(img.shape[0]//256)**2
            if (s>s_th).sum() <= p_th or img.sum() <= p_th:
                 pred_mask_current = np.zeros_like(pred_mask_current)

            if std < 10:
                 pred_mask_current = np.zeros_like(pred_mask_current)

            pred_mask[i_data + j, 0 : py1 - py0, 0 : px1 - px0] = pred_mask_current

        i_data += bs

    pred_mask = pred_mask.reshape(ds.num_h * ds.num_w, ds.pred_sz, ds.pred_sz).reshape(
        ds.num_h, ds.num_w, ds.pred_sz, ds.pred_sz
    )
    pred_mask = pred_mask.transpose(0, 2, 1, 3).reshape(
        ds.num_h * ds.pred_sz, ds.num_w * ds.pred_sz
    )
    pred_mask = pred_mask[: ds.h, : ds.w]  # back to the original slide size
    non_zero_ratio = (pred_mask).sum() / (ds.h * ds.w)
    logging.info("non_zero_ratio = {:.4f}".format(non_zero_ratio))
    return pred_mask, ds.h, ds.w
