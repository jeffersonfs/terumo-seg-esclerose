# Import Libraries
import time
from pathlib import Path
import logging

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from torch.optim import SGD
from mmengine.runner import Runner


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pretraining(cfg):
    train_param = cfg.train_param
    fit(Path(train_param.output_path),
        Path(train_param.data_csv_path),
        train_param.test_size,
        train_param.random_state,
        train_param.mean,
        train_param.std,
        train_param.batch_size,
        train_param.shuffle,
        train_param.encoder_name,
        train_param.encoder_weights,
        train_param.classes,
        train_param.activation,
        train_param.network_name,
        train_param.loop_param,
        train_param.transform_param)
    
    


def transform_train(img_size):
    t_train = A.Compose(
        [
            A.Resize(height=img_size, width=img_size, p=1.0),
            # Basic
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            # Morphology
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=(-0.2, 0.2),
                rotate_limit=(-30, 30),
                interpolation=1,
                border_mode=0,
                value=(0, 0, 0),
                p=0.5,
            ),
            A.GaussNoise(var_limit=(0, 50.0), mean=0, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            # Color
            A.RandomBrightnessContrast(
                brightness_limit=0.35,
                contrast_limit=0.35,
                brightness_by_max=True,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=0, p=0.5
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.4),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.4),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=2,
                max_height=img_size // 4,
                max_width=img_size // 4,
                min_holes=1,
                min_height=img_size // 16,
                min_width=img_size // 16,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ]
    )

    return t_train


def transform_val(img_size):
    t_val = A.Compose([A.Resize(height=img_size, width=img_size), A.HorizontalFlip()])
    return t_val


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Dice():
    __name__ = "dice_score"

    def __init__(
        self, 
        eps=1e-7, 
        threshold=0.5, 
        activation=None, 
        ignore_channels=None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        return soft_dice_score(
            y_pr,
            y_gt,
            eps=self.eps,
        )


def train_loop(
    train_loader,
    val_loader,
    model,
    output_exp,
    filename_checkpoint,
    result_csv,
    max_lr,
    epochs,
    cache_weight,
    weight_decay,
    criterion_name,
):

    if criterion_name == "bce":
        criterion = smp.losses.SoftBCEWithLogitsLoss()
    # elif criterion_name == "bce+lovasz":
    #     criterion = smp.losses.SoftBCEWithLogitsLoss() + smp.losses.LovaszLoss(smp.losses.BINARY_MODE)
    else:
        raise Exception(f"Loss {criterion_name} non defined.")

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_dice = []
    train_dice = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    # _log.info(f"Running in {device}")
    # model.to(device)
    # fit_time = time.time()
    # best_mdice = -1
    # output_path = Path(output_exp) / filename_checkpoint


    class _Model(BaseModel):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, imgs, labels, mode):
            x = self.model(imgs)
            if mode == 'loss':
                return {'loss': criterion(x, labels)}
            elif mode == 'predict':
                return x, labels


    class _Dice(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # Save the results of a batch to `self.results`
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })
        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # Returns a dictionary with the results of the evaluated metrics,
            # where the key is the name of the metric
            return dict(accuracy=100 * total_correct / total_size)


    if cache_weight != None:
        cache_weight_path = Path(cache_weight)
        logging.info(f"Loading weights in {cache_weight_path.as_posix()}")
        checkpoint = torch.load(cache_weight_path.as_posix())
        model.load_state_dict(checkpoint["model"])

    logging.info(f"Running in {device}")
    model.to(device)
    fit_time = time.time()
    best_mdice = -1
    output_path = Path(output_exp) / filename_checkpoint


    if output_path.exists():
        logging.info(f"Cached weights in {output_path.as_posix()}")
        return

    metrics = [
        Dice(),
    ]

    runner = Runner(
        model=_Model(),
        work_dir='./work_dir',
        train_dataloader=train_loader,
        # a wrapper to execute back propagation and gradient update, etc.
        optim_wrapper=dict(optimizer=dict(type='Adam', lr=0.001)),
        # set some training configs like epochs
        train_cfg=dict(by_epoch=True, max_epochs=50, val_interval=1, val_begin=2,),
        val_dataloader=val_loader,
        val_cfg=dict(),
        val_evaluator=dict(type=_Dice),
        param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[4, 8], gamma=0.1)
    )
    runner.train()
    return


def fit(
    output_exp,
    data_csv_path,
    test_size,
    random_state,
    mean,
    std,
    batch_size,
    shuffle,
    encoder_name,
    encoder_weights,
    classes,
    activation,
    network_name,
    loop_param,
    transform_param,
):

    df = pd.read_csv(data_csv_path.as_posix())
    X_train, X_val = train_test_split(
        df, test_size=test_size, random_state=random_state
    )  # #datasets

    train_set = WSIDataset(
        X_train,
        mean,
        std,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        transform=transform_train(transform_param.img_size),
    )
    val_set = WSIDataset(
        X_val,
        mean,
        std,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        transform=transform_val(transform_param.img_size),
    )

    # dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)

    model = None

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

    elif network_name == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )
    else:
        raise Exception(f"Network {network_name} not found ")

    logging.info("Being train loop")
    train_loop(train_loader, 
               val_loader, 
               model, 
               output_exp,
               loop_param.filename_checkpoint,
               loop_param.result_csv,
               loop_param.max_lr,
               loop_param.epochs,
               loop_param.cache_weight,
               loop_param.weight_decay,
               loop_param.criterion_name,
               )


class WSIDataset(Dataset):
    def __init__(self, datas, mean, std, encoder_name, encoder_weights, transform=None):
        self.datas = datas
        self.transform = transform
        self.mean = mean
        self.std = std
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

        row = self.datas.iloc[idx]
        img_path = row["filename_img"]
        img_mask = row["filename_mask"]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(img_mask, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        pre = smp.encoders.get_preprocessing_fn(self.encoder_name, self.encoder_weights)

        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        # print(np.max(img))
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype("float32")

        t = T.Compose([T.Lambda(pre), T.Lambda(to_tensor)])
        img = t(np.asarray(img).astype(np.uint8))

        mask = torch.from_numpy(mask / 255).float()
        mask = torch.unsqueeze(mask, 0)

        return img, mask


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
