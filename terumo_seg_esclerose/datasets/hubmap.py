import gc
import os
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import rasterio
import torch
import logging
import zipfile
# from .util import HuBMAPDataset, pre_processing_ing
from rasterio.windows import Window
from torch.utils.data import Dataset



def pre_processing(cfg):

    if not cfg.cache:
        logging.info("Pre processing HubMaP dataset.")
        logging.info("Extracting zip from dataset.")
        extractall_to_dir(Path(cfg.path))

    output_pre_csv_path = split_planning(cfg.path,
                                         cfg.dataset_pre_processing,
                                         cfg.batch_size,
                                         cfg.output_pre,
                                         cfg.split)

    output_csv = pre_processing_balanced(output_pre_csv_path,
                                         cfg.dataset_pre_processing,
                                         cfg.output_pre,
                                         cfg.multiplier_bin,
                                         cfg.binned_max,
                                         cfg.split,)
    
    logging.info("Create img and mask tiles diretory")
    
    pre_processing_get_imgs(output_csv, 
                            cfg.path,
                            cfg.dataset_pre_processing,
                            cfg.output_pre,
                            cfg.split)

    return output_csv




def extractall_to_dir(path):
    path_to_zip_file = path/"hubmap-kidney-segmentation.zip"
    logging.info(f"Extracting in {str(path)}")
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)


def pre_processing_get_imgs(
    balanced_csv,
    output,
    dataset_pre_processing,
    output_pre,
    split,
):
    output_pre_path = Path(output_pre)

    if split == "train": 
        output_pre_csv_path = output_pre_path / "data.csv"
    else:
        raise Exception(f"Split name not exist {split}")

    logging.info("Using pre processing cached patchs ")
    output_path = Path(output)
    idx_global = 0
    dataset_path = output_path

    # Identifier image splits
    train_split_infos = None 
    if split == "train":
        train_df = pd.read_csv(dataset_path/"train.csv")
    else:
        raise Exception(f"Split name not exist {split}")

    data_info_list = []
    data_balanced_csv = pd.read_csv(balanced_csv) 
    os.makedirs(output_pre_path, exist_ok=True)
    os.makedirs(output_pre_path / "img", exist_ok=True)
    os.makedirs(output_pre_path / "mask", exist_ok=True)
    logging.info(f"file imagens and masks in path {output_pre}")
    for row in train_df.itertuples(name="ImagesSeg"):

        idx = row.id
        img_path = dataset_path/"train"/(idx + ".tiff")
        logging.info(img_path)

        mask_path = dataset_path/"train"/(idx + ".png")

        ds = []

        for shift in dataset_pre_processing["shift_list"]:
            ds = HuBMAPDataset(
                img_path,
                mask_path,
                tile_size=dataset_pre_processing["tile_size"],
                shift_h=shift,
                shift_w=shift,
            )
            break

        filter_patchs_cvs = data_balanced_csv[data_balanced_csv.idx == idx].sort_values(by=['ratio_masked'])
        # img_path_list = []
        # mask_path_list = []
        for index, row in tqdm(filter_patchs_cvs.iterrows()):
            points = map(int, row["points"].strip('[]').split(','))

            img_save_path = Path(row["filename_img"])  # output_pre_path / "img" / f"{idx_global:04d}.png"
            mask_save_path = Path(row["filename_mask"])  # output_pre_path / "mask" / f"{idx_global:04d}.png"
            # img_path_list.append(img_save_path.as_posix())
            # mask_path_list.append(mask_save_path.as_posix())

            if not(img_save_path.exists() and mask_save_path.exists()):
                data = ds.getitem_by_points(points)
                img_patch = cv2.cvtColor(data["img"], cv2.COLOR_RGB2BGR)  # rgb -> bgr
                if not img_save_path.exists():
                    cv2.imwrite(img_save_path.as_posix(), img_patch)  # bgr -> rgb
                if not mask_save_path.exists():
                    cv2.imwrite(mask_save_path.as_posix(), data["mask"] * 255)

        # filter_patchs_cvs.loc[:, "filename_img"] = pd.Series(img_path_list, index=filter_patchs_cvs.index)
        # filter_patchs_cvs.loc[:, "filename_mask"] = pd.Series(mask_path_list, index=filter_patchs_cvs.index)

        # _log.info("Update data balanced .csv")
        # filter_patchs_cvs.to_csv()

def split_planning(
    output,
    dataset_pre_processing,
    batch_size,
    output_pre,
    split,
):
    output_pre_path = Path(output_pre)

    if split == "train": 
        output_pre_csv_path = output_pre_path / "data.csv"
    else:
        raise Exception(f"Split name not exist {split}")

    if os.path.exists(output_pre_csv_path):
        logging.info("Pre processing cached")
    else:
        output_path = Path(output)
        idx_global = 0
        dataset_path = output_path

        # Identifier image splits
        if split == "train":
            train_df = pd.read_csv(dataset_path/"train.csv")
        else:
            raise Exception(f"Split name not exist {split}")

        data_info_list = []
        idx_global = 0
        for row in train_df.itertuples(name="ImagesSeg"):
            
            idx = row.id
            rle = row.encoding


            img_path = dataset_path/"train"/(idx + ".tiff")
            logging.info(img_path)

            mask = make_mask_path(rle, img_path)
            mask_path = dataset_path/"train"/(idx + ".png")
            cv2.imwrite(mask_path.as_posix(), mask)
            del mask 
            gc.collect()

            dataset_list = []
            for shift in dataset_pre_processing.shift_list:
                ds = HuBMAPDataset(
                    img_path,
                    mask_path,
                    tile_size=dataset_pre_processing.tile_size,
                    shift_h=shift,
                    shift_w=shift,
                )
                dataset_list.append(ds)

            ds = ConcatDataset(dataset_list)

            dl = DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=True,
                # collate_fn=my_collate_fn
            )

            # img_patches = []
            # mask_patches = []
            logging.info(f"file: {img_path.name}")
            for data in tqdm(dl):
                img_patch = data["img"]
                mask_patch = (data["mask"] > 0).numpy().astype(np.uint8)

                for batch in range(mask_patch.shape[0]):
                    num_masked_pixels = mask_patch[batch].sum()
                    ratio_masked = num_masked_pixels / (mask_patch[batch].shape[0] * mask_patch[batch].shape[1])
                    is_masked = num_masked_pixels > 0
                    std = img_patch[batch].to(torch.float).std()

                    img_numpy = img_patch[batch].numpy()
                    hsv = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2HSV)
                    h,s,v = cv2.split(hsv)
                    s_th = 40
                    saturation_sum = (s>s_th).sum()
                    quant_black_level = img_numpy.sum()
                    
                    img_patch_path = output_pre_path / "img" / f"{idx_global:04d}.png"
                    mask_patch_path = output_pre_path / "mask" / f"{idx_global:04d}.png"


                    data_info = [
                        idx,
                        img_path.as_posix(),
                        mask_path.as_posix(),
                        img_patch_path.as_posix(),
                        mask_patch_path.as_posix(),
                        data["points"].numpy()[batch].tolist(),
                        num_masked_pixels.item(),
                        ratio_masked.item(),
                        std.item(),
                        is_masked.item(),
                        saturation_sum,
                        quant_black_level,
                    ]
                    data_info_list.append(data_info)
                    idx_global += 1



        data_df = pd.DataFrame(data_info_list).reset_index(drop=True)
        # print(data_df)
        data_df.columns = [
            "idx",
            "filename_img_wsi",
            "filename_mask_wsi",
            "filename_img",
            "filename_mask",
            "points",
            "num_masked_pixels",
            "ratio_masked",
            "std",
            "is_masked",
            "saturation_sum",
            "quant_black_level",
        ]
        os.makedirs(output_pre_csv_path.parent, exist_ok=True)
        data_df.to_csv(output_pre_csv_path, index=False)
    return output_pre_csv_path

 

def rle2mask(rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array <- 1(mask), 0(background)
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def make_mask_path(rle, img_path):
    data = rasterio.open(img_path)
    shape = (data.height, data.width)
    data.close()
    mask = rle2mask(rle, shape)
    return mask


def pre_processing_balanced(
    data_csv_path,
    dataset_pre_processing,
    output_pre,
    multiplier_bin,
    binned_max,
    split,
):
    output_pre_path = Path(output_pre)
    
    if split == "train": 
        output_pre_csv_path = output_pre_path / "data_balanced.csv"
    elif split == "test":
        output_pre_csv_path = output_pre_path / "data_balanced_test.csv"
    elif split == "train+test":
        output_pre_csv_path = output_pre_path / "data_balanced_train_test.csv"
    else:
        raise Exception(f"Split name not exist {split}")


    if output_pre_csv_path.exists():
        logging.info(f"Pre processing cached {output_pre_csv_path}")
    else:
        logging.info(f"Pre processing data_balanced.csv. Running ...")
        data_df = pd.read_csv(data_csv_path)
        data_df = data_df[data_df["std"] > 10].reset_index(drop=True)
        tile_size = dataset_pre_processing["tile_size"]
        p_th = 1001*(tile_size//256)**2
        data_df = data_df[data_df["saturation_sum"] > p_th].reset_index(drop=True)
        data_df = data_df[data_df["quant_black_level"] > p_th].reset_index(drop=True)


        # if (s>s_th).sum() <= p_th or img.sum() <= p_th:
        data_df["binned"] = np.round(data_df["ratio_masked"] * multiplier_bin).astype(
            int
        )
        data_df["is_masked"] = data_df["binned"] > 0
        n_sample = data_df["is_masked"].value_counts().min()
        data_df["binned"] = data_df["binned"].apply(
            lambda x: binned_max if x >= binned_max else x
        )

        data_df_false = data_df[data_df["is_masked"] == False].sample(
            n_sample, replace=True
        )
        data_df_true = data_df[data_df["is_masked"] == True].sample(
            n_sample, replace=True
        )

        n_bin = int(data_df_true["binned"].value_counts().mean())
        # print(data_df["binned"].unique())
        trn_df_list = []
        for bin_size in data_df["binned"].unique():
            trn_df_list.append(
                data_df[data_df["binned"] == bin_size].sample(n_bin, replace=True)
            )
        data_df_true = pd.concat(trn_df_list, axis=0)

        data_df_false = data_df[data_df["is_masked"] == False].sample(
            len(data_df_true) * 4, replace=True
        )

        data_df_balanced = pd.concat([data_df_false, data_df_true], axis=0).reset_index(
            drop=True
        )
        data_df_balanced.to_csv(output_pre_csv_path.as_posix(), index=False)

    return output_pre_csv_path




class HuBMAPDataset(Dataset):
    def __init__(self, img_path, mask_path, tile_size, shift_h, shift_w):
        super().__init__()
        self.data = rasterio.open(img_path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.sz = tile_size
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.pad_h = self.sz - self.h % self.sz  # add to whole slide
        self.pad_w = self.sz - self.w % self.sz  # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.sz
        self.num_w = (self.w + self.pad_w) // self.sz

        if self.h % self.sz < self.shift_h:
            self.num_h -= 1
        if self.w % self.sz < self.shift_w:
            self.num_w -= 1

        self.mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return self.num_h * self.num_w

    def __del__(self):
        self.data.close()
        del self.data

    def getitem_by_points(self, points):
        px0, py0, px1, py1 = points
        # placeholder for input tile (before resize)
        img_patch = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask_patch = np.zeros((self.sz, self.sz), np.uint8)

        # replace the value for img patch
        if self.data.count == 3:
            img_patch[0 : py1 - py0, 0 : px1 - px0] = np.moveaxis(
                self.data.read(
                    [1, 2, 3], window=Window.from_slices((py0, py1), (px0, px1))
                ),
                0,
                -1,
            )
        else:
            for i, layer in enumerate(self.layers):
                img_patch[0 : py1 - py0, 0 : px1 - px0, i] = layer.read(
                    1, window=Window.from_slices((py0, py1), (px0, px1))
                )

        # replace the value for mask patch
        mask_patch[0 : py1 - py0, 0 : px1 - px0] = self.mask[py0:py1, px0:px1]

        return {"img": img_patch, "mask": mask_patch}

    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.sz + self.shift_h
        x = i_w * self.sz + self.shift_w
        py0, py1 = max(0, y), min(y + self.sz, self.h)
        px0, px1 = max(0, x), min(x + self.sz, self.w)

        # placeholder for input tile (before resize)
        img_patch = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask_patch = np.zeros((self.sz, self.sz), np.uint8)

        # replace the value for img patch
        if self.data.count == 3:
            img_patch[0 : py1 - py0, 0 : px1 - px0] = np.moveaxis(
                self.data.read(
                    [1, 2, 3], window=Window.from_slices((py0, py1), (px0, px1))
                ),
                0,
                -1,
            )
        else:
            for i, layer in enumerate(self.layers):
                img_patch[0 : py1 - py0, 0 : px1 - px0, i] = layer.read(
                    1, window=Window.from_slices((py0, py1), (px0, px1))
                )

        # replace the value for mask patch
        mask_patch[0 : py1 - py0, 0 : px1 - px0] = self.mask[py0:py1, px0:px1]

        return {"img": img_patch, "mask": mask_patch, "points": np.asarray([px0, py0, px1, py1])}
