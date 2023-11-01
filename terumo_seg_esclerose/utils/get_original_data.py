import logging
from pathlib import Path

import cv2
import numpy as np
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance
from decouple import config
from shapely import wkt
from shapely.affinity import affine_transform
from tqdm import tqdm
import zipfile
import hashlib
def get_mask(geo, image):
    geometry_opencv = affine_transform(geo, [1, 0, 0, -1, 0, image.height])
    mask = np.zeros([image.height, image.width])
    points = [[x, y] for x, y in zip(*geometry_opencv.boundary.coords.xy)]
    mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=1)
    return mask


def get_data_cytomine(path_save, id_project, verbose=logging.INFO):
    host = config("CYTOMINE_HOST")
    public_key = config("CYTOMINE_PUBLIC_KEY")
    private_key = config("CYTOMINE_PRIVATE_KEY")

    count = 1
    with Cytomine(
        host, public_key, private_key, verbose=logging.INFO
    ) as cytomine:
        annotations = AnnotationCollection()
        annotations.project = id_project
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        annotations.fetch()
        quant_annotation = len(annotations)
        for annotation in tqdm(annotations):
            image = ImageInstance().fetch(id=annotation.image)
            geometry = wkt.loads(annotation.location)
            # Save image in directory
            path_file = path_save / "imgs" / f"{count:05d}.tiff"
            if not path_file.exists():
                image.download(path_file.as_posix())

            # Save mask in path
            mask = get_mask(geometry, image)
            path_file_mask = Path(path_save / "masks" / f"{count:05d}.png")
            path_file_mask.parent.mkdir(parents=True, exist_ok=True)
            if not path_file_mask.exists():
                cv2.imwrite(path_file_mask.as_posix(), mask*255)

            count += 1


def get_datasets():
    projects = {
        "pams": config("CYTOMINE_PROJECT_NUMBER_PAMS"),
        "pas": config("CYTOMINE_PROJECT_NUMBER_PAS"),
        "he": config("CYTOMINE_PROJECT_NUMBER_HE"),
    }
    path_base = Path("./dist/datasets")
    path_base.mkdir(parents=True,exist_ok=True)
    for i in projects.keys():
        path_base_item = path_base / i
        path_base_item.mkdir(exist_ok=True)
        get_data_cytomine(path_base_item, projects[i])

    zip_filename = './dist/datasets_esclerose.zip'
    with zipfile.ZipFile(zip_filename, 'w') as f:
        for file in path_base.glob("**/*"):
            f.write(file)

    sha256_hash = hashlib.sha256()
    sha256_filename = './dist/datasets_escherose_sha256.txt'
    with open(zip_filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)

    with open(sha256_filename, 'w') as f:
        f.write(sha256_hash.hexdigest())

