
---
# Terumo segmentation project 

[![codecov](https://codecov.io/gh/jeffersonfs/terumo-seg-esclerose/branch/main/graph/badge.svg?token=terumo-seg-esclerose_token_here)](https://codecov.io/gh/jeffersonfs/terumo-seg-esclerose)
[![CI](https://github.com/jeffersonfs/terumo-seg-esclerose/actions/workflows/main.yml/badge.svg)](https://github.com/jeffersonfs/terumo-seg-esclerose/actions/workflows/main.yml)

Awesome terumo_seg_esclerose created by jeffersonfs

## Install it from PyPI

```bash
pip install terumo_seg_esclerose
```

## Usage

```py
from terumo_seg_esclerose.cli import run_predict

quantity_sclerosis = run_predict(image_path, config_path)
```

```bash
$ python -m terumo_seg_esclerose run-predict <image> ./configs/tiny/tiny-efficientnetb0-unet-pipeline.py
#or
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
