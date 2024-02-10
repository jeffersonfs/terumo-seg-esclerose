"""Entry point for terumo_seg_esclerose."""

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

from terumo_seg_esclerose.cli import app  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    app()
