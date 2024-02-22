import logging
from typing import Callable

import dask as da
import numpy as np
import tqdm
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from sopa._sdata import get_intrinsic_cs, get_key
from sopa.segmentation import Patches2D
from sopa.utils.wsi import _get_extraction_parameters, _numpy_patch

log = logging.getLogger(__name__)


def embed_batch(model_name: str, device: str) -> tuple[Callable, int]:
    import torch

    import sopa.embedding.models as models

    assert hasattr(
        models, model_name
    ), f"'{model_name}' is not a valid model name under `sopa.embedding.models`. Valid names are: {', '.join(models.__all__)}"

    model: torch.nn.Module = getattr(models, model_name)()
    model.eval().to(device)

    def _(patch: np.ndarray):
        torch_patch = torch.tensor(patch / 255.0, dtype=torch.float32)
        if len(torch_patch.shape) == 3:
            torch_patch = torch_patch.unsqueeze(0)
        with torch.no_grad():
            embedding = model(torch_patch.to(device)).squeeze()
        return embedding.cpu()

    return _, model.output_dim


def embed_wsi_patches(
    sdata: SpatialData,
    model_name: str,
    magnification: int,
    patch_width: int,
    image_key: str | None = None,
    batch_size: int = 32,
    num_workers: int = 1,
    device: str = "cpu",
) -> bool:
    """Create an image made of patch embeddings of a WSI image.

    !!! info
        The image will be saved into the `SpatialData` object with the key `sopa_{model_name}` (see the argument below).

    Args:
        sdata: A `SpatialData` object
        model_name: Name of the computer vision model to be used. One of `Resnet50Features`, `HistoSSLFeatures`, or `DINOv2Features`.
        magnification: The target magnification.
        patch_width: Width of the patches for which the embeddings will be computed.
        image_key: Optional image key of the WSI image, unecessary if there is only one image.
        batch_size: Mini-batch size used during inference.
        num_workers: Number of workers used to extract patches.
        device: Device used for the computer vision model.

    Returns:
        `True` if the embedding was successful, else `False`
    """
    image_key = get_key(sdata, "images", image_key)
    image = sdata.images[image_key]

    assert isinstance(
        image, MultiscaleSpatialImage
    ), "Only `MultiscaleSpatialImage` images are supported"

    tiff_metadata = image.attrs["metadata"]
    coordinate_system = get_intrinsic_cs(sdata, image)

    embedder, output_dim = embed_batch(model_name=model_name, device=device)

    level, resize_factor, patch_width, success = _get_extraction_parameters(
        tiff_metadata, magnification, patch_width
    )
    if not success:
        log.error(f"Error retrieving the mpp for {image_key}, skipping tile embedding.")
        return False

    patches = Patches2D(sdata, image_key, patch_width, 0)
    embedding_image = np.zeros((output_dim, *patches.shape), dtype=np.float32)

    log.info(f"Computing {len(patches)} embeddings at level {level}")

    for i in tqdm.tqdm(range(0, len(patches), batch_size)):
        patch_boxes = patches[i : i + batch_size]

        get_batches = [
            da.delayed(_numpy_patch)(image, box, level, resize_factor, coordinate_system)
            for box in patch_boxes
        ]
        batch = da.compute(*get_batches, num_workers=num_workers)
        embedding = embedder(np.stack(batch))

        loc_x, loc_y = np.array([patches.patch_iloc(k) for k in range(i, i + len(batch))]).T
        embedding_image[:, loc_y, loc_x] = embedding.T

    embedding_image = SpatialImage(embedding_image, dims=("c", "y", "x"))
    embedding_image = Image2DModel.parse(
        embedding_image,
        transformations={coordinate_system: Scale([patch_width, patch_width], axes=("x", "y"))},
    )
    embedding_image.coords["y"] = patch_width * embedding_image.coords["y"]
    embedding_image.coords["x"] = patch_width * embedding_image.coords["x"]

    embedding_key = f"sopa_{model_name}"
    sdata.add_image(embedding_key, embedding_image)

    log.info(f"WSI embeddings saved as an image in sdata['{embedding_key}']")

    return True
