import numpy as np
from multiscale_spatial_image import MultiscaleSpatialImage
from spatialdata import bounding_box_query


def _get_best_level_for_downsample(
    level_downsamples: list[float], downsample: float, epsilon: float = 0.01
) -> int:
    """Return the best level for a given downsampling factor"""
    if downsample <= 1.0:
        return 0
    for level, ds in enumerate(level_downsamples):
        if ds > downsample + epsilon:
            return level - 1
    return len(level_downsamples) - 1


def _get_extraction_parameters(
    tiff_metadata: dict, magnification: int, patch_width: int
) -> tuple[int, int, int, bool]:
    """
    Given the metadata for the slide, a target magnification and a patch width,
    it returns the best scale to get it from (level), a resize factor (resize_factor)
    and the corresponding patch size at scale0 (patch_width)
    """
    if tiff_metadata["properties"].get("tiffslide.objective-power"):
        objective_power = int(tiff_metadata["properties"].get("tiffslide.objective-power"))
        downsample = objective_power / magnification

    elif tiff_metadata["properties"].get("tiffslide.mpp-x"):
        mppx = float(tiff_metadata["properties"].get("tiffslide.mpp-x"))

        mpp_objective = min([80, 40, 20, 10, 5], key=lambda obj: abs(10 / obj - mppx))
        downsample = mpp_objective / magnification
    else:
        return None, None, None, False

    level = _get_best_level_for_downsample(tiff_metadata["level_downsamples"], downsample)
    resize_factor = tiff_metadata["level_downsamples"][level] / downsample
    patch_width = int(patch_width * downsample)

    return level, resize_factor, patch_width, True


def _numpy_patch(
    image: MultiscaleSpatialImage,
    box: tuple[int, int, int, int],
    level: int,
    resize_factor: float,
    coordinate_system: str,
) -> np.ndarray:
    """Extract a numpy patch from the MultiscaleSpatialImage given a bounding box"""
    import cv2

    multiscale_patch = bounding_box_query(
        image, ("y", "x"), box[:2][::-1], box[2:][::-1], coordinate_system
    )
    patch = np.array(
        next(iter(multiscale_patch[f"scale{level}"].values())).transpose("y", "x", "c")
    )

    if resize_factor != 1:
        dim = (int(patch.shape[0] * resize_factor), int(patch.shape[1] * resize_factor))
        patch = cv2.resize(patch, dim)

    return patch.transpose(2, 0, 1)