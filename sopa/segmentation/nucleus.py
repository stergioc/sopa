import argparse
from typing import List

import cv2
import numpy as np
import xarray as xr
from cellpose import models
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

from ..io.xenium import save_polygons
from .tiling import TileMaker


def smooth(poly: Polygon, radius: int = 5) -> Polygon:
    smooth = poly.buffer(-radius).buffer(radius * 2).buffer(-radius)
    return poly if isinstance(smooth, MultiPolygon) else smooth


def extract_polygons(mask: np.ndarray) -> List[Polygon]:
    polys = []

    for cell_id in range(1, mask.max() + 1):
        mask_id = (mask == cell_id).astype("uint8")
        contours, _ = cv2.findContours(mask_id, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        polys_ = [smooth(Polygon(c[:, 0, :])) for c in contours if c.shape[0] >= 4]
        polys_ = [p for p in polys_ if not p.is_empty]

        if len(polys_) > 1:
            print(len(polys_))
        polys.extend(polys_)

    return polys


def pad(
    polygon: Polygon, min_vertices: int, max_vertices: int, tolerance: float = 1
) -> np.ndarray:
    n_vertices = len(polygon.exterior.coords)
    assert n_vertices >= min_vertices

    coords = polygon.exterior.coords._coords

    if n_vertices == max_vertices:
        return coords.flatten()

    if n_vertices < max_vertices:
        return np.pad(
            coords, ((0, max_vertices - n_vertices), (0, 0)), mode="edge"
        ).flatten()

    polygon = polygon.simplify(tolerance=tolerance)
    return pad(polygon, min_vertices, max_vertices, tolerance + 1)


def patch_coordinates(
    xarr: xr.DataArray,
    model: models.CellposeModel,
    x_bounds: List[int],
    y_bounds: List[int],
    c: str,
    min_vertices: int = 3,
    max_vertices: int = 13,
) -> np.ndarray:
    patch = xarr.sel(
        c=c,
        x=slice(*x_bounds),
        y=slice(*y_bounds),
    ).values

    masks, *_ = model.eval(
        [patch],
        diameter=15,
        channels=[[0, 0]],
        flow_threshold=2,
        cellprob_threshold=-6,
    )

    polygons = extract_polygons(masks[0])

    coordinates = np.stack([pad(p, min_vertices, max_vertices) for p in polygons])
    coordinates[:, ::2] += x_bounds[0]
    coordinates[:, 1::2] += y_bounds[0]
    coordinates /= 4.705882
    return coordinates


def main(args):
    xarr = xr.open_zarr(args.path)["image"]
    model = models.Cellpose(model_type="cyto2")
    tile_maker = TileMaker(0, xarr.shape[2], 0, xarr.shape[1], args.width)

    coordinates = np.concatenate(
        [
            patch_coordinates(xarr, model, x_bounds, y_bounds, args.dapi)
            for x_bounds, y_bounds in tqdm(tile_maker)
        ],
        axis=0,
    )

    save_polygons(args.output, coordinates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the zarr image",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output zip file",
    )
    parser.add_argument(
        "-d",
        "--dapi",
        type=str,
        required=True,
        help="DAPI channel name",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=5000,
        help="Tile width",
    )

    main(parser.parse_args())
