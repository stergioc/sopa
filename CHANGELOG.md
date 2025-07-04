## [2.1.1] - xxxx-xx-xx


## [2.1.0] - 2025-06-27

### Added
- Add `no_overlap` argument in `sopa.aggregate` to avoid cells from overlapping when aggregating the channels/bins
- Map each VisiumHD bin to one unique cell using PCA proximity (see `no_overlap` argument)
- Better documentation for `sopa.io.visium_hd` and a warning if the full res image is not loaded (#254)
- Support `CONCH` for H&E patches inference.
- Support `cellpose>=4.0.0` @lguerard (#252, #264)

### Changed
- Use the `global` coordinate system by default in the remaining readers that were still using the `pixels` coordinate system
- Default to `prior_shapes_key: auto` in all Snakemake config - it will automatically find the right key based on the technology
- To use cellpose with GPU, `gpu=True` must be passed directly as an arg/kwarg instead of inside `cellpose_eval_kwargs`, or via `--gpu` for the CLI, or via adding `gpu: true` to the Snakemake config (under the cellpose section).
- (Internal) use `disk` from `skimage` for opening/closing in `sopa.segmentation.tissue`
- (Internal) refactor `Patches2D` to make it faster when the ROI is complex with 100,000+ shapes

### Fixed
- Fixed report (transcript section) when `adata.X` is not sparse + add spatial count distribution
- Support `x/y_global_mm` for transcripts in the CosMx reader (#274)

### Removed
- Removed the `open-cv` dependency (#239)
- Removed all deprecated functions that were announced to be removed in this version

## [2.0.7] - 2025-05-19

### Added
- `sopa.patches.compute_embeddings` returns `key_added` for conveniency
- Added `sopa.io.bioio` generic reader

### Fixed
- Pin `cellpose<4.0.0` (#252)
- Using bounding boxes center instead of the shape centroids for patches location in `adata.obsm` after using `sopa.patches.compute_embeddings`
- Force sopa version in Docker images CI @Clemsazert (#251)
- CosmX reader fix when only 4 channels are used instead of 5 @professor-sagittarius (#258)

## [2.0.6] - 2025-04-24

### Added
- New Resolve Bioscience reader `sopa.io.molecular_cartography` (#240)
- Adding `roi_key` argument in `sopa.patches.compute_embeddings` to filter the patches by any shapes element (not just the segmented tissue). For instance, keep only the patches behind the cells.
- [Docker images](https://hub.docker.com/r/quentinblampey/sopa) auto build on tag release (#242) @Clemsazert

### Fixed
- When installing the `stardist` extra, force `numpy<2.0.0` (#245)

## [2.0.5] - 2025-04-24

Yanked release (missing dask distributed, cannot install)

## [2.0.4] - 2025-04-08

### Added
- Add `prior_shapes_key="auto"` to automatically detect the prior proprietary segmentation when making transcript patches
- Direct stardist CLI/snakemake support (not just via `generic-staining`)
- Added a snakemake config for Visium HD data (with stardist segmentation)
- Proseg support for Snakemake
- CLI command `sopa --version` to show the version of Sopa

### Fixed
- Fix CosMX reader issues related to the channel names, FOV names, and image flipping (#180, #227, #231)
- Fix `expand_radius_ratio=None` usage for bins aggregation (#226)
- Stardist: use `clip=True` when normalizing images to avoid `-1e20` values

### Changed
- Snakemake pipeline refactoring to better support the increasing number of segmentation tools
- Tangram now has to be installed via `pip install tangram-sc` rather than via the sopa extra

## [2.0.3] - 2025-03-13

### Added
- Experimental support of [proseg](https://github.com/dcjones/proseg) @lguerard (#223) - tutorial coming soon
- Use symlink to Xenium output morphology/transcripts files to avoid duplicating data (#221)
- Run `module load baysor` in Snakemake pipeline if the module is available.

### Fixed
- Use `sdata.path.resolve()` to compute the cache dir (more robust to execution path change)

### Changed
- Using `density_prior = "uniform"` by default for Tangram (#174)
- [`spatialdata_plot`](https://spatialdata.scverse.org/projects/plot/en/latest/index.html) is now a default dependency of Sopa
- Use `prob_thresh=0.2` and `nms_thresh=0.6` by default in `stardist`
- During segmentation, pixels outside of the ROI / tissue use the mean channels value instead of 0 (#222)

## [2.0.2] - 2025-02-21

### Added
- Added H-optimus-0 model for H&E patches embeddings @stergioc (#208)
- Can provide `qv_threshold` argument to the Xenium reader to filter low quality transcripts @callum-jpg (#210)

### Changed
- Storing patches embeddings as an `AnnData` object instead of images (#212) (see [updated tuto](https://gustaveroussy.github.io/sopa/tutorials/he/))

### Fixed
- Right sorting scales for WSI reader with openslide backend @stergioc (#209)
- When a polygon cannot be simplified (for the Xenium Explorer), convert it to a circle (#206)
- `sopa explorer write`: use correct default argument for `table_key` in the CLI (#211)
- Fix Baysor usage on Windows (#185)
- Fix tight patches returning a different number of patches (#214)

## [2.0.1] - 2025-02-10

### Fixed
- Safer check dataframe series is of integer dtype (#179)
- Ensure `feature_key` is converted correctly to a string (#185)
- Fixed the WSI readers @stergioc (#192)
- Fixed `points_key` usage in `sopa.aggregate` (#194)

### Added
- Aggregation and segmentation now exclude non-interesting gene names (e.g., "blank", "unassigned", ...) (#144)
- Can filter low-quality transcript for transcript-based segmentation (#79)
- Possibility to choose the table name for the report (#183)
- Possibility to choose the table name for `sopa.io.explorer.write` (#183)
- Can set all `spatialdata_io.xenium` arguments in `sopa.io.xenium`
- CLI for stardist @jeffquinn-msk (#189)
- Baysor logs if running on one patch, and return the right error code in CLI @jeffquinn-msk (#199)
- Baysor parallelization per patch @Marius1311 (#203)

### Changed
- `sopa.io.write_report` is copying the adata to avoid modifying it (#196)

## [2.0.0] - 2025-01-20

This version introduces many new API features but also some breaking changes; check [our migration guide](https://github.com/gustaveroussy/sopa/discussions/138) to smoothly update your code base.

### Added
- Full Visium HD support (including, notably, bins aggregation)
- Dask parallelization backend for faster segmentation (useful for API users). This can be done via `sopa.settings.parallelization_backend = 'dask'`. More details in the FAQ.
- Support for one-line segmentation, e.g. `sopa.segmentation.cellpose(sdata, ...)` or `sopa.segmentation.baysor(sdata, ...)`. This will implicitly use the selected parallelization backend.
- Spatial elements keys are saved in `sdata.attrs` so that Sopa knows automatically which element should be used in which function. It is still possible to precise `image_key` / `points_key` / `shapes_key` if needed. More details in the FAQ.
- Allows changing the auto-save settings (i.e. decide if spatial elements are saved on-disk automatically or not). This can be done via `sopa.settings.auto_save_on_disk = False`.
- Can recover a failed/stopped segmentation when using the API (and also `force` a segmentation, for Baysor)
- Better cache handling (invisible to API users)
- New tissue segmentation for non-H&E data (`sopa.segmentation.tissue`)
- Full support for `baysor>=0.7.0`
- Added `Stardist` segmentation (mainly used for H&E data)
- Added support for Python 3.12

### Changes
- The `sopa.io.uniform` dataset is now deprecated (use `sopa.io.toy_dataset` instead)
- API: The image patches are now called `sdata["image_patches"]` instead of `sdata["sopa_patches"]`

### Breaking changes
- Drop support for Python 3.9
- API: `sopa.segmentation.Patches2D` is deprecated. Instead, use the functions `sopa.make_image_patches` or `sopa.make_transcript_patches`
- API: Use `sopa.overlay_segmentation` instead of `sopa.segmentation.overlay_segmentation`
- API: The `Aggregator` class has been replaced by a simple function wrapper: `sopa.aggregate`
- API: The annotations methods are moved to the utils. For instance, use `sopa.utils.higher_z_score` instead of `sopa.annotation.higher_z_score`
- CLI: `sopa read` has been renamed `sopa convert` to avoid confusion.
- CLI: during segmentation, use `--cache-dir-name` instead of `--patch-dir`
- Drop support for Python 3.9

### Fixes
- Snakemake path issue on Windows (#64)
- Issues related to incompatible versions of Baysor


## [1.1.6] - 2024-11-29

### Fix
- Support `baysor>=0.7.0` (#125, @lguerard).
  - NB: For Snakemake, please remove the `new_component_*` arguments from the Baysor config.
- Use `DataTree` from `xarray` - use new spatialdata version (#159)

## [1.1.5] - 2024-09-17

### Fix
- Accept `object` dtype for channel names (#114)

### Changed
- Update MACSima reader to read the channel names of the latest file format

## [1.1.4] - 2024-08-21

### Hotfix
- Fixed Baysor issue on MERSCOPE data with the Vizgen prior
- Fix patch-maker issue due to new API temporary implementation


## [1.1.3] - 2024-08-18

### Fix
- Fixed aggregation issue when gene names are `NaN` or `None` (#101)
- Fix Xenium reader for old Xenium data format (#105)

### Added
- Support multipolygons in ROI rasterization
- Added bins aggregation
- Added Visium HD reader (tutorial coming soon)

### Changed
- Import submodules in init (segmentation, io, utils)
- API simplification in progress (new API + tutorial coming soon)

## [1.1.2] - 2024-07-24

### Fix
- Convert intensities values in integer for the `ome_tif` and `aicsimageio` readers
- Fix cellpose `pretrained_model` weights unused (@pakiessling, #90)
- Prevent spillover during image preprocessing before segmentation (@pakiessling, #90)

### Added
- Blur and CLAHE can be disabled by setting the parameter to 0 (@pakiessling, #90)
- Added an optional parameter clahe_kernel_size for skimage.exposure.equalize_adapthist (@pakiessling, #90)
- Check that the image has an integer dtype before segmentation (better error log #92)

## [1.1.1] - 2024-07-05

### Added
- Support Xenium multimodal segmentation as a prior for Baysor (#80)
- For snakemake, you can set a `BAYSOR_EXECUTABLE_PATH` environment variable to indicate the path of the Baysor executable
- Added [ComSeg](https://github.com/fish-quant/ComSeg) segmentation by @tdefa (#76)

### Fix
- Fix Xenium reader issue for recent machine versions (#80)
- Fix type issue (`DataTree` and `DataArray`) related to `spatialdata>=0.2.0` (#85)
- Fix `sjoin` issue related to `geopandas>=1.0.0`

### Changed
- Fully depends on `spatialdata-io` for the MERSCOPE and the Xenium reader
- Use `DataArray` and `DataTree` typing instead of (Multiscale)SpatialImage (as in `spatialdata>=0.2.0`)

## [1.1.0] - 2024-06-11

First post-publication release

### Changed
- Using `rioxarray` as a default backend for MERSCOPE data if installed
- Lower RAM usage for channels aggregation
- Transcript-segmentation API more general (not Baysor-specific)

### Fixed
- Encoding issue while writing the report (#64)

## [1.0.14] - 2024-04-25

### Changed
- Renamed `embed_wsi_patches` to `infer_wsi_patches`
- `infer_wsi_patches` now accepts also callables
- Improves tile gathering speed and decreases overall computation of tile-wise inference

## [1.0.13] - 2024-04-22

### Changed
- Xenium reader now adds channel names, and support more recent versions (#68)
- Renamed `sopa.embedding` into `sopa.patches`, and moved internal files
- Don't recompute `to_multiscale` if the right scales are already used for Xenium Explorer image writing

### Added
- New tutorial on Xenium Explorer interoperability

## [1.0.12] - 2024-05-17

### Fix
- Fix polygon selection when no channel is provided
- Fix CosMX reader for proteins
- Fix FOV column issue for CosMX data (#65)

### Added
- Check the columns of CosMX data to see if the correct export module was used

### Changed
- Ensure categorical variables are used for patches clustering

## [1.0.11] - 2024-04-26

### Added
- Can overlay a custom segmentation (merge boundaries)
- Xenium Explorer selection(s) can be added as shapes in a SpatialData object
- Optionnal OpenSlide backend for WSI data
- New `sopa.io.aicsimageio` reader for special formats (#58)

### Changed
- Rename `Aggregator.update_table` to `Aggregator.compute_table`

## [1.0.10] - 2024-04-08

### Added
- CosMX reader with image stitching (experimental)

### Changed
- Default `min_transcripts` set in snakemake configs
- Minimum number of transcripts per patch set to 4000 (#41)
- Config files refactoring (configs added or renamed)
- Readers refactoring
- Section with error during report are not displayed (instead of throwing an error)

## [1.0.9] - 2024-04-03

### Added:
- Support multiple tables

### Fixed
- Spatial elements not saved when data is not backed

## [1.0.8] - 2024-04-02

Hotfix: resolve issues related to `spatialdata>=1.0.0`

## [1.0.7] - 2024-03-29

### Changed
- Improvements in the CLI and API tutorials
- Sequential segmentation now requires `patchify` to be run independently
- Dependency `spatialdata>=0.1.1`

### Added
- Kwargs can be provided to Cellpose model init

### Fixed
- `set_transformation` issue for image alignment
- Import issue #37 #39

## [1.0.6] - 2024-03-13

### Added
- Spatial join between shapes (`from sdata.spatial import sjoin`)
- H&E tutorial (basic usage)
- New backend for the MERSCOPE reader (requires `rioxarray`, currently experimental, should use less RAM)

### Changed
- Using `MultiscaleSpatialImage` by default for multiplex imaging technologies

### Fixed
- Issue in report creation when channel names are integers

## [1.0.5] - 2024-03-01

### Changed
- Faster image writing for the Xenium Explorer (about x5 speedup)
- Cellpose default model set to `"cyto3"` (new cellpose version)
- Cell GeoDataFrame index consistent with `obs_names`

### Added
- Support for python 3.9 to 3.11 (we still recommend `python==3.10`)
- Support WSI analysis: reader, tissue segmentation, patch embedding (tutorials coming soon)
- Supporting multiple region-of-interest queries
- Can load a custom cellpose model using the `pretrained_model`/`model_type` argument

## [1.0.4] - 2024-02-14

### Fix
- Missing transcript count in cells due to concurrent writing processes (#20)

### Changed
- Explorer images should have a higher contrast (not clipping values anymore)

## [1.0.3] - 2024-02-12

### Breaking changes
- `pixelsize` argument has been renamed to `pixel_size` (the snakemake pipeline is only deprecating it for now)

### Added
- The `phenocycler` reader can now also read `.tif` files (not just `.qptiff`)
- Added missing legend in the HTML report under the "Channels" section (#15)
- The cell area is also stored in the table (in `.obs["area"]`)

### Changed
- The `uniform` toy dataset now has two coordinate systems (better test case)
- Faster table conversion to the Xenium Explorer

### Fixed
- Tight patching more stable with epsilon constant

## [1.0.2] - 2024-01-15

### Fix
- When geometries are `GeometryCollection`, convert them back to Polygons (#11)
- Give `min_area` parameter to the right Baysor function in snakemake

### Added
- API tutorial
- `sopa.spatial` tutorial
- Docstrings for the snakemake pipeline utils
- Show right micron scale in the Xenium Explorer

### Changed
- `sopa.stats` is now called `sopa.spatial`

## [1.0.1] - 2024-01-10

### Added

- Tutorial on CLI usage
- Tutorial on image alignment with the Xenium Explorer
- Multi-step segmentation ([#8](https://github.com/gustaveroussy/sopa/issues/8))
- Tutorial for multi-step segmentation and custom segmentation
- Improved installation guide

### Fix
- CLI issue (missing file) when used without Snakemake

### Change

- Use `.parquet` instead of `.zarr.zip` format to store intermediate boundary polygons

## [1.0.0] - 2023-12-26

### Added

- First official release
- Preprint at https://www.biorxiv.org/content/10.1101/2023.12.22.571863v1
