# TreeAIBox Model & Function Decision Tree

A guide for choosing the right model and function for your LiDAR point cloud analysis task.

---

## 1. Architecture: SegFormer vs ESegFormer

Both are 3D voxel-based transformer architectures for semantic segmentation of point clouds.

**SegFormer3D** is the standard architecture. It uses full-resolution attention with larger channel dimensions and deeper transformer blocks. It produces high-quality results but requires more GPU memory (4-12 GB). Model names contain `segformer3D`.

**ESegFormer3D** (Efficient SegFormer) is a memory-optimized variant. Key differences:
- Uses `DyT` (Dynamic Tanh) normalization instead of LayerNorm, reducing memory overhead
- Uses `EfficientAttention` which leverages PyTorch's `scaled_dot_product_attention` and spatial reduction via strided convolutions
- Reduces MLP hidden dimensions by half internally
- Achieves comparable accuracy with significantly lower GPU memory (typically 3 GB)

Model names contain `esegformer3D`. **For most users, ESegFormer models are recommended** since they fit on consumer GPUs while maintaining quality.

**How to tell which architecture a model uses:** The model name encodes it directly:
```
treefiltering_als_esegformer3D_128_15cm(GPU3GB)
                 ^^^^^^^^^^^
                 architecture
```

---

## 2. Model Name Convention

Every model name follows this pattern:

```
{task}_{sensor}_{scene}_{subtask}_{architecture}_{block_size}_{resolution}({gpu_req})
```

| Component | Values | Meaning |
|-----------|--------|---------|
| task | `treefiltering`, `urbanfiltering`, `woodcls`, `treeisonet` | Top-level task family |
| sensor | `als`, `tls`, `uav` | Sensor platform |
| scene | `boreal`, `mixedwood`, `reclamation` | Training dataset biome (treeisonet only) |
| subtask | `stem`, `branch`, `treeloc`, `treeoff`, `stemcls`, `crownoff` | Specific operation |
| architecture | `segformer3D`, `esegformer3D` | Model architecture |
| block_size | `128`, `112` | Voxels per block dimension |
| resolution | `2.5cm` to `80cm` | Voxel size in meters |
| gpu_req | `GPU2GBDistilled` to `GPU12GB` | Minimum VRAM |

---

## 3. Decision Tree: Which Model and Function Do I Use?

### Step 1: What is your sensor platform?

```
Your sensor platform?
 |
 +-- ALS (Airborne Laser Scanning) --> Go to Step 2A
 +-- TLS (Terrestrial Laser Scanning) --> Go to Step 2B
 +-- UAV (Unmanned Aerial Vehicle / Drone LiDAR) --> Go to Step 2C
```

### Step 2A: ALS Data

```
What do you want to do?
 |
 +-- Separate vegetation from ground
 |    |
 |    +-- Point spacing > 30 cm --> treefiltering_als_esegformer3D_128_80cm(GPU3GB)
 |    +-- Point spacing 15-30 cm --> treefiltering_als_esegformer3D_128_50cm(GPU3GB)
 |    +-- Point spacing < 15 cm  --> treefiltering_als_esegformer3D_128_15cm(GPU3GB)
 |    |
 |    Function: tree_filtering()
 |
 +-- Classify urban scene (7 classes: ground, vegetation, vehicles, powerlines, fences, poles, buildings)
 |    |
 |    Model: urbanfiltering_als_esegformer3D_112_30cm(GPU3GB)
 |    Function: tree_filtering()  (same function, different model)
 |
 +-- Detect individual tree locations
 |    |
 |    Model: treeisonet_als_reclamation_treeloc_esegformer3D_128_10cm(GPU4GB)
 |    Function: tree_location()  then  post_peak_extraction()
 |
 +-- Segment individual tree crowns
      |
      Step 1: Detect trees with tree_location()
      Step 2: Assign points to trees with tree_offset()
              Model: treeisonet_als_reclamation_treeoff_esegformer3D_128_10cm(GPU4GB)
      Step 3: Optionally refine with crown_clustering()
```

### Step 2B: TLS Data

```
What is the extent of your point cloud?
 |
 +-- LANDSCAPE / PLOT (multiple trees)
 |    |
 |    What do you want to do?
 |    |
 |    +-- Separate vegetation from ground
 |    |    |
 |    |    Model: treefiltering_tls_esegformer3D_128_8cm(GPU3GB)
 |    |    Function: tree_filtering()
 |    |
 |    +-- Detect tree stem base locations
 |    |    |
 |    |    Model: treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)
 |    |       or: treeisonet_tls_boreal_treeloc_segformer3D_128_8cm(GPU4GB)
 |    |    Function: tree_location(if_stem=True)
 |    |
 |    +-- Detect tree top locations
 |    |    |
 |    |    Same model as above
 |    |    Function: tree_location(if_stem=False) then post_peak_extraction()
 |    |
 |    +-- Segment individual trees (full pipeline)
 |         |
 |         See "Full TLS Segmentation Pipeline" below
 |
 +-- SINGLE TREE (one isolated tree)
      |
      What do you want to do?
      |
      +-- Classify stem vs branches
      |    |
      |    Stem classification:
      |      Model: woodcls_stem_tls_esegformer3D_128_4cm(GPU3GB)
      |         or: woodcls_stem_tls_esegformer3D_128_10cm(GPU3GB)
      |         or: woodcls_stem_tls_segformer3D_112_4cm(GPU12GB)    [highest quality]
      |         or: woodcls_stem_tls_segformer3D_112_20cm(GPU8GB)
      |      Function: tree_filtering()  (aliased as wood_classification())
      |
      |    Branch classification:
      |      Model: woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)
      |         or: woodcls_branch_tls_segformer3D_112_4cm(GPU6GB)
      |         or: woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled)
      |      Function: tree_filtering()
      |
      +-- Build a tree skeleton (QSM)
      |    |
      |    Step 1: Classify stem/branch with tree_filtering() using woodcls model
      |    Step 2: init_segmentation() to segment stem and branches
      |    Step 3: apply_qsm() to extract skeleton, branch paths, radii
      |
      +-- Remove noise / small clusters
           |
           Function: clean_small_clusters()
```

### Step 2C: UAV Data

```
What do you want to do?
 |
 +-- Separate vegetation from ground
 |    |
 |    Model: treefiltering_uav_esegformer3D_128_12cm(GPU3GB)
 |    Function: tree_filtering()
 |
 +-- Detect tree stem bases
 |    |
 |    Model: treeisonet_uav_mixedwood_stemcls_esegformer3D_128_8cm(GPU3GB)
 |    Function: tree_location(if_stem=True)
 |
 +-- Detect tree top locations
 |    |
 |    Model: treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)
 |    Function: tree_location(if_stem=False) then post_peak_extraction()
 |
 +-- Segment individual tree crowns
      |
      Step 1: Detect trees with tree_location()
      Step 2: Assign points with tree_offset()  (use ALS reclamation treeoff model)
      Step 3: Refine with crown_offset()
              Model: treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)
```

---

## 4. Common Pipelines

### 4.1 Ground/Vegetation Classification (any sensor)

```python
from treeaibox import load_point_cloud, tree_filtering, save_point_cloud

data = load_point_cloud("input.laz")
labels = tree_filtering(data["points"], model_name="treefiltering_als_esegformer3D_128_15cm(GPU3GB)",
                        use_cuda=False)
# labels: 1 = ground/understory, 2 = overstory vegetation
save_point_cloud("output.laz", data["points"], fields={"treefilter": labels})
```

### 4.2 Tree Detection + Crown Segmentation (ALS/UAV)

```python
from treeaibox import *

data = load_point_cloud("forest.laz")
points = data["points"]

# Step 1: Find tree tops
preds = tree_location(points, "treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)",
                      if_stem=False, use_cuda=False)
tree_tops = post_peak_extraction(preds, K=5, nms_thresh=0.3)

# Step 2: Assign each point to the nearest tree
tree_ids = tree_offset(points, tree_tops[:, :3],
                       "treeisonet_als_reclamation_treeoff_esegformer3D_128_10cm(GPU4GB)",
                       use_cuda=False)

# Step 3 (optional): Refine with crown offset
tree_ids = crown_offset(points, tree_ids,
                        "treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)",
                        use_cuda=False)
```

### 4.3 Full TLS Segmentation Pipeline

```python
from treeaibox import *

data = load_point_cloud("plot.laz")
points = data["points"]

# Step 1: Ground filtering
filter_labels = tree_filtering(points, "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)",
                               use_cuda=False)

# Step 2: Stem classification on vegetation points
veg_mask = filter_labels == 2
stem_cls = tree_filtering(points[veg_mask],
                          "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
                          use_cuda=False)

# Step 3: Detect stem bases
stem_locs = tree_location(points[veg_mask],
                          "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
                          if_stem=True, use_cuda=False)

# Step 4: Cluster stems to individual trees
tree_ids = stem_clustering(points[veg_mask], stem_cls, stem_locs)

# Step 5: Refine crown assignment
tree_ids = crown_offset(points[veg_mask], tree_ids,
                        "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
                        use_cuda=False)
```

### 4.4 QSM (Quantitative Structure Model) for a Single Tree

```python
from treeaibox import *
import numpy as np

data = load_point_cloud("single_tree.laz")
points = data["points"]

# Step 1: Classify wood (stem vs branch)
wood_labels = tree_filtering(points, "woodcls_stem_tls_esegformer3D_128_4cm(GPU3GB)",
                             use_cuda=False)

# Step 2: Initialize segmentation
pts_with_cls = np.column_stack([points, wood_labels])
init_segs = init_segmentation(pts_with_cls, K_stem=5, K_branch=10)

# Step 3: Build QSM skeleton
pts_full = np.column_stack([points, wood_labels, init_segs])
tree, centroids, seg_labels, radii = apply_qsm(pts_full, output_dir="qsm_output/")
# Outputs: tree_structure.xml and tree_mesh.obj
```

---

## 5. Complete Function Reference

### Deep Learning Functions (require a model)

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `tree_filtering()` | Classify points (ground/veg, wood, urban) | (N,3) points + model name | (N,) int labels |
| `tree_location()` | Detect tree tops or stem bases | (N,3) points + model name | (M,3) locations or (N,5) predictions |
| `tree_offset()` | Assign points to detected trees | (N,3) points + (M,3) tree locs + model | (N,) tree IDs |
| `crown_offset()` | Refine crown segmentation | (N,3) points + (N,) tree IDs + model | (N,) refined tree IDs |

### Post-Processing Functions (no model needed)

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `post_peak_extraction()` | Extract tree locations from prediction heatmap | (N,5) preds from tree_location | (M,4) [x,y,z,radius] |
| `stem_clustering()` | Cluster stem points to individual trees | points + stem labels + base locs | (N,) tree IDs |
| `crown_clustering()` | Graph-based crown segmentation | points + tree offset labels | (N,) tree IDs |
| `clean_small_clusters()` | Remove small noise clusters | (N,3) points | (N,) cluster labels (0=removed) |
| `init_segmentation()` | Segment stem/branches for QSM | (N,4) [x,y,z,stemcls] | (N,) segment labels |
| `apply_qsm()` | Build tree skeleton model | (N,5) [x,y,z,stemcls,segs] | skeleton, centroids, labels, radii |

### Utility Functions

| Function | Purpose |
|----------|---------|
| `load_point_cloud()` | Load .las/.laz/.txt/.csv/.xyz files |
| `save_point_cloud()` | Save point clouds with fields |
| `create_dtm()` | Generate Digital Terrain Model |
| `tree_statistics()` | Compute per-tree height, crown area, centroid |
| `list_available_models()` | Show all models in the model zoo |

### Aliases

These are identical to `tree_filtering()` but exist for readability:
- `stem_classification()` -- for stem classification models
- `wood_classification()` -- for wood classification models
- `urban_filtering()` -- for urban scene classification

---

## 6. Parameter Reference

### Common Parameters (all DL functions)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | np.ndarray | required | (N,3) XYZ coordinates |
| `model_name` | str | required | Model name from model zoo |
| `use_cuda` | bool | `True` | Use GPU. Set `False` if no CUDA-enabled GPU |
| `progress_callback` | callable | `None` | Called with int 0-100. Auto-creates tqdm bar if None |
| `auto_download` | bool | `True` | Download model weights if not cached locally |
| `check_classified` | bool | `True` | Warn and return None if points already have classification fields |
| `fields` | dict | `None` | Existing scalar fields from `load_point_cloud()["fields"]` (used by check_classified) |

### tree_filtering() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `if_bottom_only` | bool | `True` | `True`: slice point cloud into 2D (XY) blocks -- faster, works well for canopy-level classification. `False`: use 3D (XYZ) blocks -- needed when vertical structure matters (e.g., multi-story buildings) |

### tree_location() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `if_stem` | bool | `False` | `True`: detect stem base locations (returns (M,3) XYZ array). `False`: predict per-point confidence and crown radius (returns (N,5) array for post_peak_extraction) |
| `cutoff_thresh` | float | `1.0` | Fraction of tree height to include when detecting stems. `0.2` = only bottom 20% of each tile. `1.0` = full height. Lower values focus on trunk base |
| `custom_resolution` | np.ndarray | `None` | Override voxel resolution from config, e.g. `np.array([0.1, 0.1, 0.3])`. Use when your data density differs significantly from the training data |

### post_peak_extraction() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preds_tops` | np.ndarray | required | (N,5) array from `tree_location(if_stem=False)`. Columns: [x, y, z, confidence, radius] |
| `K` | int | `5` | Nearest neighbors for grouping nearby high-confidence points into clusters |
| `max_gap` | float | `0.3` | Maximum distance (meters) between points in the same cluster |
| `min_rad` | float | `0.2` | Minimum crown radius (meters) to keep. Filters out small false detections |
| `nms_thresh` | float | `0.3` | Non-maximum suppression threshold. Lower = more aggressive merging of nearby detections |

### tree_offset() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tree_locations` | np.ndarray | required | (M,3) XYZ array of detected tree locations (from tree_location or post_peak_extraction) |
| `custom_resolution` | np.ndarray | `None` | Override voxel resolution, same as tree_location |

### stem_clustering() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stem_cls` | np.ndarray | required | (N,) array where values > 1 indicate stem points |
| `base_locations` | np.ndarray | required | (M,3) XYZ of tree base locations |
| `min_res` | float | `0.06` | Voxel size for downsampling before graph construction. Smaller = more detail but slower |
| `max_isolated_distance` | float | `0.3` | Maximum gap (meters) allowed between connected nodes. Points farther apart are disconnected |

### crown_clustering() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `treeoff` | np.ndarray | required | (N,) tree offset labels from tree_offset() |
| `min_res` | float | `0.15` | Voxel resolution for graph decimation |
| `K` | int | `5` | Nearest neighbors for graph construction |
| `reg_strength` | float | `1.0` | Regularization for graph-cut segmentation. Higher = larger, smoother segments |
| `max_isolated_distance` | float | `0.3` | Max gap for connectivity |

### crown_offset() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stem_id` | np.ndarray | required | (N,) tree IDs from a prior step (0 = unassigned). The model refines these assignments |

### clean_small_clusters() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_gap` | float | `3.0` | Maximum distance for graph connectivity |
| `min_points` | int | `100` | Clusters with fewer points are removed |
| `min_res` | float | `0.03` | Voxel resolution for graph decimation |

### init_segmentation() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | np.ndarray | required | (N,4) array: [x, y, z, stemcls]. stemcls > 1 = stem |
| `K_stem` | int | `5` | Nearest neighbors for stem graph. Lower = finer stem segments |
| `reg_strength_stem` | float | `1.0` | Regularization for stem segmentation |
| `K_branch` | int | `10` | Nearest neighbors for branch graph. Higher because branches are sparser |
| `reg_strength_branch` | float | `1.0` | Regularization for branch segmentation |

### apply_qsm() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | np.ndarray | required | (N,5) array: [x, y, z, stemcls, init_segs] |
| `k_neighbors` | int | `6` | Nearest neighbors for skeleton graph |
| `max_graph_distance` | float | `40` | Maximum path length (meters) for branch tracing |
| `max_connectivity_search_distance` | float | `0.03` | Search distance for connecting graph components |
| `occlusion_distance_cutoff` | float | `0.4` | Distance threshold for handling occluded regions |
| `output_dir` | str | `None` | If set, saves `tree_structure.xml` and `tree_mesh.obj` to this directory |

### create_dtm() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | float | `1.0` | Grid cell size in meters |
| `ground_class` | np.ndarray | `None` | (N,) array where values <= 1 = ground points |
| `tile_size` | tuple | `None` | (x, y) tile dimensions for processing large datasets |
| `buffer_size` | tuple | `None` | (x, y) overlap buffer around each tile |

### tree_statistics() specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tree_ids` | np.ndarray | required | (N,) integer tree ID labels |
| `pcd_min` | np.ndarray | `None` | (3,) original coordinate minimum for restoring absolute positions |
| `treefilter` | np.ndarray | `None` | (N,) ground classification (<=1 = ground) for height normalization |
| `output_path` | str | `None` | Save results as CSV |
| `dtm_resolution` | float | `1.0` | Resolution for internal DTM generation |

---

## 7. Resolution Selection Guide

Match model resolution to your point cloud density:

| Point spacing | Recommended resolution | Example models |
|---------------|----------------------|----------------|
| < 3 cm | 2.5-4 cm | `woodcls_branch_tls_esegformer3D_128_2.5cm` |
| 3-8 cm | 4-8 cm | `treefiltering_tls_..._8cm`, `woodcls_stem_..._4cm` |
| 8-15 cm | 10-12 cm | `treeisonet_..._10cm`, `treefiltering_uav_..._12cm` |
| 15-40 cm | 15-30 cm | `urbanfiltering_..._30cm`, `crownoff_..._15cm` |
| > 40 cm | 50-80 cm | `treefiltering_als_..._50cm`, `..._80cm` |

**Rule of thumb:** The model voxel resolution should be roughly 1-3x your average point spacing. Too fine = sparse voxels with poor predictions. Too coarse = loss of spatial detail.

---

## 8. Available Models Summary

| Model Name | Sensor | Task | GPU |
|------------|--------|------|-----|
| `treefiltering_als_esegformer3D_128_15cm(GPU3GB)` | ALS | Ground/veg filter | 3 GB |
| `treefiltering_als_esegformer3D_128_50cm(GPU3GB)` | ALS | Ground/veg filter | 3 GB |
| `treefiltering_als_esegformer3D_128_80cm(GPU3GB)` | ALS | Ground/veg filter | 3 GB |
| `treefiltering_tls_esegformer3D_128_8cm(GPU3GB)` | TLS | Ground/veg filter | 3 GB |
| `treefiltering_uav_esegformer3D_128_12cm(GPU3GB)` | UAV | Ground/veg filter | 3 GB |
| `urbanfiltering_als_esegformer3D_112_30cm(GPU3GB)` | ALS | 7-class urban | 3 GB |
| `treeisonet_als_reclamation_treeloc_esegformer3D_128_10cm(GPU4GB)` | ALS | Tree location | 4 GB |
| `treeisonet_als_reclamation_treeoff_esegformer3D_128_10cm(GPU4GB)` | ALS | Tree offset | 4 GB |
| `treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)` | TLS | Stem classification | 3 GB |
| `treeisonet_tls_boreal_stemcls_esegformer3D_128_10cm(GPU3GB)` | TLS | Stem classification | 3 GB |
| `treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)` | TLS | Tree location | 3 GB |
| `treeisonet_tls_boreal_treeloc_segformer3D_128_8cm(GPU4GB)` | TLS | Tree location | 4 GB |
| `treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)` | TLS | Crown refinement | 4 GB |
| `treeisonet_uav_mixedwood_stemcls_esegformer3D_128_8cm(GPU3GB)` | UAV | Stem classification | 3 GB |
| `treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)` | UAV | Tree location | 3 GB |
| `treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)` | UAV | Crown refinement | 4 GB |
| `woodcls_stem_tls_esegformer3D_128_4cm(GPU3GB)` | TLS | Stem classification | 3 GB |
| `woodcls_stem_tls_esegformer3D_128_10cm(GPU3GB)` | TLS | Stem classification | 3 GB |
| `woodcls_stem_tls_segformer3D_112_4cm(GPU12GB)` | TLS | Stem classification | 12 GB |
| `woodcls_stem_tls_segformer3D_112_20cm(GPU8GB)` | TLS | Stem classification | 8 GB |
| `woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)` | TLS | Branch classification | 3 GB |
| `woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled)` | TLS | Branch classification | 2 GB |
| `woodcls_branch_tls_segformer3D_112_4cm(GPU6GB)` | TLS | Branch classification | 6 GB |
