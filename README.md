([Français](#plugin-treeaibox-pour-cloudcompare))

# <img src="https://github.com/NRCan/TreeAIBox/blob/b0918d50b1343d62e1b518c29cd50d38f801b359/treeaibox-header.bmp" alt="treeaibox_logo" width="60"/> TreeAIBox

A Python library for LiDAR-based forest analysis: tree detection, segmentation, and structure modeling using deep learning.

TreeAIBox provides a clean Python API that works standalone, with CloudComPy, or in Jupyter notebooks. It also supports use as a CloudCompare Python plugin with its original GUI.

## Overview

TreeAIBox brings together four core LiDAR-processing workflows:

- **TreeFiltering** — Supervised deep-learning filtering to separate understory and overstory points.
- **TreeisoNet** — End-to-end crown segmentation pipeline (StemCls, TreeLoc, TreeOff, CrownOff3D).
- **WoodCls** — 3D stem & branch classification on TLS data.
- **QSM** — Plot-level skeletonization and export of tree structure to XML/OBJ.
- **UrbanFiltering** — Classification of urban scenes into 7 categories (ground, vegetation, vehicles, powerlines, fences, poles, buildings).

## Features

- **20+ pretrained AI models** — Downloadable automatically; lightweight/distilled versions fine-tuned with annotated datasets.
- **3D targeted** — Operates directly on raw 3D point clouds using voxel-based AI architectures.
- **Multiple sensors** — Supports TLS, ALS, and UAV LiDAR across boreal, mixedwood, reclamation, and urban scenes.
- **GPU/CPU toggle** — Runs on either GPU (CUDA) or CPU.
- **Standalone Python API** — No CloudCompare or GUI required. Works in scripts, notebooks, and pipelines.
- **CloudComPy bridge** — Optional functions to convert between CloudComPy clouds and numpy arrays.
- **DBSCAN fallback** — Crown clustering works without `cut_pursuit_py` using a DBSCAN-based alternative.

## Installation

### Standalone Python API (pip)

```bash
git clone https://github.com/NRCan/TreeAIBox.git
cd TreeAIBox
pip install -e ".[viz]"
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate treeaibox
pip install -e ".[viz]"
```

### Dependencies

Core dependencies (installed automatically):
- numpy, scipy, scikit-learn, scikit-image
- torch (PyTorch)
- laspy[lazrs]
- numpy-indexed, numpy-groupies
- timm, requests, tqdm

Optional:
- `matplotlib` — for visualization (`pip install -e ".[viz]"`)
- `circle-fit` — for QSM radius fitting (`pip install -e ".[qsm]"`)
- `cut_pursuit_py` — for graph-based crown clustering (DBSCAN fallback available)
- CloudComPy — for CloudCompare integration (install via conda)

### CloudCompare Plugin (Windows)

See the [releases page](https://github.com/NRCan/TreeAIBox/releases) for the Windows installer.

## Python API Usage

```python
import numpy as np
from treeaibox import tree_filtering, tree_location, post_peak_extraction, tree_offset
from treeaibox_io import load_point_cloud, save_point_cloud

# Load a point cloud
data = load_point_cloud("forest.laz")
points = data["points"]  # (N, 3) numpy array

# Step 1: Filter ground/vegetation
labels = tree_filtering(points, "treefiltering_als_esegformer3D_128_15cm(GPU3GB)")

# Step 2: Detect tree locations
preds = tree_location(points, "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)")
tree_tops = post_peak_extraction(preds, K=5, nms_thresh=0.3)

# Step 3: Segment individual trees
tree_ids = tree_offset(points, tree_tops[:, :3],
                       "treeisonet_tls_boreal_treeoff_esegformer3D_128_10cm(GPU3GB)")

# Save results
save_point_cloud("output.laz", points, fields={"treefilter": labels, "tree_id": tree_ids})
```

### Available Functions

| Function | Description |
|----------|-------------|
| `tree_filtering()` | Ground/vegetation separation using deep learning |
| `urban_filtering()` | Urban scene classification (7 classes) |
| `wood_classification()` | Stem vs branch classification |
| `stem_classification()` | Stem point detection |
| `tree_location()` | Tree top or stem base detection |
| `post_peak_extraction()` | Extract tree locations from predictions via NMS |
| `tree_offset()` | Assign crown points to trees via offset prediction |
| `crown_offset()` | Refine crown segmentation with 3D offsets |
| `stem_clustering()` | Cluster stem points to individual trees |
| `crown_clustering()` | Graph-based crown clustering |
| `init_segmentation()` | Initialize QSM branch/stem segmentation |
| `apply_qsm()` | Reconstruct tree skeleton (QSM) |
| `create_dtm()` | Generate Digital Terrain Model |
| `tree_statistics()` | Compute per-tree height, crown area, centroid |
| `clean_small_clusters()` | Remove small isolated point clusters |

### I/O Functions

```python
from treeaibox_io import load_point_cloud, save_point_cloud

# LAS/LAZ files
data = load_point_cloud("input.laz")
save_point_cloud("output.laz", data["points"], fields={"label": labels})

# Text files
data = load_point_cloud("input.txt")
save_point_cloud("output.xyz", data["points"])
```

### CloudComPy Bridge

```python
from treeaibox_io import cloudcompy_to_numpy, numpy_to_cloudcompy

# Convert CloudComPy cloud to numpy
data = cloudcompy_to_numpy(cc_cloud)
points = data["points"]

# Process with TreeAIBox
labels = tree_filtering(points, model_name)

# Convert back
result_cloud = numpy_to_cloudcompy(points, fields={"labels": labels})
```

### Model Management

```python
from treeaibox_models import list_available_models, download_model

# List all models
models = list_available_models()

# Pre-download a model
download_model("treefiltering_als_esegformer3D_128_15cm(GPU3GB)")
```

Models are automatically downloaded on first use and cached in `~/.treeaibox/models/`.

### Visualization (Jupyter)

```python
from treeaibox_viz import plot_point_cloud, plot_tree_locations, plot_classification, plot_dtm

plot_point_cloud(points, labels=tree_ids, title="Tree Segmentation")
plot_tree_locations(points, tree_tops)
plot_classification(points, labels, class_names={1: "Ground", 2: "Vegetation"})
```

### Progress Callbacks

All processing functions accept an optional `progress_callback`:

```python
def my_callback(percent):
    print(f"Progress: {percent}%")

labels = tree_filtering(points, model_name, progress_callback=my_callback)
```

## Model Table

The table below summarizes the voxel resolution and GPU memory used by the current AI models:

<table>
  <thead>
    <tr>
      <th align="center">Sensor</th>
      <th align="center">Task</th>
      <th align="center">Component</th>
      <th align="center">Scene</th>
      <th align="center">Resolution</th>
      <th align="center">VRAM</th>
    </tr>
  </thead>
  <tbody>
    <!-- ALS Classification -->
    <tr>
      <td align="center" rowspan="4"><strong>ALS (or UAV without stems)</strong></td>
      <td align="center" rowspan="4">Classification</td>
      <td align="center" rowspan="3">Vegetation layer</td>
      <td align="center">Mountainous</td>
      <td align="center">80 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Regular</td>
      <td align="center">50 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Wellsite</td>
      <td align="center">15 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Urban layers</td>
      <td align="center">Urban</td>
      <td align="center">30 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- UAV Classification -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (with stems)</strong></td>
      <td align="center" rowspan="2">Classification</td>
      <td align="center">Vegetation layer</td>
      <td align="center">Regular</td>
      <td align="center">12 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Stems</td>
      <td align="center">Mixedwood</td>
      <td align="center">8 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- TLS Classification -->
    <tr>
      <td align="center" rowspan="7"><strong>TLS</strong></td>
      <td align="center" rowspan="7">Classification</td>
      <td align="center">Vegetation layer</td>
      <td align="center">Regular</td>
      <td align="center">8 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center" rowspan="3">Stems</td>
      <td align="center" rowspan="3">Boreal</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">4 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">20 cm</td>
      <td align="center">8 GB</td>
    </tr>
    <tr>
      <td align="center">Stems</td>
      <td align="center">Regular</td>
      <td align="center">4 cm</td>
      <td align="center">12 GB</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Stems + branches</td>
      <td align="center" rowspan="2">Regular</td>
      <td align="center">4 cm</td>
      <td align="center">2 GB</td>
    </tr>
    <tr>
      <td align="center">2.5 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- ALS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>ALS (or UAV without stems)</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree tops</td>
      <td align="center">Wellsite</td>
      <td align="center">10 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Wellsite</td>
      <td align="center">10 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <!-- UAV Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (with stems)</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree bases</td>
      <td align="center">Mixedwood</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Mixedwood</td>
      <td align="center">15 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <!-- TLS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>TLS</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree bases</td>
      <td align="center">Boreal</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Boreal</td>
      <td align="center">15 cm</td>
      <td align="center">4 GB</td>
    </tr>
  </tbody>
</table>

## Folder Structure

```
TreeAIBox/
    treeaibox.py              # Main Python API
    treeaibox_io.py           # I/O utilities (LAS/LAZ, CloudComPy bridge)
    treeaibox_models.py       # Model download and management
    treeaibox_viz.py          # Matplotlib visualization for Jupyter
    pyproject.toml            # Package configuration
    environment.yml           # Conda environment specification
    model_zoo.json            # List of available pre-trained models
    TreeAIBox.py              # Original CloudCompare plugin (GUI)
    treeaibox_ui.html         # Original web-style GUI
    examples/
        example_notebook.ipynb  # Jupyter notebook tutorial
    tests/
        test_treeaibox.py     # Unit tests
    modules/
        filter/               # TreeFiltering, WoodCls, UrbanFiltering
            componentFilter.py
            createDTM.py
            vox3DESegFormer.py
            vox3DSegFormer.py
            *.json
        treeisonet/           # TreeisoNet pipeline
            treeLoc.py
            treeOff.py
            stemCluster.py
            crownCluster.py
            crownOff.py
            treeStat.py
            cleanSmallerClusters.py
            vox3DSegFormerDetection.py
            vox3DSegFormerRegression.py
            *.json
        qsm/                  # QSM module
            applyQSM.py
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## How to Contribute

1. Fork, create a feature branch, submit a PR.
2. Follow existing style and add tests as needed.

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Unless otherwise noted, the source code of this project is covered under Crown Copyright, Government of Canada, and is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE.txt).

The Canada wordmark and related graphics associated with this distribution are protected under trademark law and copyright law. No permission is granted to use them outside the parameters of the Government of Canada's corporate identity program. For more information, see [Federal identity requirements](https://www.canada.ca/en/treasury-board-secretariat/topics/government-communications/federal-identity-requirements.html).

Developed by Zhouxin Xi, tested by Charumitha Selvaraj

*Born from over a decade of LiDAR research with support from dedicated collaborators.*
