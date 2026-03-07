"""TreeAIBox — Python API for LiDAR-based forest analysis.

A standalone Python library for tree detection, segmentation, and structural
analysis from LiDAR point clouds. Compatible with CloudComPy and Jupyter notebooks.

Usage::

    from treeaibox import tree_filtering, load_point_cloud, save_point_cloud

    data = load_point_cloud("forest.laz")
    labels = tree_filtering(data["points"], model_name="treefiltering_als_esegformer3D_128_15cm(GPU3GB)")
    save_point_cloud("output.laz", data["points"], fields={"treefilter": labels})
"""

import sys
import os
import warnings
import numpy as np

from treeaibox_models import get_model_path, get_config_path, list_available_models
from treeaibox_io import load_point_cloud, save_point_cloud

_CLASSIFICATION_FIELD_NAMES = {
    "classification", "class", "class_i", "treefilter",
    "label", "labels", "tree_id", "stemcls",
}


def _check_existing_classification(fields, func_name):
    """Check if fields contain existing classification data.

    Returns True (and warns) if classification fields are found.
    """
    if not fields:
        return False
    found = [k for k in fields if k.lower() in _CLASSIFICATION_FIELD_NAMES]
    if found:
        warnings.warn(
            f"{func_name}: Points already have classification fields: {found}. "
            "Set check_classified=False to run anyway."
        )
        return True
    return False


def _make_progress_callback(progress_callback=None):
    """Create a progress callback, using tqdm if none provided and in a notebook."""
    if progress_callback is not None:
        return progress_callback

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=100, desc="Processing", unit="%")
        last_val = [0]

        def _tqdm_callback(val):
            delta = val - last_val[0]
            if delta > 0:
                pbar.update(delta)
                last_val[0] = val
            if val >= 100:
                pbar.close()

        return _tqdm_callback
    except ImportError:
        return lambda x: None


# ---------------------------------------------------------------------------
# TreeFiltering / UrbanFiltering / WoodCls / StemCls
# ---------------------------------------------------------------------------

def tree_filtering(points, model_name, use_cuda=True, if_bottom_only=True,
                   progress_callback=None, auto_download=True,
                   check_classified=True, fields=None):
    """Classify points using a deep learning filter model.

    Works for TreeFiltering, UrbanFiltering, WoodCls, and StemCls models.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    model_name : str
        Model name from model_zoo.json (e.g. "treefiltering_als_esegformer3D_128_15cm(GPU3GB)").
    use_cuda : bool
        Use GPU acceleration if available.
    if_bottom_only : bool
        If True, use 2D blocks (faster). If False, use 3D blocks.
    progress_callback : callable, optional
        Called with progress percentage (0-100).
    auto_download : bool
        Automatically download model if not found locally.
    check_classified : bool
        If True, check for existing classification fields and warn/stop if found.
    fields : dict, optional
        Dictionary of existing scalar fields (e.g. from load_point_cloud).

    Returns
    -------
    np.ndarray or None
        Integer classification labels for each point, or None if already classified.
        For TreeFiltering: 1=ground/understory, 2=overstory vegetation.
        For UrbanFiltering: 1-7 class labels.
        For WoodCls: stem vs branch classification.
    """
    if check_classified and _check_existing_classification(fields, "tree_filtering"):
        return None

    from modules.filter.componentFilter import filterPoints

    config_file = get_config_path(model_name)
    model_path = get_model_path(model_name, auto_download=auto_download)
    cb = _make_progress_callback(progress_callback)

    pcd = np.asarray(points, dtype=np.float64)
    if pcd.ndim != 2 or pcd.shape[1] < 3:
        raise ValueError("points must be an (N, 3) array")

    labels = filterPoints(
        config_file, pcd[:, :3], model_path,
        if_bottom_only=if_bottom_only,
        use_efficient="esegformer" in model_name,
        use_cuda=use_cuda,
        progress_callback=cb,
    )
    return labels


# Aliases
stem_classification = tree_filtering
wood_classification = tree_filtering
urban_filtering = tree_filtering


# ---------------------------------------------------------------------------
# TreeisoNet: Tree Location Detection
# ---------------------------------------------------------------------------

def tree_location(points, model_name, use_cuda=True, if_stem=False,
                  cutoff_thresh=1.0, custom_resolution=None,
                  progress_callback=None, auto_download=True,
                  check_classified=True, fields=None):
    """Detect tree top or stem base locations.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    model_name : str
        Tree location model name (e.g. "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)").
    use_cuda : bool
        Use GPU acceleration.
    if_stem : bool
        If True, detect stem bases. If False, detect tree tops with confidence/radius.
    cutoff_thresh : float
        Height cutoff threshold for stem detection (0-1 fraction of tree height).
    custom_resolution : np.ndarray, optional
        Custom voxel resolution [x, y, z] in meters. Uses model default if None.
    progress_callback : callable, optional
        Called with progress percentage (0-100).
    check_classified : bool
        If True, check for existing classification fields and warn/stop if found.
    fields : dict, optional
        Dictionary of existing scalar fields (e.g. from load_point_cloud).

    Returns
    -------
    np.ndarray or None
        If if_stem=True: (M, 3) array of stem base XYZ locations.
        If if_stem=False: (N, 5) array of [x, y, z, confidence, radius] per point.
        None if already classified.
    """
    if check_classified and _check_existing_classification(fields, "tree_location"):
        return None

    from modules.treeisonet.treeLoc import treeLoc

    config_file = get_config_path(model_name)
    model_path = get_model_path(model_name, auto_download=auto_download)
    cb = _make_progress_callback(progress_callback)

    pcd = np.asarray(points, dtype=np.float64)[:, :3]
    if custom_resolution is None:
        custom_resolution = np.array([0, 0, 0])

    preds = treeLoc(
        config_file, pcd, model_path,
        use_cuda=use_cuda,
        if_stem=if_stem,
        cutoff_thresh=cutoff_thresh,
        progress_callback=cb,
        custom_resolution=custom_resolution,
    )
    return preds


def post_peak_extraction(preds_tops, K=5, max_gap=0.3, min_rad=0.2,
                         nms_thresh=0.3, progress_callback=None):
    """Extract tree locations from per-point predictions via peak finding and NMS.

    Parameters
    ----------
    preds_tops : np.ndarray
        (N, 5) array from tree_location with if_stem=False.
        Columns: [x, y, z, confidence, radius].
    K : int
        Number of nearest neighbors for connected component analysis.
    max_gap : float
        Maximum distance between connected points.
    min_rad : float
        Minimum crown radius to keep.
    nms_thresh : float
        Non-maximum suppression threshold.

    Returns
    -------
    np.ndarray
        (M, 4) array of [x, y, z, radius] for detected tree locations.
    """
    from modules.treeisonet.treeLoc import postPeakExtraction

    cb = _make_progress_callback(progress_callback)
    return postPeakExtraction(preds_tops, K=K, max_gap=max_gap,
                              min_rad=min_rad, nms_thresh=nms_thresh,
                              progress_callback=cb)


# ---------------------------------------------------------------------------
# TreeisoNet: Crown Segmentation
# ---------------------------------------------------------------------------

def tree_offset(points, tree_locations, model_name, use_cuda=True,
                custom_resolution=None, progress_callback=None, auto_download=True,
                check_classified=True, fields=None):
    """Segment crown points to tree locations using deep learning offset prediction.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    tree_locations : np.ndarray
        (M, 3) array of tree top/base XYZ coordinates.
    model_name : str
        TreeOff model name.
    use_cuda : bool
        Use GPU acceleration.
    check_classified : bool
        If True, check for existing classification fields and warn/stop if found.
    fields : dict, optional
        Dictionary of existing scalar fields (e.g. from load_point_cloud).

    Returns
    -------
    np.ndarray or None
        Integer tree ID labels for each point (starting from 1), or None if already classified.
    """
    if check_classified and _check_existing_classification(fields, "tree_offset"):
        return None

    from modules.treeisonet.treeOff import treeOff

    config_file = get_config_path(model_name)
    model_path = get_model_path(model_name, auto_download=auto_download)
    cb = _make_progress_callback(progress_callback)

    pcd = np.asarray(points, dtype=np.float64)[:, :3]
    treeloc = np.asarray(tree_locations, dtype=np.float64)[:, :3]

    if custom_resolution is None:
        custom_resolution = np.array([0, 0, 0])

    return treeOff(config_file, pcd, treeloc, model_path,
                   use_cuda=use_cuda, progress_callback=cb,
                   custom_resolution=custom_resolution)


def stem_clustering(points, stem_cls, base_locations, min_res=0.06,
                    max_isolated_distance=0.3, progress_callback=None,
                    check_classified=True, fields=None):
    """Cluster stem points to individual trees using shortest path algorithm.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    stem_cls : np.ndarray
        (N,) array of stem classification (>1 = stem point).
    base_locations : np.ndarray
        (M, 3) array of tree base XYZ coordinates.
    check_classified : bool
        If True, check for existing classification fields and warn/stop if found.
    fields : dict, optional
        Dictionary of existing scalar fields (e.g. from load_point_cloud).

    Returns
    -------
    np.ndarray or None
        Integer tree ID labels for each point, or None if already classified.
    """
    if check_classified and _check_existing_classification(fields, "stem_clustering"):
        return None

    from modules.treeisonet.stemCluster import shortestpath3D

    cb = _make_progress_callback(progress_callback)
    pcd = np.asarray(points, dtype=np.float64)[:, :3]
    return shortestpath3D(pcd, stem_cls, base_locations,
                          min_res=min_res,
                          max_isolated_distance=max_isolated_distance,
                          progress_callback=cb)


def crown_clustering(points, treeoff, min_res=0.15, K=5, reg_strength=1.0,
                     max_isolated_distance=0.3, progress_callback=None,
                     check_classified=True, fields=None):
    """Cluster crown points using graph-based segmentation.

    Parameters
    ----------
    points : np.ndarray
        (N, 3+) array of point coordinates.
    treeoff : np.ndarray
        (N,) array of tree offset labels from tree_offset.
    min_res : float
        Voxel resolution for decimation.
    K : int
        Number of nearest neighbors for graph construction.
    reg_strength : float
        Regularization strength for cut pursuit (or DBSCAN eps scaling).
    check_classified : bool
        If True, check for existing classification fields and warn/stop if found.
    fields : dict, optional
        Dictionary of existing scalar fields (e.g. from load_point_cloud).

    Returns
    -------
    np.ndarray or None
        Integer tree ID labels for each point, or None if already classified.
    """
    if check_classified and _check_existing_classification(fields, "crown_clustering"):
        return None

    from modules.treeisonet.crownCluster import init_cutpursuit, shortestpath3D

    cb = _make_progress_callback(progress_callback)
    pcd = np.asarray(points, dtype=np.float64)

    initsegs, _ = init_cutpursuit(pcd, min_res=min_res, K=K,
                                  reg_strength=reg_strength,
                                  progress_callback=cb)
    labels = shortestpath3D(pcd, treeoff, initsegs,
                            min_res=min_res * 0.4,
                            max_isolated_distance=max_isolated_distance,
                            progress_callback=cb)
    return labels


def crown_offset(points, stem_id, model_name, use_cuda=True,
                 progress_callback=None, auto_download=True,
                 check_classified=True, fields=None):
    """Refine crown segmentation using deep learning 3D offset prediction.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    stem_id : np.ndarray
        (N,) array of stem/tree ID labels (0 = unassigned).
    model_name : str
        CrownOff model name.
    check_classified : bool
        If True, check for existing classification fields and warn/stop if found.
    fields : dict, optional
        Dictionary of existing scalar fields (e.g. from load_point_cloud).

    Returns
    -------
    np.ndarray or None
        Refined integer tree ID labels for each point, or None if already classified.
    """
    if check_classified and _check_existing_classification(fields, "crown_offset"):
        return None

    from modules.treeisonet.crownOff import crownOff

    config_file = get_config_path(model_name)
    model_path = get_model_path(model_name, auto_download=auto_download)
    cb = _make_progress_callback(progress_callback)

    pcd = np.asarray(points, dtype=np.float64)[:, :3]
    return crownOff(config_file, pcd, stem_id, model_path,
                    use_cuda=use_cuda, progress_callback=cb)


def clean_small_clusters(points, max_gap=3.0, min_points=100, min_res=0.03):
    """Remove small connected components from a point cloud.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    max_gap : float
        Maximum distance for graph connectivity.
    min_points : int
        Minimum number of points for a cluster to keep.
    min_res : float
        Voxel resolution for decimation.

    Returns
    -------
    np.ndarray
        (N,) array: positive values for kept clusters, 0 for removed.
    """
    from modules.treeisonet.cleanSmallerClusters import applySmallClusterClean

    pcd = np.asarray(points, dtype=np.float64)[:, :3]
    return applySmallClusterClean(pcd, max_gap_clean=max_gap,
                                 min_points_clean=min_points, MIN_RES=min_res)


# ---------------------------------------------------------------------------
# QSM (Quantitative Structure Model)
# ---------------------------------------------------------------------------

def init_segmentation(points, K_stem=5, reg_strength_stem=1.0,
                      K_branch=10, reg_strength_branch=1.0,
                      progress_callback=None):
    """Initialize branch/stem segmentation for QSM.

    Parameters
    ----------
    points : np.ndarray
        (N, 4) array of [x, y, z, stemcls]. stemcls > 1 indicates stem points.

    Returns
    -------
    np.ndarray
        Integer segment labels for each point.
    """
    from modules.qsm.applyQSM import initSegmentation

    cb = _make_progress_callback(progress_callback)
    return initSegmentation(points, K_stem=K_stem,
                            reg_strength_stem=reg_strength_stem,
                            K_branch=K_branch,
                            reg_strength_branch=reg_strength_branch,
                            progress_callback=cb)


def apply_qsm(points, k_neighbors=6, max_graph_distance=40,
               max_connectivity_search_distance=0.03,
               occlusion_distance_cutoff=0.4,
               progress_callback=None, output_dir=None):
    """Apply Quantitative Structure Model to extract tree skeleton.

    Parameters
    ----------
    points : np.ndarray
        (N, 5) array of [x, y, z, stemcls, init_segs].
    k_neighbors : int
        Number of nearest neighbors for graph construction.
    max_graph_distance : float
        Maximum graph distance for branch path finding.
    output_dir : str, optional
        Directory to save XML and OBJ outputs.

    Returns
    -------
    tuple
        (tree, segs_centroids, segs_labels, tree_centroid_radius)
        - tree: list of branch paths (node index lists)
        - segs_centroids: (M, 3) array of segment centroids
        - segs_labels: (N,) integer branch labels
        - tree_centroid_radius: list of branch node arrays [x,y,z,radius]
    """
    from modules.qsm.applyQSM import applyQSM as _applyQSM
    from modules.qsm.applyQSM import saveTreeToXML, saveTreeToObj

    cb = _make_progress_callback(progress_callback)
    tree, segs_centroids, segs_labels, tree_centroid_radius = _applyQSM(
        points, k_neighbors=k_neighbors,
        max_graph_distance=max_graph_distance,
        max_connectivity_search_distance=max_connectivity_search_distance,
        occlusion_distance_cutoff=occlusion_distance_cutoff,
        progress_callback=cb,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        xml_path = os.path.join(output_dir, "tree_structure.xml")
        obj_path = os.path.join(output_dir, "tree_mesh.obj")
        saveTreeToXML(tree, tree_centroid_radius, xml_path)
        saveTreeToObj(tree_centroid_radius, obj_path)
        print(f"QSM outputs saved to: {output_dir}")

    return tree, segs_centroids, segs_labels, tree_centroid_radius


# ---------------------------------------------------------------------------
# DTM and Statistics
# ---------------------------------------------------------------------------

def create_dtm(points, resolution=1.0, ground_class=None,
               tile_size=None, buffer_size=None):
    """Create a Digital Terrain Model from point cloud data.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) or (N, 4) array. If 4 columns, the last is ground classification.
    resolution : float
        DTM grid resolution in meters.
    ground_class : np.ndarray, optional
        (N,) array of ground classification (<=1 = ground).
    tile_size : tuple, optional
        (x, y) tile size for tiled processing of large datasets.
    buffer_size : tuple, optional
        (x, y) buffer around each tile.

    Returns
    -------
    np.ndarray
        (M, 3) array of DTM grid points [x, y, z].
    """
    from modules.filter.createDTM import createDtm

    pcd = np.asarray(points, dtype=np.float64)
    res = np.array([resolution, resolution])

    if ground_class is not None:
        pcd = np.column_stack([pcd[:, :3], ground_class])

    ts = np.array(tile_size) if tile_size is not None else None
    bs = np.array(buffer_size) if buffer_size is not None else None

    return createDtm(pcd, resolution=res, tile_size=ts, buffer_size=bs)


def tree_statistics(points, tree_ids, pcd_min=None, treefilter=None,
                    output_path=None, dtm_resolution=1.0,
                    progress_callback=None):
    """Compute per-tree statistics (height, crown area, centroid).

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    tree_ids : np.ndarray
        (N,) array of integer tree ID labels.
    pcd_min : np.ndarray, optional
        (3,) array of point cloud minimum coordinates for coordinate restoration.
    treefilter : np.ndarray, optional
        (N,) array of ground/vegetation classification (<=1 = ground).
    output_path : str, optional
        Path to save statistics CSV file.
    dtm_resolution : float
        Resolution for DTM generation.

    Returns
    -------
    np.ndarray
        (M, 6) array: [tree_id, x, y, z, height, crown_area] per tree.
    """
    from modules.treeisonet.treeStat import treeStat

    cb = _make_progress_callback(progress_callback)
    pcd = np.asarray(points, dtype=np.float64)[:, :3]
    ids = np.asarray(tree_ids, dtype=np.float64)

    return treeStat(pcd, ids, pcd_min=pcd_min, treefilter=treefilter,
                    outpath=output_path, dtm_resolution=dtm_resolution,
                    progress_callback=cb)
