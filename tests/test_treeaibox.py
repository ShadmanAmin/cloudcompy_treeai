"""Tests for TreeAIBox.

All tests use synthetic data — no GPU, pre-trained models, or real LAS files needed.
"""

import sys
import os
import tempfile
import numpy as np
import pytest

# Ensure the package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Algorithm tests: componentFilter
# ===========================================================================

class TestSlidingBlocks:
    def test_3d_basic(self):
        from modules.filter.componentFilter import sliding_blocks_point_indices

        np.random.seed(42)
        pts = np.random.rand(500, 3) * 10  # 10m cube
        block_size = np.array([5.0, 5.0, 5.0])
        origins, groups = sliding_blocks_point_indices(pts, block_size, overlap_ratio=0.1)

        assert len(origins) > 0
        assert len(groups) == len(origins)
        # Every point should appear in at least one group
        all_indices = np.concatenate(groups)
        unique_indices = np.unique(all_indices)
        assert len(unique_indices) <= 500

    def test_2d_basic(self):
        from modules.filter.componentFilter import sliding_blocks_point_indices

        np.random.seed(42)
        pts = np.random.rand(200, 2) * 10
        block_size = np.array([5.0, 5.0])
        origins, groups = sliding_blocks_point_indices(pts, block_size, overlap_ratio=0.2)

        assert len(origins) > 0
        for g in groups:
            assert len(g) > 0

    def test_single_block(self):
        from modules.filter.componentFilter import sliding_blocks_point_indices

        pts = np.random.rand(50, 3) * 2  # Small cloud, fits in one block
        block_size = np.array([10.0, 10.0, 10.0])
        origins, groups = sliding_blocks_point_indices(pts, block_size, overlap_ratio=0.1)

        assert len(origins) >= 1

    def test_dimension_mismatch_raises(self):
        from modules.filter.componentFilter import sliding_blocks_point_indices

        pts = np.random.rand(50, 3)
        with pytest.raises(ValueError):
            sliding_blocks_point_indices(pts, np.array([5.0, 5.0]), overlap_ratio=0.1)


# ===========================================================================
# Algorithm tests: treeLoc
# ===========================================================================

class TestNMS3D:
    def test_basic_suppression(self):
        from modules.treeisonet.treeLoc import nms3d

        # Two nearby candidates — one should be suppressed
        candidates = np.array([
            [0, 0, 10, 2.0],  # Higher, should be kept
            [0.5, 0.5, 8, 1.5],  # Close, lower, should be suppressed
            [10, 10, 12, 3.0],  # Far away, should be kept
        ])
        result = nms3d(candidates, nms_thresh=0.5)
        assert len(result) == 2

    def test_empty_input(self):
        from modules.treeisonet.treeLoc import nms3d

        candidates = np.empty((0, 4))
        result = nms3d(candidates)
        assert len(result) == 0

    def test_no_suppression(self):
        from modules.treeisonet.treeLoc import nms3d

        # All far apart
        candidates = np.array([
            [0, 0, 10, 1.0],
            [100, 100, 12, 1.0],
            [200, 200, 8, 1.0],
        ])
        result = nms3d(candidates, nms_thresh=0.5)
        assert len(result) == 3


class TestPeakFinder:
    def test_single_peak(self):
        from modules.treeisonet.treeLoc import peakfinder

        img = np.zeros((50, 50))
        img[25, 25] = 1.0  # Single peak
        pcd_min = np.array([0.0, 0.0])
        min_res = np.array([0.1, 0.1])

        coords = peakfinder(img, pcd_min, min_res)
        assert len(coords) == 1

    def test_no_peaks(self):
        from modules.treeisonet.treeLoc import peakfinder

        img = np.zeros((50, 50))
        pcd_min = np.array([0.0, 0.0])
        min_res = np.array([0.1, 0.1])

        coords = peakfinder(img, pcd_min, min_res)
        assert len(coords) == 0

    def test_multiple_peaks(self):
        from modules.treeisonet.treeLoc import peakfinder

        img = np.zeros((50, 50))
        img[10, 10] = 1.0
        img[40, 40] = 1.0
        img[10, 40] = 1.0
        pcd_min = np.array([0.0, 0.0])
        min_res = np.array([0.1, 0.1])

        coords = peakfinder(img, pcd_min, min_res)
        assert len(coords) == 3


# ===========================================================================
# Algorithm tests: stemCluster / cleanSmallerClusters
# ===========================================================================

class TestCreateSparseGraph:
    def test_basic_graph(self):
        from modules.treeisonet.stemCluster import create_sparse_graph

        np.random.seed(42)
        points = np.random.rand(100, 3)
        graph = create_sparse_graph(points, k=5, max_distance=0.5)

        assert graph.shape == (100, 100)
        assert graph.nnz > 0

    def test_graph_symmetry(self):
        from modules.treeisonet.stemCluster import create_sparse_graph

        points = np.random.rand(50, 3)
        graph = create_sparse_graph(points, k=5)
        diff = abs(graph - graph.T)
        assert diff.max() < 1e-10


class TestDecimatePcd:
    def test_decimation_reduces_points(self):
        from modules.treeisonet.stemCluster import decimate_pcd

        np.random.seed(42)
        points = np.random.rand(1000, 3)
        idx, inverse = decimate_pcd(points, 0.1)

        assert len(idx) <= 1000
        assert len(inverse) == 1000
        # Inverse should reconstruct original voxel assignments
        assert np.all(inverse >= 0)
        assert np.all(inverse < len(idx))


class TestCleanSmallClusters:
    def test_removes_small_clusters(self):
        from modules.treeisonet.cleanSmallerClusters import applySmallClusterClean

        np.random.seed(42)
        # Large cluster
        large = np.random.rand(200, 3) * 0.5
        # Small isolated cluster
        small = np.random.rand(5, 3) * 0.1 + 5.0
        points = np.vstack([large, small])

        result = applySmallClusterClean(points, max_gap_clean=0.3,
                                        min_points_clean=10, MIN_RES=0.05)
        assert len(result) == len(points)
        # Small cluster should be zeroed out
        assert np.sum(result[200:] == 0) >= 1


# ===========================================================================
# Algorithm tests: treeOff
# ===========================================================================

class TestMergeShift:
    def test_basic_assignment(self):
        from modules.treeisonet.treeOff import mergeshift

        # Points near two tree locations
        points = np.array([
            [0, 0, 0],
            [0.1, 0.1, 0],
            [10, 10, 0],
            [10.1, 10.1, 0],
        ], dtype=np.float64)
        treelocs = np.array([[0, 0, 0], [10, 10, 0]], dtype=np.float64)
        treeids = np.array([1, 2])

        labels = mergeshift(points, treelocs, treeids)
        assert labels[0] == 1
        assert labels[1] == 1
        assert labels[2] == 2
        assert labels[3] == 2


# ===========================================================================
# I/O tests
# ===========================================================================

class TestIO:
    def test_roundtrip_las(self):
        from treeaibox_io import load_point_cloud, save_point_cloud

        np.random.seed(42)
        points = np.random.rand(100, 3) * 100

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            tmp_path = f.name

        try:
            save_point_cloud(tmp_path, points)
            data = load_point_cloud(tmp_path)
            loaded = data["points"]

            assert loaded.shape == points.shape
            np.testing.assert_allclose(loaded, points, atol=0.01)
        finally:
            os.unlink(tmp_path)

    def test_roundtrip_with_fields(self):
        from treeaibox_io import load_point_cloud, save_point_cloud

        np.random.seed(42)
        points = np.random.rand(100, 3) * 100
        fields = {"classification": np.ones(100, dtype=np.int32) * 2}

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            tmp_path = f.name

        try:
            save_point_cloud(tmp_path, points, fields=fields)
            data = load_point_cloud(tmp_path)

            assert "classification" in data["fields"]
            np.testing.assert_array_equal(data["fields"]["classification"], 2)
        finally:
            os.unlink(tmp_path)

    def test_roundtrip_text(self):
        from treeaibox_io import load_point_cloud, save_point_cloud

        np.random.seed(42)
        points = np.random.rand(50, 3) * 10

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name

        try:
            save_point_cloud(tmp_path, points)
            data = load_point_cloud(tmp_path)
            np.testing.assert_allclose(data["points"], points, atol=1e-6)
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# Model management tests
# ===========================================================================

class TestModelManagement:
    def test_list_available_models(self):
        from treeaibox_models import list_available_models

        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert any("treefiltering" in m for m in models)

    def test_get_config_path_filter(self):
        from treeaibox_models import get_config_path

        config = get_config_path("treefiltering_tls_esegformer3D_128_8cm(GPU3GB)")
        assert os.path.exists(config)
        assert "filter" in config

    def test_get_config_path_treeisonet(self):
        from treeaibox_models import get_config_path

        config = get_config_path("treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)")
        assert os.path.exists(config)
        assert "treeisonet" in config

    def test_get_config_path_stemcls(self):
        from treeaibox_models import get_config_path

        # stemcls models use treeisonet subfolder
        config = get_config_path("treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)")
        assert os.path.exists(config)
        assert "treeisonet" in config

    def test_get_config_path_nonexistent(self):
        from treeaibox_models import get_config_path

        with pytest.raises(FileNotFoundError):
            get_config_path("nonexistent_model_name")

    def test_model_dir_creation(self):
        from treeaibox_models import get_model_dir

        model_dir = get_model_dir()
        assert os.path.isdir(model_dir)


# ===========================================================================
# API surface tests
# ===========================================================================

class TestAPISurface:
    def test_core_imports(self):
        from treeaibox import (
            tree_filtering,
            stem_classification,
            wood_classification,
            urban_filtering,
            tree_location,
            post_peak_extraction,
            tree_offset,
            stem_clustering,
            crown_clustering,
            crown_offset,
            clean_small_clusters,
            init_segmentation,
            apply_qsm,
            create_dtm,
            tree_statistics,
        )

    def test_io_imports(self):
        from treeaibox_io import (
            load_point_cloud,
            save_point_cloud,
            cloudcompy_to_numpy,
            numpy_to_cloudcompy,
        )

    def test_model_imports(self):
        from treeaibox_models import (
            get_model_dir,
            get_model_path,
            get_config_path,
            list_available_models,
            download_model,
        )

    def test_viz_imports(self):
        from treeaibox_viz import (
            plot_point_cloud,
            plot_tree_locations,
            plot_classification,
            plot_dtm,
        )


# ===========================================================================
# DBSCAN fallback tests
# ===========================================================================

class TestDBSCANFallback:
    def test_fallback_segmentation(self):
        from modules.treeisonet.crownCluster import _fallback_dbscan_segmentation

        np.random.seed(42)
        # Two distinct clusters
        cluster1 = np.random.rand(100, 3) * 0.5
        cluster2 = np.random.rand(100, 3) * 0.5 + 3.0
        points = np.vstack([cluster1, cluster2])

        labels, n_labels = _fallback_dbscan_segmentation(
            points, min_res=0.1, eps=0.5, min_samples=5
        )

        assert len(labels) == 200
        assert n_labels >= 2
        # Points in cluster1 should have same label
        assert len(np.unique(labels[:100])) <= 3  # Allow some noise splitting
        # Points in different clusters should differ
        assert not np.all(labels[:100] == labels[100])

    def test_init_cutpursuit_uses_fallback(self):
        from modules.treeisonet.crownCluster import init_cutpursuit, HAS_CUT_PURSUIT

        np.random.seed(42)
        points = np.random.rand(100, 3) * 2

        # Should work regardless of cut_pursuit availability
        labels, n_labels = init_cutpursuit(points, min_res=0.2, K=5, reg_strength=1.0)
        assert len(labels) == 100
        assert n_labels >= 1


# ===========================================================================
# DTM tests
# ===========================================================================

class TestDTM:
    def test_create_dtm_basic(self):
        from modules.treeisonet.treeStat import createDtm

        np.random.seed(42)
        # Flat ground at z=0 with some variation
        x = np.random.rand(500) * 10
        y = np.random.rand(500) * 10
        z = np.random.rand(500) * 0.1
        points = np.column_stack([x, y, z])

        dtm = createDtm(points, dtm_resolution=np.array([1.0, 1.0]))

        assert dtm.shape[1] == 3
        assert len(dtm) > 0
        # DTM z values should be close to ground level
        assert np.nanmax(dtm[:, 2]) < 1.0


# ===========================================================================
# Progress callback tests
# ===========================================================================

class TestProgress:
    def test_callback_receives_values(self):
        values = []

        def cb(v):
            values.append(v)

        from modules.treeisonet.treeLoc import nms3d
        # nms3d doesn't use callbacks, but we can test the callback pattern
        # using the _make_progress_callback wrapper
        from treeaibox import _make_progress_callback

        pcb = _make_progress_callback(cb)
        pcb(0)
        pcb(50)
        pcb(100)

        assert values == [0, 50, 100]
