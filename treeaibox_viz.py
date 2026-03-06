"""Visualization utilities for TreeAIBox.

Provides matplotlib-based plotting for point clouds, tree locations,
classification results, and DTMs. Designed for use in Jupyter notebooks.
"""

import numpy as np


def plot_point_cloud(points, labels=None, cmap="tab20", point_size=0.3,
                     title="Point Cloud", figsize=(12, 8), elev=30, azim=45,
                     subsample=None, ax=None):
    """Plot a 3D point cloud with optional label coloring.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    labels : np.ndarray, optional
        (N,) array of integer labels for coloring.
    cmap : str
        Matplotlib colormap name.
    point_size : float
        Point marker size.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.
    subsample : int, optional
        If set, randomly subsample to this many points for faster rendering.
    ax : matplotlib Axes3D, optional
        Existing axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points)[:, :3]
    if labels is not None:
        labels = np.asarray(labels)

    # Subsample for large clouds
    if subsample is not None and len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts = pts[idx]
        if labels is not None:
            labels = labels[idx]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if labels is not None:
        scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                             c=labels, cmap=cmap, s=point_size, alpha=0.8)
        plt.colorbar(scatter, ax=ax, shrink=0.6, label="Label")
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=pts[:, 2], cmap="viridis", s=point_size, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    return fig


def plot_tree_locations(points, locations, point_size=0.1, marker_size=50,
                        title="Tree Locations", figsize=(12, 8),
                        subsample=50000, ax=None):
    """Plot a point cloud with detected tree locations overlaid.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    locations : np.ndarray
        (M, 3+) array of tree location XYZ coordinates.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points)[:, :3]
    locs = np.asarray(locations)[:, :3]

    if subsample is not None and len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts = pts[idx]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c="lightgray", s=point_size, alpha=0.3)
    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2],
               c="red", s=marker_size, marker="^", label=f"Trees ({len(locs)})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_classification(points, classes, class_names=None, point_size=0.3,
                        title="Classification", figsize=(12, 8),
                        subsample=50000, ax=None):
    """Plot classified point cloud with a legend.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    classes : np.ndarray
        (N,) array of integer class labels.
    class_names : dict, optional
        Mapping from class ID to name (e.g. {1: "ground", 2: "vegetation"}).
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points)[:, :3]
    cls = np.asarray(classes)

    if subsample is not None and len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts = pts[idx]
        cls = cls[idx]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    unique_classes = np.unique(cls)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_classes), 2)))

    for i, c in enumerate(unique_classes):
        mask = cls == c
        name = class_names.get(c, f"Class {c}") if class_names else f"Class {int(c)}"
        ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                   c=[colors[i % len(colors)]], s=point_size, label=name, alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_dtm(dtm_points, cmap="terrain", title="Digital Terrain Model",
             figsize=(10, 8)):
    """Plot a DTM as a 2D scatter colored by elevation.

    Parameters
    ----------
    dtm_points : np.ndarray
        (M, 3) array of DTM grid points [x, y, z].
    """
    import matplotlib.pyplot as plt

    dtm = np.asarray(dtm_points)
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(dtm[:, 0], dtm[:, 1], c=dtm[:, 2],
                         cmap=cmap, s=1, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Elevation (m)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig
