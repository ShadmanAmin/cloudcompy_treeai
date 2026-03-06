"""I/O utilities for TreeAIBox.

Provides functions to load and save point clouds using laspy,
with optional CloudComPy bridge functions.
"""

import numpy as np
from pathlib import Path


def load_point_cloud(path):
    """Load a point cloud from a LAS/LAZ or text file.

    Parameters
    ----------
    path : str or Path
        Path to the point cloud file (.las, .laz, .txt, .csv, .xyz).

    Returns
    -------
    dict
        Dictionary with keys:
        - "points": np.ndarray of shape (N, 3) with XYZ coordinates
        - "fields": dict mapping field names to np.ndarray of shape (N,)
        - "header": laspy header object (only for LAS/LAZ files)
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".las", ".laz"):
        return _load_las(path)
    elif ext in (".txt", ".csv", ".xyz"):
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .las, .laz, .txt, .csv, or .xyz")


def _load_las(path):
    """Load a LAS/LAZ file using laspy."""
    import laspy

    las = laspy.read(str(path))
    points = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    # Extract extra scalar fields
    fields = {}
    standard_dims = {"X", "Y", "Z", "x", "y", "z"}
    for dim in las.point_format.dimension_names:
        if dim not in standard_dims:
            try:
                fields[dim] = np.array(getattr(las, dim))
            except Exception:
                pass

    return {"points": points, "fields": fields, "header": las.header}


def _load_text(path):
    """Load a text/CSV file as a point cloud."""
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    points = data[:, :3]
    fields = {}
    # If there are more than 3 columns, treat them as unnamed fields
    for i in range(3, data.shape[1]):
        fields[f"field_{i-3}"] = data[:, i]

    return {"points": points, "fields": fields, "header": None}


def save_point_cloud(path, points, fields=None, source_header=None):
    """Save a point cloud to a LAS/LAZ or text file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    fields : dict, optional
        Dictionary mapping field names to np.ndarray of shape (N,).
    source_header : laspy.LasHeader, optional
        Header from an input LAS file to preserve metadata.
    """
    path = Path(path)
    ext = path.suffix.lower()
    fields = fields or {}

    if ext in (".las", ".laz"):
        _save_las(path, points, fields, source_header)
    elif ext in (".txt", ".csv", ".xyz"):
        _save_text(path, points, fields)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _save_las(path, points, fields, source_header):
    """Save to LAS/LAZ format using laspy."""
    import laspy

    if source_header is not None:
        las = laspy.LasData(source_header)
    else:
        las = laspy.LasData(laspy.LasHeader(point_format=0, version="1.4"))

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    for name, values in fields.items():
        dtype_map = {
            np.dtype("int32"): "int32",
            np.dtype("int64"): "int32",
            np.dtype("float32"): "float32",
            np.dtype("float64"): "float64",
        }
        values = np.asarray(values)
        las_type = dtype_map.get(values.dtype, "float64")

        try:
            # Try setting existing dimension first
            setattr(las, name, values)
        except Exception:
            # Add as extra dimension
            las.add_extra_dim(
                laspy.ExtraBytesParams(name=name, type=las_type, description=name)
            )
            setattr(las, name, values)

    las.write(str(path))


def _save_text(path, points, fields):
    """Save to text format."""
    if fields:
        extra = np.column_stack([v for v in fields.values()])
        data = np.column_stack([points, extra])
    else:
        data = points
    np.savetxt(str(path), data)


# --- Optional CloudComPy bridge ---

def cloudcompy_to_numpy(cc_cloud):
    """Convert a CloudComPy point cloud to numpy arrays.

    Parameters
    ----------
    cc_cloud : cloudComPy.ccPointCloud
        A CloudComPy point cloud object.

    Returns
    -------
    dict
        Same format as load_point_cloud output.

    Raises
    ------
    ImportError
        If cloudComPy is not available.
    """
    try:
        import cloudComPy as cc
    except ImportError:
        raise ImportError(
            "cloudComPy is not installed. Install it via conda to use this function.\n"
            "See: https://github.com/CloudCompare/CloudComPy"
        )

    points = cc_cloud.toNpArrayCopy()

    fields = {}
    for i in range(cc_cloud.getNumberOfScalarFields()):
        sf = cc_cloud.getScalarField(i)
        name = sf.getName()
        fields[name] = sf.toNpArray()

    return {"points": points, "fields": fields, "header": None}


def numpy_to_cloudcompy(points, fields=None, name="cloud"):
    """Create a CloudComPy point cloud from numpy arrays.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.
    fields : dict, optional
        Dictionary mapping field names to np.ndarray of shape (N,).
    name : str
        Name for the CloudComPy cloud object.

    Returns
    -------
    cloudComPy.ccPointCloud

    Raises
    ------
    ImportError
        If cloudComPy is not available.
    """
    try:
        import cloudComPy as cc
    except ImportError:
        raise ImportError(
            "cloudComPy is not installed. Install it via conda to use this function.\n"
            "See: https://github.com/CloudCompare/CloudComPy"
        )

    cloud = cc.ccPointCloud(name)
    coords = np.ascontiguousarray(points[:, :3].astype(np.float32))
    cloud.coordsFromNPArray_copy(coords)

    if fields:
        for field_name, values in fields.items():
            sf_idx = cloud.addScalarField(field_name)
            sf = cloud.getScalarField(sf_idx)
            arr = sf.toNpArray()
            arr[:] = np.asarray(values, dtype=np.float32)
            sf.computeMinAndMax()

    return cloud
