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
    import csv

    with open(path, "r") as f:
        reader = csv.reader(f)
        header_row = next(reader)
        # Strip comment prefixes (e.g. "//X" -> "X")
        header_row = [h.lstrip("/").strip() for h in header_row]

        rows = list(reader)

    if not rows:
        points = np.empty((0, 3), dtype=np.float64)
        return {"points": points, "fields": {}, "header": None}

    # Determine which columns are numeric by inspecting the first data row
    n_cols = len(header_row)
    is_numeric = []
    for val in rows[0]:
        try:
            float(val)
            is_numeric.append(True)
        except ValueError:
            is_numeric.append(False)

    # XYZ are always the first 3 columns
    points = np.array([[float(row[i]) for i in range(3)] for row in rows], dtype=np.float64)

    fields = {}
    for col_idx in range(3, min(n_cols, len(is_numeric))):
        col_name = header_row[col_idx] if col_idx < len(header_row) else f"field_{col_idx - 3}"
        if is_numeric[col_idx]:
            fields[col_name] = np.array([float(row[col_idx]) for row in rows], dtype=np.float64)
        else:
            fields[col_name] = np.array([row[col_idx] for row in rows])

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
