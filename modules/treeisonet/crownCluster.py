import numpy as np
import numpy_indexed as npi
import numpy_groupies as npg

from statistics import mode
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

try:
    from cut_pursuit_py import _cut_pursuit
    HAS_CUT_PURSUIT = True
except ImportError:
    HAS_CUT_PURSUIT = False

def decimate_pcd(columns,min_res):
    _, block_idx_uidx, block_inverse_idx = np.unique(np.floor(columns[:,:3]/min_res).astype(np.int32),axis=0, return_index=True, return_inverse=True)
    return block_idx_uidx,block_inverse_idx

def create_node_graph(points,k=10,max_distance=0.4):#max_allowed distance for a component not to be considered as an outlier
    node_total=len(np.unique(points[:,-1]))# the last column of the points is the node ID
    _, v_group = npi.group_by(points[:, -1].astype(np.int32), np.arange(len(points[:, -1])))#extract the mean value of each node centroid
    centroids = np.array([np.mean(points[idx, :3], 0) for idx in v_group])
    n_centroids = len(centroids)
    kdtree = cKDTree(centroids[:, :3])
    _, indices = kdtree.query(centroids[:, :3], k=min(k + 1,len(centroids)))#create a table with pairs of closest K+1 nodes
    distance_pairs = np.zeros([len(centroids) * k, 4])#x,y,z, distance in 2D, and distance in 3D
    for i, v in enumerate(v_group):
        nn_idx = indices[i, 1:]
        tree = cKDTree(points[v, :3])
        for j, nv in enumerate(nn_idx):
            nn_dist, nn_j = tree.query(points[v_group[nv], :3], k=1)
            nn_ij = np.argmin(nn_dist)
            pts1=points[v_group[nv], :3][nn_ij]
            pts2=points[v, :3][nn_j[nn_ij]]
            nn_dist2d=np.linalg.norm(pts1[:2]-pts2[:2]) # minimal point distance between two nodes in 2D
            nn_dist = np.linalg.norm(pts1[:3]-pts2[:3]) # minimal point distance between two nodes in 3D; equivalent to: nn_dist = np.min(nn_dist)
            distance_pairs[k * i + j, 0] = i
            distance_pairs[k * i + j, 1] = nv
            distance_pairs[k * i + j, 2] = nn_dist #use 3D distance to create distance graph for connectivity analysis
            distance_pairs[k * i + j, 3] = nn_dist2d

    # use 2D distance to filter out far neighbors (3D distance is not suitable here due to strong vertical point occlusion)
    distance_pairs = distance_pairs[distance_pairs[:, -1] < max_distance]
    distance_pairs = distance_pairs[distance_pairs[:, -2] < max_distance*2]
    graph = csr_matrix((distance_pairs[:, -2], (distance_pairs[:, 0], distance_pairs[:, 1])), shape=(n_centroids, n_centroids))
    graph = (graph + graph.T) / 2
    return graph,node_total


def cluster_graph(graph,node_total,target_idx):
    distances = dijkstra(graph, indices=target_idx, return_predecessors=False)#shortest paths from each node to each stem node
    distances = distances.T
    # for each row (node), find the stem node id (column) with the shortest path
    min_dist_idx = np.argmin(distances, axis=1, keepdims=True)
    min_dist = np.take_along_axis(distances, min_dist_idx, axis=1)[:, 0]
    nearest_targets = target_idx[min_dist_idx[:, 0]]#stem node id (column) with the shortest path
    filter_ind = ~np.isinf(min_dist)
    nearest_targets = nearest_targets[filter_ind]#exclude the infinite ones
    nearest_idx = np.where(filter_ind)[0]
    label_pred = np.zeros(node_total)-1#assign -1 as default label (assigning 0 will mess up with the node ID starting from 0)
    label_pred[nearest_idx] = nearest_targets

    label_unq,label_pred=np.unique(label_pred,return_inverse=True)#shuffle the labels in order from 0-N
    return label_pred,len(label_unq)

def filter_g(g):#get the mode value among the positive values
    ind=g>0
    if np.sum(ind)>0:
        return mode(g[ind])
    else:
        return 0

def _fallback_dbscan_segmentation(points, min_res=0.15, eps=None, min_samples=5, progress_callback=lambda x: None):
    """DBSCAN-based fallback for cut_pursuit segmentation.

    Used when cut_pursuit_py is not installed. Provides similar oversegmentation
    using sklearn's DBSCAN algorithm.
    """
    from sklearn.cluster import DBSCAN

    dec_idx_uidx, dec_inverse_idx = decimate_pcd(points[:, :3], min_res)
    pcd_dec = points[dec_idx_uidx, :3]

    progress_callback(10)

    if eps is None:
        eps = min_res * 3.0

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pcd_dec)
    labels = clustering.labels_

    # Relabel noise points (-1) to unique individual labels
    noise_mask = labels == -1
    if np.any(noise_mask):
        max_label = labels.max() + 1
        labels[noise_mask] = np.arange(max_label, max_label + noise_mask.sum())

    progress_callback(60)

    if min_res > 0:
        return labels[dec_inverse_idx], len(np.unique(labels))
    else:
        return labels, len(np.unique(labels))


def init_cutpursuit(points,min_res=0.15,K = 5,reg_strength = 1.0, progress_callback=lambda x: None):
    if not HAS_CUT_PURSUIT:
        print("Warning: cut_pursuit_py not installed. Using DBSCAN fallback for segmentation.")
        print("  For best results, install cut_pursuit_py: pip install cut-pursuit")
        return _fallback_dbscan_segmentation(points, min_res=min_res, progress_callback=progress_callback)

    dec_idx_uidx, dec_inverse_idx = decimate_pcd(points[:, :3], min_res)  # reduce points first
    pcd_dec = points[dec_idx_uidx,:3]

    progress_callback(10)

    kdtree = cKDTree(pcd_dec)
    _, nn_idx = kdtree.query(pcd_dec, k=K+1)

    progress_callback(15)

    # Prepare graph structure
    indices = nn_idx[:, 1:]  # exclude self
    n_nodes = len(pcd_dec)

    # Create edge lists
    eu = np.repeat(np.arange(n_nodes), K)
    ev = indices.ravel()

    # Edge weights
    edge_weights = np.ones_like(eu, dtype=np.float32)*reg_strength

    # Perform cut pursuit
    labels = _cut_pursuit.perform_cut_pursuit(
        reg_strength=reg_strength,  # Regularization strength
        D=3,  # Dimension of points
        pc_vec=pcd_dec.astype(np.float32),  # Point cloud
        edge_weights=edge_weights,
        Eu=eu.astype(np.uint32),
        Ev=ev.astype(np.uint32),
        verbose=False
    )

    progress_callback(60)

    if min_res>0:
        return labels[dec_inverse_idx],len(np.unique(labels))
    else:
        return labels, len(np.unique(labels))

def shortestpath3D(points,treeoff,initsegs,min_res=0.06, max_isolated_distance = 0.3,progress_callback=lambda x: None):

    dec_idx_uidx, dec_inverse_idx = decimate_pcd(points[:, :3], min_res)  # reduce points first
    pcd_dec = points[dec_idx_uidx]
    treeoff_dec = treeoff[dec_idx_uidx]
    initsegs_dec = initsegs[dec_idx_uidx]

    progress_callback(70)

    _, blob_inverse_idx, blob_size = np.unique(initsegs_dec, return_inverse=True, return_counts=True)
    pcd_dec_blob_treeoff = npg.aggregate(blob_inverse_idx, treeoff_dec, func=filter_g).astype(np.int32)

    blob_labels=blob_inverse_idx
    blob_treeoff_idx=np.where(pcd_dec_blob_treeoff>0)[0]

    _, blob_idx_groups = npi.group_by(blob_labels, np.arange(len(blob_labels)))

    blob_graph, blob_total = create_node_graph(np.concatenate([pcd_dec, blob_labels[:, np.newaxis]], axis=-1), k=20, max_distance=max_isolated_distance)

    progress_callback(80)

    seg_pred, seg_total = cluster_graph(blob_graph, blob_total, blob_treeoff_idx)



    seg_treeoff = npg.aggregate(seg_pred, pcd_dec_blob_treeoff, func=filter_g).astype(np.int32)
    final_pred=seg_treeoff[seg_pred]
    final_pred=final_pred[blob_inverse_idx][dec_inverse_idx]
    treeoff_ind=treeoff>0
    final_pred[treeoff_ind]=treeoff[treeoff_ind]

    progress_callback(90)
    return final_pred


if __name__ =='__main__':
    import laspy
    # path_to_las = r"..\..\data\2023_NWT-7N-31_REG-V2_SUB-2_SOR-6-3_2.las"
    # path_to_las = r"..\..\data\2023_NWT-10-7_REG-V2_SUB-2_SOR-6-3_test.las"
    path_to_las = r"C:\Users\ZXI\Downloads\2023_NWT-10-7_REG-V2_SUB-2_SOR-6-3_test_res.laz"

    las = laspy.read(path_to_las)
    pcd=np.transpose([las.x,las.y,las.z,las.init_segs,las.treeoff])
    min_res=0.06
    max_isolated_distance = 0.4

    labels=shortestpath3D(pcd, pcd[:,-1],pcd[:,-2], min_res,max_isolated_distance)
    las.add_extra_dim(laspy.ExtraBytesParams(name="shortestpath3d", type="int32", description="shortestpath3d"))
    las.shortestpath3d = labels
    las.write(r"C:\Users\ZXI\Downloads\shortestpath3d.laz")