import time
import math
import numpy as np
import rerun as rr
import scipy.spatial
import ransac_simple
from sklearn.cluster import DBSCAN
import dbscan


def normal_from_3pt(pt1, pt2, pt3):
    v1 = pt2 - pt1
    v2 = pt3 - pt1
    normal = np.cross(v1, v2)

    # Normalize the normal vector
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise Exception(f"Points are collinear!\n{pt1}\n{pt2}\n{pt3}\n")
        
    return normal / normal_norm

def ransac(kdtree: scipy.spatial.KDTree, mask=None, iterations=1000, epsilon=0.1):
    """
    RANSAC algorithm for plane detection in point cloud data.
    
    Parameters:
        kdtree: scipy.spatial.KDTree built with a (N, 3) array containing [x, y, z] coordinates
        iterations: number of iterations for RANSAC
        epsilon: maximum distance from point to plane to be considered an inlier
    
    Returns:
        best_score: inliner points amount in this run
        best_inliers: best fitted inliner points (index) in this run
        best_plane: tuple (A, B, C, D) best fitted plane in this run
    """
    if not isinstance(mask, np.ndarray):
        mask = np.ones(len(kdtree.data), dtype=int)

    best_score = 0
    best_inliers = np.array([])
    best_plane = (0.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng()
    available = np.where(mask==1)[0]
    prob_mask = mask / len(available)

    for _ in range(iterations):
        # 1st random choice
        sampled_index = rng.choice(len(kdtree.data), size=1, replace=False, p=prob_mask)[0]
        re = kdtree.query_ball_point(kdtree.data[sampled_index], 5.0)

        mask_2 = mask[re]
        available_2 = np.where(mask_2==1)[0]
        prob_mask_2 = mask_2 / len(available_2)
        if len(available_2) < 3:
            # print("unavailable")
            continue

        # 2nd random choice
        sampled_indices = rng.choice(re, size=3, replace=False, p=prob_mask_2)
        pt1, pt2, pt3 = kdtree.data[sampled_indices]

        try:
            normal = normal_from_3pt(pt1, pt2, pt3)
        except Exception as e:
            print(e)
            continue
                
        # Formula: ax + by + cz + d = 0, calculate d using point p1
        d = -(np.dot(normal, pt1))
        distances = np.abs(kdtree.data.dot(normal) + d)
        
        # Count inliers (points within distance threshold)
        inliers = np.where((distances < epsilon) & (mask == 1))[0]
        score = len(inliers)

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_plane = (normal[0], normal[1], normal[2], d)
    
    return best_score, best_inliers, best_plane

def extract_planes(pts: np.ndarray, min_score=400, iterations=1000, epsilon=0.1):
    """
    Extract multiple planes from point cloud using RANSAC
    
    Parameters: 
        pts: numpy array of shape (N, 4) containing [x, y, z, id] coordinates
        min_score: minimum number of inliers required for a plane to be considered valid
    
    Returns: 
        pts: a NumPy array Nx4; each point has x-y-z-segmentid
        planes: a NumPy array of planes parameters [[a, b, c, d]...] satisfy ax + by + cz + d = 0
    """

    planes = []
    id_count = 1
    invalid_count = 0
    kdtree = scipy.spatial.KDTree(pts[:, :3])
    mask = np.ones(len(pts), dtype=int)

    while True:
        score, inliers, plane = ransac(kdtree, mask, iterations, epsilon)

        if invalid_count > 5:
            break

        if score < min_score:
            invalid_count += 1
            continue
        else:
            invalid_count = 0

        pts[inliers, 3] = id_count
        mask[inliers] = 0
        planes.append(plane)
        
        print("\nid:", id_count)
        print("score:", score)
        # print(f"plane: {plane[0]}x + {plane[1]}y + {plane[2]}z + {plane[3]:.2f} = 0")
        print("remaining pts:", len(np.where(mask==1)[0]))

        id_count += 1

    del kdtree
    return pts, np.array(planes)

def post_process(pts: np.ndarray, planes: np.ndarray, epsilon=0.1, multiplier=5.0):
    """
    Reclassify points based on their neighboring data; this process corrects points that were initially misclassified.
    
    Parameters:
        pts: numpy array of shape (N, 4) containing [x, y, z, id] coordinates
        planes: a NumPy array of planes parameters [[a, b, c, d]...] satisfy ax + by + cz + d = 0
        epsilon: maximum distance from point to plane to be considered an inlier
    
    Returns:
        pts: a NumPy array Nx4; each point has x-y-z-segmentid
    """

    kdtree = scipy.spatial.KDTree(pts[:, :3])
    pts[:, 3] = 0

    for i, plane in enumerate(planes):
        a, b, c, d = plane

        distances = np.abs(pts[:, :3].dot([a, b, c]) + d)
        inliers = np.where(distances < epsilon)[0]

        collison = []
        for index in inliers:
            if pts[index][3] == 0:
                pts[index][3] = i + 1
            else:
                collison.append(index)
        
        pts_copy = pts.copy()
        radius = epsilon * multiplier
        for index in collison:
            pts[index, 3] = 0
            neighbor_indices = np.array(kdtree.query_ball_point(pts[index, :3], radius))
            indices = neighbor_indices[pts_copy[neighbor_indices, 3] > 0]

            if len(indices) == 0:
                pts[index, 3] = i + 1
                continue
            
            uniq = np.unique_counts(pts[indices, 3])
            max_index = np.argmax(uniq.counts)
            id = uniq.values[max_index]
            pts[index, 3] = id

    del kdtree
    return pts

def distance_cluster(pts: np.ndarray, delta, n_min=10):
    """
    Refine RANSAC-segmented planes using simplified DBscan (only distance is considered, density is not take into account) to split into separate clusters

    Parameters:
        pts: numpy array of shape (N, 4) containing [x, y, z, segment_id] coordinates
        delta: maximum distance between samples for them to be considered in the same cluster
        n_min: Minimum number of samples in a cluster

    Returns:
        pts: Refined point cloud with updated segment IDs
    """
    pts_copy = pts.copy()
    pts_copy[:, 3] = 0
    segment_ids = np.unique(pts[:, 3]).astype(int)
    id_counter = 1

    for segment_id in segment_ids:
        if segment_id == 0:  # Skip unsegmented points
            continue

        # Extract indices of points belonging to the current segment
        segment_indices = np.where(pts[:, 3] == segment_id)[0]
        segment_points = pts[segment_indices, :3]  # Only use x, y, z for clustering

        # Apply DBSCAN clustering to the points in the current segment
        labels = dbscan.cluster_by_distance(segment_points, delta, n_min)

        # Update the segment IDs for each cluster
        for cluster_label in np.unique(labels):
            if cluster_label == -1:  # Noise points, skip them
                continue

            # Get indices of points in the current cluster
            cluster_indices = segment_indices[labels == cluster_label]
            pts_copy[cluster_indices, 3] = id_counter
            id_counter += 1

    return pts_copy

def detect(lazfile, params: dict, viz=False):
    """
    Function that detects all the planes in the input LAZ file.

    Inputs:
      lazfile: a laspy input file
      params: a dictionary with all the parameters necessary for the algorithm
      viz: whether the visualiser (rerun, or polyscope) should be displaying results or not

    Output:
      - a NumPy array Nx4; each point has x-y-z-segmentid
    """

    start_time = time.time()

    # get parameters
    k = params.get("k", 500)
    min_score = params.get("min_score", 300)
    epsilon = params.get("epsilon", 0.1)
    multiplier = params.get("multiplier", 5.0)
    delta = params.get("delta", 4.0)
    min_samples = params.get("min_samples", 10)
    print(f"parameters:\nk: {k}\nmin_score: {min_score}\nepsilon: {epsilon}\nmultiplier: {multiplier}\ndelta: {delta}\nmin_samples: {min_samples}")
    
    ids = np.zeros(lazfile.header.point_count, dtype=int)
    pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, ids)).transpose()

    # pts, planes = ransac_simple.extract_planes(pts, min_score, k, epsilon)
    pts, planes = extract_planes(pts, min_score, k, epsilon)
    pts = post_process(pts, planes, epsilon, multiplier)

    # clustering
    pts = distance_cluster(pts, delta, min_samples)

    id_count = int(np.max(pts[:, 3])) + 1

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken is: {total_time:4f} s")
    print(f"Surfaces found: {id_count-1}")


    if viz:
        # -- init rerun viewer
        rr.init("myview", spawn=True)
        # -- log all the points
        rr.log("allpts", rr.Points3D(pts[:, :3], colors=[78, 205, 189], radii=0.1))
        # -- log each class one-by-one
        for i in range(id_count):
        # for i in range(101):
            subset = pts[pts[:, 3] == float(i)][:, :3]
            rr.log(
                "subset_{}".format(i),
                rr.Points3D(
                    subset[:],
                    colors=[
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    ],
                    radii=0.1,
                ),
            )
            rr.log(
                "logs_{}".format(i),
                rr.TextLog(
                    "size subset_{}=={}".format(i, subset.shape[0]),
                    level=rr.TextLogLevel.TRACE,
                ),
            )
            time.sleep(0.1)

    return pts
