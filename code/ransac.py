import time
import math
import numpy as np
import rerun as rr
import scipy.spatial
import ransac_simple


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
        # distances, re = kdtree.query(kdtree.data[sampled_index], k=30)
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
        pts: a NumPy array Nx4; each point has x-y-z-segmentid.
        scores: a NumPy array of inline scores except uncatagorized (id = 0).
        planes: a NumPy array of planes parameters [[a, b, c, d]...] satisfy ax + by + cz + d = 0. 
    """

    planes = []
    scores = []
    id_count = 1
    invalid_count = 0
    kdtree = scipy.spatial.KDTree(pts[:, :3])
    mask = np.ones(len(pts), dtype=int)

    while True:
        score, inliers, plane = ransac(kdtree, mask, iterations, epsilon)

        if invalid_count > 10:
            break

        if score < min_score:
            invalid_count += 1
            continue
        else:
            invalid_count = 0

        pts[inliers, 3] = id_count
        mask[inliers] = 0
        planes.append(plane)
        scores.append(score)
        
        print("\nid:", id_count)
        print("score:", score)
        print("pts available:", len(np.where(mask==1)[0]))

        id_count += 1

    return pts, np.array(scores), np.array(planes)

def post_process(pts: np.ndarray, scores: np.ndarray, planes: np.ndarray, epsilon=0.1):
    """
    Extract multiple planes from point cloud using RANSAC
    
    Parameters:
        pts: numpy array of shape (N, 4) containing [x, y, z, id] coordinates
        scores: 
        planes: 
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
        radius = epsilon * 5
        for index in collison:
            pts[index, 3] = 0
            n_indices = np.array(kdtree.query_ball_point(pts[index, :3], radius))
            indices = n_indices[pts_copy[n_indices, 3] > 0]

            if len(indices) == 0:
                pts[index, 3] = i + 1
                continue
            
            uniq = np.unique_counts(pts[indices, 3])
            max_index = np.argmax(uniq.counts)
            id = uniq.values[max_index]
            pts[index, 3] = id

    return pts


def merge_planes(pts: np.ndarray, planes: np.ndarray, angle_threshold=2, distance_threshold=0.1):
    """
    Merge similar planes after RANSAC.

    Parameters:
        pts (numpy.ndarray): Array of shape (N, 4) containing [x, y, z, id] coordinates.
        planes (list of tuples): List of plane equations (A, B, C, D).
        angle_threshold (float): Threshold for the angle difference between planes.
        distance_threshold (float): Threshold for the distance difference between planes.

    Returns:
        pts (numpy.ndarray): Array of shape (N, 4) containing [x, y, z, new_id] coordinates.
    """
    radians_threshold = math.radians(angle_threshold)
    planes_all = np.vstack(([1, 1, 1, np.inf], planes))
    num_planes = len(planes)
    merge_map = np.arange(num_planes)

    # Check pairwise similarity between planes
    for i in range(num_planes):
        if i == 0:
            continue
        for j in range(i + 1, num_planes):
            # Get plane normals and constants
            a1, b1, c1, d1 = planes_all[i]
            a2, b2, c2, d2 = planes_all[j]
            n1 = np.array([a1, b1, c1])
            n2 = np.array([a2, b2, c2])

            # Check angle between normals
            angle_cos = np.dot(n1, n2)
            angle = math.acos(angle_cos)
            if angle > radians_threshold:
                continue  # Normals are not parallel

            # Check distance difference
            if abs(d1 - d2) > distance_threshold:
                continue  # Planes are too far apart

            # Merge plane j into plane i
            merge_map[j] = merge_map[i]
            print(f"merged plane {j} into plane {i}")

    # Create new ids based on merged planes
    for old_id, new_id in enumerate(merge_map):
        pts[pts[:, 3] == old_id, 3] = new_id

    return pts

def detect(lazfile, params, viz=False):
    """
    Function that detects all the planes in the input LAZ file.

    Inputs:
      lazfile: a laspy input file
      params: a dictionary with all the parameters necessary for the algorithm
      viz: whether the visualiser (rerun, or polyscope) should be displaying results or not

    Output:
      - a NumPy array Nx4; each point has x-y-z-segmentid
    """
    ids = np.zeros(lazfile.header.point_count, dtype=int)
    pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, ids)).transpose()

    # pts, scores, planes = ransac_simple.extract_planes(pts, params["min_score"], params["k"], params["epsilon"])
    # pts, scores, planes = ransac_simple.extract_planes(pts, 500, 1000, 0.1)
    # pts, scores, planes = extract_planes(pts, params["min_score"], params["k"], params["epsilon"])
    pts, scores, planes = extract_planes(pts, 50, 1000, 0.1)
    pts = post_process(pts, scores, planes, 0.1)

    id_count = len(scores) + 1

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
            time.sleep(0.5)

    return pts
