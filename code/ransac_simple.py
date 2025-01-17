import numpy as np

def normal_from_3pt(pt1, pt2, pt3):
    v1 = pt2 - pt1
    v2 = pt3 - pt1
    normal = np.cross(v1, v2)

    # Normalize the normal vector
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise Exception(f"Points are collinear!\n{pt1}\n{pt2}\n{pt3}\n")
        
    return normal / normal_norm

def ransac(pts: np.ndarray, mask=None, iterations=1000, epsilon=0.1):
    """
    Simple RANSAC algorithm from terrainbook.
    
    Parameters:
    pts: numpy array of shape (N, 3) containing [x, y, z] coordinates
    iterations: number of iterations for RANSAC
    epsilon: maximum distance from point to plane to be considered an inlier
    
    Returns:

    """
    if not isinstance(mask, np.ndarray):
        mask = np.ones(len(pts), dtype=int)

    best_score = 0
    best_inliers = np.array([])
    best_plane = (0.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng()
    available = np.where(mask==1)[0]
    prob_mask = mask / len(available)

    for _ in range(iterations):

        sampled_indices = rng.choice(len(pts), size=3, replace=False, p=prob_mask)
        pt1, pt2, pt3 = pts[sampled_indices]
        normal = normal_from_3pt(pt1, pt2, pt3)
        
        # Formula: ax + by + cz + d = 0, calculate d using point p1
        d = -(np.dot(normal, pt1))
        distances = np.abs(pts.dot(normal) + d)
        
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
    pts3d = pts[:, :3].copy()
    mask = np.ones(len(pts), dtype=int)

    while True:
        score, inliers, plane = ransac(pts3d, mask, iterations, epsilon)

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
        print("best_score:", score)
        print("pts available:", len(np.where(mask==1)[0]))

        id_count += 1

    return pts, np.array(planes)