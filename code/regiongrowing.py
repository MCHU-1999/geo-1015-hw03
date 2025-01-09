import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import svd
import rerun

def detect(lazfile, params, viz=False):
    """
    !!! TO BE COMPLETED !!!
    !!! You are free to subdivide the functionality of this function into several functions !!!

    Function that detects all the planes in the input LAZ file.

    Inputs:
      lazfile: a laspy input file
      params: a dictionary with all the parameters necessary for the algorithm
      viz: whether the visualiser (rerun, or polyscope) should be displaying results or not

    Output:
      - a NumPy array Nx4; each point has x-y-z-segmentid
    """

    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()

    k = params["k"]  # number of nearest neighbors
    max_angle = params["max_angle"]  # angle threshold in degrees

    # Create a KDTree nearest neighbor search
    tree = KDTree(points)

    def compute_normals_batch(points, tree, k, batch_size=1000):
        """
        Compute normals for points in batches.

        Inputs:
          points: Nx3 array of point coordinates.
          tree: KDTree for nearest neighbor search.
          k: Number of nearest neighbors.
          batch_size: Number of points to process in each batch.

        Output:
          normals: Nx3 array of normal vectors.
        """
        n = len(points)
        normals = np.zeros_like(points)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_points = points[start:end]

            # find k-nearest neighbors for the batch
            distances, indices = tree.query(batch_points, k=k + 1)

            # compute normals for the batch
            for i, (point, neighbor_indices) in enumerate(zip(batch_points, indices)):
                neighbors = points[neighbor_indices[1:]]  # Exclude the point itself
                neighbors_centered = neighbors - np.mean(neighbors, axis=0)
                cov_matrix = np.cov(neighbors_centered, rowvar=False)
                _, _, vh = svd(cov_matrix)
                normal = vh[2]
                if normal[2] < 0:
                    normal *= -1
                normals[start + i] = normal

        return normals

    normals = compute_normals_batch(points, tree, k, batch_size=1000)


    def region_growing(points, normals, k, max_angle):
        max_angle_rad = np.deg2rad(max_angle)  # Convert to radians
        n = len(points)
        segment_ids = np.zeros(n, dtype=int)  # 0 means unassigned
        current_segment_id = 1
        processed = np.zeros(n, dtype=bool)  # Track processed points

        for i in range(n):
            if processed[i]:
                continue  # Skip already processed points

            # Initialize region
            S = [i]  # Use a list
            R = []  # Current region

            while S:
                p = S.pop(0)  # Pop from the beginning of the list
                R.append(p)
                segment_ids[p] = current_segment_id
                processed[p] = True  # Mark the point as processed

                # Find neighbors of p
                neighbors = tree.query(points[p], k=k + 1)[1][1:]  # Exclude p itself

                for c in neighbors:
                    if processed[c]:
                        continue  # Skip already processed points

                    # if c fits with R
                    angle = np.arccos(np.clip(np.dot(normals[p], normals[c]), -1.0, 1.0))
                    if angle < max_angle_rad:
                        S.append(c)

            # increment segment ID for the next region
            current_segment_id += 1

        return segment_ids

    segment_ids = region_growing(points, normals, k, max_angle)

    # points and segment IDs into the required Nx4 array
    result = np.hstack((points, segment_ids.reshape(-1, 1)))

    return result

# segment_ids = np.random.randint(low=0, high=10, size=lazfile.header.point_count)
    # pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, segment_ids)).transpose()
    # print(pts)
    # return pts
