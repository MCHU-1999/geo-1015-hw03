import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import svd
import rerun as rr
import time


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

    # Extract points from the LAZ file
    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()
    print("Points shape:", points.shape)  # Debug print

    # Extract parameters
    k = params["k"]  # Number of nearest neighbors
    max_angle = params["max_angle"]  # Angle threshold in degrees

    # Create a KDTree for nearest neighbor search
    tree = KDTree(points)

    # Function to compute normals and plane fitting errors together
    def compute_normals_and_fitting_error(points, tree, k):
        """
        Compute normals and geometric features using PCA of local neighborhoods.

        Inputs:
          points: Nx3 array of point coordinates
          tree: KDTree for nearest neighbor search
          k: Number of nearest neighbors (10-20 recommended)

        Outputs:
          normals: Nx3 array of normal vectors
          planarity: Nx1 array of planarity values (higher means more planar)
        """
        n = len(points)
        normals = np.zeros_like(points)
        planarity = np.zeros(n)

        for i in range(n):
            # Find k nearest neighbors
            distances, indices = tree.query(points[i], k=k + 1)
            neighbors = points[indices[1:]]  # Exclude the point itself

            # Center the points
            centroid = np.mean(neighbors, axis=0)
            neighbors_centered = neighbors - centroid

            # Compute covariance matrix
            cov_matrix = np.cov(neighbors_centered.T)

            try:
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # Sort eigenvalues and eigenvectors in descending order
                idx = eigenvalues.argsort()[::-1]  # Reverse to get descending order
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # The normal vector is the eigenvector corresponding to smallest eigenvalue
                normal = eigenvectors[:, 2]  # Last column after sorting

                # Ensure normal points upward (positive z)
                if normal[2] < 0:
                    normal = -normal

                normals[i] = normal

                # Compute planarity feature: (λ2 - λ3)/λ1
                # Higher value means more planar (closer to perfect plane)
                planarity[i] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]

            except np.linalg.LinAlgError:
                normals[i] = np.array([0, 0, 1])
                planarity[i] = 0

            if i % 1000 == 0:
                print(f"Processed {i}/{n} points")

        return normals, planarity

    def select_seeds(planarity, min_seeds=1000):
        """
        Select seed points based on planarity score.
        Higher planarity means better plane fit.
        """
        # Sort points by planarity (descending)
        sorted_indices = np.argsort(-planarity)  # Negative to sort in descending order

        # Take points with highest planarity
        n_seeds = max(min_seeds, len(planarity) // 50)  # At least min_seeds or 2% of points
        return sorted_indices[:n_seeds]

    def region_growing(points, normals, k, max_angle, tree, seed_points):
        """
        Region growing using normal similarity.
        """
        max_angle_rad = np.deg2rad(max_angle)
        n = len(points)
        processed = np.zeros(n, dtype=bool)
        regions = []  # LR in pseudocode
        min_region_size = 10  # Minimum points for a valid region

        for seed in seed_points:  # for each s in LS do
            if processed[seed]:
                continue

            S = {seed}  # S ← {s}
            R = set()  # R ← ∅
            region_normals = []

            while S:  # while S is not empty do
                p = S.pop()  # p ← pop(S)

                # Find neighbours(p)
                _, neighbors = tree.query(points[p], k=k + 1)

                # foreach candidate point c ∈ neighbours(p) do
                for c in neighbors[1:]:  # Skip first neighbor (point itself)
                    # Check if point fits regardless of processed state
                    if R:
                        region_normal = np.mean(region_normals, axis=0)
                        region_normal = region_normal / np.linalg.norm(region_normal)
                    else:
                        region_normal = normals[seed]

                    # Check if point fits with region
                    angle = np.arccos(np.clip(np.abs(np.dot(region_normal, normals[c])), -1.0, 1.0))

                    # Only add to region if unprocessed AND fits
                    if angle < max_angle_rad and not processed[c]:
                        S.add(c)  # add c to S
                        R.add(c)  # add c to R
                        processed[c] = True
                        region_normals.append(normals[c])

            # append R to LR if it meets minimum size
            if len(R) >= min_region_size:
                regions.append(list(R))
                print(f"Found region {len(regions)} with {len(R)} points")

        # Convert to segment IDs format
        segment_ids = np.zeros(n, dtype=int)
        for i, region in enumerate(regions, start=1):
            segment_ids[list(region)] = i

        return segment_ids

    # Main processing pipeline
    print("Computing normals and fitting errors...")
    normals, fitting_errors = compute_normals_and_fitting_error(points, tree, k)

    # Select seed points based on fitting errors
    print("Selecting seed points...")
    seed_points = select_seeds(fitting_errors)
    print(f"Selected {len(seed_points)} seed points")

    # Perform region growing
    print("Growing regions...")
    segment_ids = region_growing(points, normals, k, max_angle, tree, seed_points)

    # Combine results
    result = np.hstack((points, segment_ids.reshape(-1, 1)))

    # Visualize results
    if viz:
        # Initialize rerun viewer
        rr.init("plane_detection", spawn=True)

        # Log all points with a default color
        rr.log("all_points", rr.Points3D(points, colors=[78, 205, 189], radii=0.1))

        # Log each segment with a unique random color
        unique_segment_ids = np.unique(segment_ids)
        for segment_id in unique_segment_ids:
            subset = points[segment_ids == segment_id]
            rr.log(
                f"segment_{segment_id}",
                rr.Points3D(
                    subset,
                    colors=[
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    ],
                    radii=0.1,
                ),
            )
            rr.log(
                f"logs_{segment_id}",
                rr.TextLog(
                    f"size segment_{segment_id} == {subset.shape[0]}",
                    level=rr.TextLogLevel.TRACE,
                ),
            )
            time.sleep(0.5)

    return result

