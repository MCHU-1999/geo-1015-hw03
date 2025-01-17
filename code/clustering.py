import numpy as np
from scipy.spatial import KDTree


# This is actually not DBscan AT ALL!
def cluster_by_distance(points, distance_threshold, min_samples):
    """
    Cluster points based purely on distance using an iterative approach.
    
    Parameters:
    points: np.array of shape (n_points, n_dimensions)
    distance_threshold: float, maximum distance between connected points
    min_samples: minimum points in a sub-cluster, clusters with pts amount below this threshold will be mark as noise
    
    Returns:
    labels: np.array of shape (n_points,) containing cluster labels
    n_clusters: number of clusters found
    """
    n_points = len(points)
    labels = np.full(n_points, -1)
    current_label = 0
    
    tree = KDTree(points)
    
    # Process each unassigned point
    for point_idx in range(n_points):
        if labels[point_idx] != -1:  # Skip if already assigned
            continue
            
        # Start new cluster
        to_process = [point_idx]
        labels[point_idx] = current_label
        
        # Process points in the cluster
        i = 0
        while i < len(to_process):
            current_idx = to_process[i]
            
            # Find neighbors
            neighbors = tree.query_ball_point(points[current_idx], distance_threshold)
            
            # Add unprocessed neighbors to the list
            for neighbor_idx in neighbors:
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = current_label
                    to_process.extend([neighbor_idx])
            
            i += 1

        current_label += 1

    for l in range(current_label):
        if len(labels[labels == l]) < min_samples:
            labels[labels == l] = -1
    
    return labels


