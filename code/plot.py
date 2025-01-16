import matplotlib.pyplot as plt
import numpy as np

import json
import os
import sys
from pathlib import Path

import houghtransform
import houghtransform_boundingbox
import laspy
import ransac
import regiongrowing

RERUN_VIZ  = False

def visualize_point_cloud(pts, algo, params: dict):
    """
    Visualizes a 3D point cloud with random colors for each segment.

    Parameters:
        pts (numpy.ndarray): A 2D array of shape (N, 4) where each row is [x, y, z, segment_id].
    """
    if pts.shape[1] != 4:
        raise ValueError("Input pts must have shape (N, 4) with [x, y, z, segment_id].")
    
    points = pts[pts[:, 3] != 0]
    # Extract x, y, z, and segment_id
    x, y, z, segment_ids = points.T
    
    # Get unique segment IDs and assign random colors
    unique_segments = np.unique(segment_ids)
    np.random.seed(420)
    colors = {seg: np.random.rand(3,) for seg in unique_segments}
    
    # Map colors to pts
    point_colors = np.array([colors[seg] for seg in segment_ids])
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    ax.scatter(x, y, z, c=point_colors, s=5, alpha=0.8)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title(f"3D Point Cloud from {algo}")

    # Adjust for better visualization
    ax.view_init(elev=15, azim=15)
    plt.subplots_adjust(left=0.0, right=0.9, top=1.0, bottom=0.0)

    # Add notes beside the plot
    param_str = [f"{key}: {value}" for key, value in params.items()]
    param_str.insert(0, "Parameters")
    note = "\n".join(param_str)
    fig.text(0.75, 0.15, note, fontsize=16, va='center', ha='left', family='monospace')
    
    # plt.show()
    plt.savefig(f"../images/{algo}_plot.png")


def main():
    # -- get the path to the params.json (assuming the directory as in the git repository)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # param_path = dir_path + "/../data/params.json"
    param_path = dir_path + "/../data/params_houghtransform.json"
    # -- or you can supply the params.json as the first argument to this program
    if len(sys.argv) == 2:
        param_path = sys.argv[1]
    # -- change to the directory containing the params.json (so that file paths in the params.json are read relative to that directory)
    os.chdir(Path(param_path).parent)
    print("Active working directory: " + os.getcwd())
    jparams = json.load(open(param_path))

    try:
        lazfile = laspy.read(jparams["input_file"])
    except Exception as e:
        print(e)
        sys.exit()

    if "RANSAC" in jparams:
        print("==> RANSAC")
        pts = ransac.detect(lazfile, jparams["RANSAC"], RERUN_VIZ)
        visualize_point_cloud(pts, "RANSAC", jparams["RANSAC"])
        write_ply(pts, "out_ransac.ply")
    if "RegionGrowing" in jparams:
        print("==> RegionGrowing")
        pts = regiongrowing.detect(lazfile, jparams["RegionGrowing"], RERUN_VIZ)
        visualize_point_cloud(pts, "RegionGrowing", jparams["RANSAC"])
        write_ply(pts, "out_regiongrowing.ply")
    if "HoughTransform" in jparams:
        print("==> HoughTransform")
        # pts = houghtransform.detect(lazfile, jparams["HoughTransform"], RERUN_VIZ)
        pts = houghtransform_boundingbox.detect(lazfile, jparams["HoughTransform"], RERUN_VIZ)
        visualize_point_cloud(pts, "HoughTransform", jparams["RANSAC"])
        write_ply(pts, "out_houghtransform.ply")


def write_ply(pts, filename):
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(pts.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int segment_id\n")
        f.write("end_header\n")
        for pt in pts:
            f.write("{} {} {} {}\n".format(pt[0], pt[1], pt[2], int(pt[3])))


if __name__ == "__main__":
    main()

