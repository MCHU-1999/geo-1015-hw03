# -- geo1015_hw03.py
# -- hw03 GEO1015.2024

import json
import os
import sys
from pathlib import Path

import houghtransform
import laspy
import ransac
import regiongrowing

RERUN_VIZ = False
RERUN_VIZ = True


def main():
    # -- get the path to the params.json (assuming the directory as in the git repository)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    param_path = dir_path + "/../data/params.json"
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
        write_ply(pts, "out_ransac.ply")
    if "RegionGrowing" in jparams:
        print("==> RegionGrowing")
        pts = regiongrowing.detect(lazfile, jparams["RegionGrowing"], RERUN_VIZ)
        write_ply(pts, "out_regiongrowing.ply")
    if "HoughTransform" in jparams:
        print("==> HoughTransform")
        pts = houghtransform.detect(lazfile, jparams["RegionGrowing"], RERUN_VIZ)
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
