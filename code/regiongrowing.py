import numpy as np


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
    segment_ids = np.random.randint(low=0, high=10, size=lazfile.header.point_count)
    pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, segment_ids)).transpose()
    print(pts)
    return pts
