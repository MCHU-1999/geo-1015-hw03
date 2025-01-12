import numpy as np
import math
import statistics
import sys


def perpendicular_distance(normal, vertex, sample):
    return abs(np.dot(np.asarray(sample) - np.asarray(vertex), normal))


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

    # 1. Create Accumulator using hesse normal form: For each spherical coordinate axis, create a list of possible values
    # 2. For each cell, convert its spherical coordinates into a unit vector and a point
    # 3. For each sample, and then for each cell, subtract the samples position from the cells point, then take the dot product
    #    of the resulting vector and the plane unit vector (in that order)

    # Theta is the angle between the normal of the plane and the X axis: 360 degrees
    # Phi is the angle between the normal of the plane and the Z axis: 180 degrees
    #    Probably a smart idea to consider the lowest available point the origin Z, and then only have to use 90 degrees
    # Rho is the distance between the origin and the point on the plane

    # X = rho * sin(theta) * cos(phi)
    # Y = rho * sin(theta) * sin(phi)
    # Z = rho * cos(theta)

    epsilon = params["epsilon"]

    h = lazfile.header
    extent = [*h.min, *h.max]

    theta_max = math.radians(360)
    theta_segment_count = 30
    theta_range = np.linspace(0, theta_max, theta_segment_count)

    phi_max = math.radians(180)
    phi_segment_count = 30
    phi_range = np.linspace(0, phi_max, phi_segment_count)

    rho_max = math.ceil(math.sqrt(abs(extent[0] - extent[3]) ** 2 + abs(extent[1] - extent[4]) ** 2 + abs(extent[2] - extent[5]) ** 2) / 2)
    rho_segment_count = 30
    rho_range = np.linspace(epsilon, rho_max, rho_segment_count)

    theta_values, phi_values, rho_values = np.meshgrid(theta_range, phi_range, rho_range)

    spherical_coordinates = np.stack([theta_values, phi_values, rho_values], axis=3)

    spherical_coordinates = spherical_coordinates.reshape(theta_segment_count * phi_segment_count * rho_segment_count, 3)

    plane_vectors = [
        (rho * math.sin(theta) * math.cos(phi), rho * math.sin(theta) * math.sin(phi), rho * math.cos(theta))
        for theta, phi, rho in spherical_coordinates
    ]

    plane_unit_vectors = plane_vectors / np.sqrt(np.einsum("...i,...i", plane_vectors, plane_vectors))[..., None]

    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()

    points = points - np.array([statistics.mean(lazfile.x), statistics.mean(lazfile.y), statistics.mean(lazfile.z)])

    for point in points:
        point_to_plane_vectors = plane_vectors - point

        dot_product = np.einsum("ij, ij->i", plane_unit_vectors, point_to_plane_vectors)

        dot_product = abs(dot_product)

        print(dot_product[:10])

        sys.exit()

    segment_ids = np.random.randint(low=0, high=10, size=lazfile.header.point_count)
    pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, segment_ids)).transpose()

    return pts


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    b = np.array([1, 2, 3])[..., None]

    print(np.divide(a, b))
