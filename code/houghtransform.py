import numpy as np
import math
import statistics
import sys
import rerun as rr
import time


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
    # Rho is the distance between the origin and the point on the plane

    # X = rho * sin(theta) * cos(phi)
    # Y = rho * sin(theta) * sin(phi)
    # Z = rho * cos(theta)

    epsilon = params["epsilon"]
    alpha = params["alpha"]

    h = lazfile.header
    extent = [*h.min, *h.max]

    theta_max = math.radians(180)
    theta_segment_count = 15
    theta_range = np.linspace(0, theta_max, theta_segment_count, endpoint=False)

    phi_max = math.radians(360)
    phi_segment_count = 15
    phi_range = np.linspace(0, phi_max, phi_segment_count, endpoint=False)

    rho_max = math.ceil(math.sqrt(abs(extent[0] - extent[3]) ** 2 + abs(extent[1] - extent[4]) ** 2 + abs(extent[2] - extent[5]) ** 2) / 2)
    rho_segment_count = 15
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

    centered_points = points - np.array([statistics.mean(lazfile.x), statistics.mean(lazfile.y), statistics.mean(lazfile.z)])

    ########################################################################

    accumulator = [[] for _ in range(len(plane_vectors))]

    for point_index, centered_point in enumerate(centered_points):
        point_to_plane_vectors = plane_vectors - centered_point

        dot_product = np.einsum("ij, ij->i", plane_unit_vectors, point_to_plane_vectors)

        dot_product = abs(dot_product)

        mask = [1 if x <= epsilon else 0 for x in dot_product]

        for plane_index, mask_value in enumerate(mask):
            if mask_value == 1:
                accumulator[plane_index].append(point_index)

    points_with_ids = [None for _ in range(len(points))]

    accumulator.sort(key=len, reverse=True)

    #######################################################################

    plane_counter = 0

    while plane_counter < len(accumulator):
        current_plane = accumulator[plane_counter]

        for i in range(plane_counter + 1, len(accumulator) - 1, 1):
            target_plane = accumulator[i]

            accumulator[i] = [x for x in target_plane if x not in current_plane]

        accumulator = [x for x in accumulator if len(x) >= alpha]

        plane_counter += 1

        accumulator.sort(key=len, reverse=True)

        print(len(accumulator))

    ########################################################################

    current_id = -1

    for plane_index, plane in enumerate(accumulator):
        if len(plane) < alpha:
            continue

        id_updated = False

        for point_index in plane:
            if points_with_ids[point_index] is None:
                if id_updated == False:
                    id_updated = True
                    current_id += 1

                points_with_ids[point_index] = np.append(points[point_index], current_id)

        done = True

        for point_with_id in points_with_ids:
            if point_with_id is None:
                done = False
                break

        if done:
            break

    ########################################################################

    for point_index, point_with_id in enumerate(points_with_ids):
        if point_with_id is None:
            points_with_ids[point_index] = np.append(points[point_index], -1)

    if viz:
        rerun_visualization(np.asarray(points_with_ids))

    return np.asarray(points_with_ids)

    # segment_ids = np.random.randint(low=0, high=10, size=lazfile.header.point_count)
    # pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, segment_ids)).transpose()
    # return pts


def rerun_visualization(pts):
    # -- init rerun viewer

    rr.init("myview", spawn=True)
    # -- log all the points

    rr.log("allpts", rr.Points3D(pts[:, :3], colors=[78, 205, 189], radii=0.1))
    # -- log each class one-by-one
    for i in range(int(np.max(pts[:, 3])) + 1):  # Changed this since I don't have an ID count anymore
        subset = pts[pts[:, 3] == i][:, :3]
        rr.log(
            "subset_{}".format(i),
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
            "logs_{}".format(i),
            rr.TextLog(
                "size subset_{}=={}".format(i, subset.shape[0]),
                level=rr.TextLogLevel.TRACE,
            ),
        )
        time.sleep(0.1)


if __name__ == "__main__":
    a = np.zeros(5)

    b = np.append(a, 1)

    print(b)
