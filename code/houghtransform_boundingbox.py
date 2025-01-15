import math
import statistics
import sys
import time
from itertools import compress

import numpy as np
import rerun as rr


def perpendicular_distance(normal, vertex, sample):
    return abs(np.dot(np.asarray(sample) - np.asarray(vertex), normal))


def plane_intersects_bbox(plane_point, plane_normal, bbox_min, bbox_max):
    # Generate the 8 corners of the bounding box
    bbox_corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ])

    # Calculate the signed distance from the plane for each corner
    distances = np.dot(bbox_corners - plane_point, plane_normal)
    # print(distances)

    # Check if there are both positive and negative distances (intersection occurs)
    # Like a knife slicing a cake, if there is cake on both sides, it cuts, if there is only cake on 1 it doesn't
    return np.any(distances <= 0) and np.any(distances >= 0)


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

    start_time = time.time()

    epsilon = params["epsilon"]
    alpha = params["alpha"]

    h = lazfile.header
    extent = [*h.min, *h.max]

    # good: 10, 10, 0.2

    theta_max = math.radians(180)
    theta_segment_size = 10
    theta_segment_count = round(360 / theta_segment_size)
    theta_range = np.linspace(0, theta_max, theta_segment_count, endpoint=False)

    phi_max = math.radians(360)
    phi_segment_size = 10
    phi_segment_count = round(360 / phi_segment_size)
    phi_range = np.linspace(0, phi_max, phi_segment_count, endpoint=False)

    rho_max = math.ceil(math.sqrt(
        abs(extent[0] - extent[3]) ** 2 + abs(extent[1] - extent[4]) ** 2 + abs(extent[2] - extent[5]) ** 2) / 2)
    rho_segment_size = 0.2
    rho_segment_count = round((rho_max - epsilon) / rho_segment_size)
    rho_range = np.linspace(epsilon, rho_max, rho_segment_count)

    theta_values, phi_values, rho_values = np.meshgrid(theta_range, phi_range, rho_range)

    spherical_coordinates = np.stack([theta_values, phi_values, rho_values], axis=3)

    spherical_coordinates = spherical_coordinates.reshape(theta_segment_count * phi_segment_count * rho_segment_count,
                                                          3)

    # Creation plane vectors
    plane_vectors = [
        (rho * math.sin(theta) * math.cos(phi), rho * math.sin(theta) * math.sin(phi), rho * math.cos(theta))
        for theta, phi, rho in spherical_coordinates
    ]

    # Retrieval of points, randomising them and shifting them around (0, 0, 0).
    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()
    rng = np.random.default_rng()
    rng.shuffle(points)
    center_point = np.array([statistics.mean(lazfile.x), statistics.mean(lazfile.y), statistics.mean(lazfile.z)])
    centered_points = points - center_point

    # Definition of all the planes, without filtering or anything
    plane_vectors = np.asarray(plane_vectors)
    print(f"Number of planes before filtering: {len(plane_vectors)}")
    plane_unit_vectors = plane_vectors / np.sqrt(np.einsum("...i,...i", plane_vectors, plane_vectors))[..., None]

    # -----------------------------------------------------------------------------------------------
    # Bounding Box filtering
    # Retrieval of the bounding box, using the centered_points, aka translated to the origin.
    bbox_min = np.min(centered_points, axis=0)
    bbox_max = np.max(centered_points, axis=0)

    # Filter planes based on bounding box intersection
    filtered_planes = []
    filtered_plane_normals = []
    for i, (plane_vector, plane_normal) in enumerate(zip(plane_vectors, plane_unit_vectors)):
        if plane_intersects_bbox(plane_vector, plane_normal, bbox_min, bbox_max):
            filtered_planes.append(plane_vector)
            filtered_plane_normals.append(plane_normal)

    # Convert back to numpy arrays
    plane_vectors = np.array(filtered_planes)
    print(f"Number of planes after filtering: {len(plane_vectors)}")
    plane_unit_vectors = np.array(filtered_plane_normals)
    # End of Bounding Box filtering
    # -----------------------------------------------------------------------------------------------

    ########################################################################

    print("Started Voting")

    voting_start_time = time.time()

    section_length = round(len(points) / 30)

    accumulator = [[] for _ in range(len(plane_vectors))]

    for point_index, centered_point in enumerate(centered_points):
        point_to_plane_vectors = plane_vectors - centered_point

        dot_product = np.einsum("ij, ij->i", plane_unit_vectors, point_to_plane_vectors)

        dot_product = abs(dot_product)

        mask = [1 if x <= epsilon else 0 for x in dot_product]

        for plane_index, mask_value in enumerate(mask):
            if mask_value == 1:
                accumulator[plane_index].append(point_index)

        # One time plane count reduction
        if point_index == section_length:
            sorted_accumulator = accumulator[:]
            sorted_accumulator.sort(key=len, reverse=True)
            plane_accumulation = [len(x) for x in sorted_accumulator]

            top_median = statistics.median(plane_accumulation[:section_length])

            mask = [False if len(x) >= top_median else True for x in accumulator]

            plane_vectors = np.delete(plane_vectors, mask, axis=0)
            plane_unit_vectors = np.delete(plane_unit_vectors, mask, axis=0)

            print("Accumulator size pre-removal: ", len(accumulator))

            accumulator = list(compress(accumulator, [not x for x in mask]))

            print("Accumulator size post-removal: ", len(accumulator))

            removal_time = time.time()

            print("time to removal: ", removal_time - voting_start_time)

    voting_end_time = time.time()

    print("voting time: ", voting_end_time - voting_start_time)

    points_with_ids = [None for _ in range(len(points))]

    accumulator.sort(key=len, reverse=True)

    #######################################################################

    print("Started Consolidating")

    plane_counter = 0

    while plane_counter < len(accumulator):
        print(len(accumulator))
        current_plane = accumulator[plane_counter]

        for i in range(plane_counter + 1, len(accumulator) - 1, 1):
            target_plane = accumulator[i]

            accumulator[i] = [x for x in target_plane if x not in current_plane]

        accumulator = [x for x in accumulator if len(x) >= alpha]

        plane_counter += 1

        accumulator.sort(key=len, reverse=True)

    ########################################################################

    current_id = -1

    for plane_index, plane in enumerate(accumulator):
        if len(plane) < alpha:
            continue

        id_updated = False

        for point_index in plane:
            if points_with_ids[point_index] is None:
                if not id_updated:
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

    end_time = time.time()

    total_time = end_time - start_time

    print(f"Time taken is: {total_time}")

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
    a = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    b = [True, False, True, False, True]

    c = np.delete(a, b, axis=0)

    print(c)
