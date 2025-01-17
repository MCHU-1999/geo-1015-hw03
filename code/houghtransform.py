import numpy as np
import math
import statistics
import sys
import rerun as rr
import time
from scipy.spatial import cKDTree
from itertools import compress
from collections import Counter


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

    # MAYBE TODO:
    # Check if point in plane during plane cleaning
    # Add cleaning parameters to params

    start_time = time.time()

    epsilon = params["epsilon"]
    alpha = params["alpha"]
    chunk_size = params["chunk size"]
    reprocessing = params["reprocessing"]

    mean_x = statistics.mean(lazfile.x)
    mean_y = statistics.mean(lazfile.y)
    mean_z = statistics.mean(lazfile.z)

    plane_points, plane_normals, plane_ids = generate_planes(lazfile, params)

    chunks = generate_pointcloud_chunks(lazfile, chunk_size)

    chunks = [item for row in chunks for item in row]

    processed_points = []

    for index, chunk in enumerate(chunks):
        if len(chunk) < alpha:
            continue

        print("Processing chunk: ", index + 1, "of: ", len(chunks))
        processed_points.append(process_chunk(chunk, params, plane_points, plane_normals, plane_ids))

    processed_points = np.concatenate(processed_points)

    if reprocessing:
        print("Beginning Reprocessing")

        good_planes_ids = np.unique(processed_points[:, -1])

        print("Number of planes for reprocessing: ", len(good_planes_ids))

        good_planes_ids = good_planes_ids.astype("int")

        good_plane_points = plane_points[good_planes_ids]
        good_plane_normals = plane_normals[good_planes_ids]

        points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()

        rng = np.random.default_rng(5)

        rng.shuffle(points)

        centered_points = points - np.array([mean_x, mean_y, mean_z])

        params["acceleration factor"] = 1

        processed_points = process_chunk(centered_points, params, good_plane_points, good_plane_normals, good_planes_ids)

    else:
        processed_points = np.asarray(processed_points)

    # processed_points = plane_cleaning(processed_points, params)

    processed_points[:, -1] = np.unique(processed_points[:, -1], return_inverse=True)[1]

    processed_points[:, 0] += mean_x
    processed_points[:, 1] += mean_y
    processed_points[:, 2] += mean_z

    end_time = time.time()

    total_time = end_time - start_time

    print(f"Time taken is: {total_time}")

    if viz:
        rerun_visualization(processed_points)

    return processed_points


def process_chunk(points, params, plane_points, plane_normals, plane_ids):
    epsilon = params["epsilon"]
    alpha = params["alpha"]
    acceleration_factor = params["acceleration factor"]

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)

    invalid_plane_mask = plane_intersects_bbox(plane_points, plane_normals, bbox_min, bbox_max)

    print("Total planes: ", len(plane_points))

    plane_points = np.delete(plane_points, invalid_plane_mask, axis=0)
    plane_normals = np.delete(plane_normals, invalid_plane_mask, axis=0)
    plane_ids = np.delete(plane_ids, invalid_plane_mask, axis=0)

    print("Reduced planes: ", len(plane_points))

    ########################################################################

    print("Started Voting")

    voting_start_time = time.time()

    section_length = round(len(points) / acceleration_factor)
    # section_length = 100000000

    accumulator = [[] for _ in range(len(plane_points))]

    for point_index, centered_point in enumerate(points):
        point_to_plane_point_vectors = plane_points - centered_point

        dot_product = np.einsum("ij, ij->i", plane_normals, point_to_plane_point_vectors)

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

            plane_points = np.delete(plane_points, mask, axis=0)
            plane_normals = np.delete(plane_normals, mask, axis=0)
            plane_ids = np.delete(plane_ids, mask, axis=0)

            print("Accumulator size pre-removal: ", len(accumulator))

            accumulator = list(compress(accumulator, [not x for x in mask]))

            print("Accumulator size post-removal: ", len(accumulator))

            removal_time = time.time()

            print("time to removal: ", removal_time - voting_start_time)

    voting_end_time = time.time()

    print("voting time: ", voting_end_time - voting_start_time)

    #######################################################################

    print("Started Consolidating")

    accumulator = list(zip(accumulator, plane_ids))

    accumulator = [[set(x[0]), x[1]] for x in accumulator]

    accumulator.sort(key=lambda x: len(x[0]), reverse=True)

    plane_counter = 0

    while plane_counter < len(accumulator):
        # print(len(accumulator))
        current_plane = accumulator[plane_counter][0]

        for i in range(plane_counter + 1, len(accumulator) - 1, 1):
            target_plane = accumulator[i][0]

            accumulator[i][0] = target_plane - current_plane

        accumulator = [x for x in accumulator if len(x[0]) >= alpha]

        plane_counter += 1

        accumulator.sort(key=lambda x: len(x[0]), reverse=True)

    print("Final plane count: ", len(accumulator))

    ########################################################################

    points_with_ids = [None for _ in range(len(points))]

    for plane in accumulator:
        plane_id = plane[1]

        for point_index in plane[0]:
            points_with_ids[point_index] = np.append(points[point_index], plane_id)

    for point_index, point_with_id in enumerate(points_with_ids):
        if point_with_id is None:
            points_with_ids[point_index] = np.append(points[point_index], -1)

    points_with_ids = np.asarray(points_with_ids)

    return points_with_ids


def generate_planes(lazfile, params):
    epsilon = params["epsilon"]
    alpha = params["alpha"]

    h = lazfile.header
    extent = [*h.min, *h.max]

    theta_max = math.radians(180)
    theta_segment_size = params["theta segment size"]
    theta_segment_count = round(360 / theta_segment_size)
    theta_range = np.linspace(0, theta_max, theta_segment_count, endpoint=False)

    phi_max = math.radians(360)
    phi_segment_size = params["phi segment size"]
    phi_segment_count = round(360 / phi_segment_size)
    phi_range = np.linspace(0, phi_max, phi_segment_count, endpoint=False)

    rho_max = math.ceil(math.sqrt(abs(extent[0] - extent[3]) ** 2 + abs(extent[1] - extent[4]) ** 2 + abs(extent[2] - extent[5]) ** 2) / 2)
    rho_segment_size = params["rho segment size"]
    rho_segment_count = round((rho_max - epsilon) / rho_segment_size)
    rho_range = np.linspace(epsilon, rho_max, rho_segment_count)

    theta_values, phi_values, rho_values = np.meshgrid(theta_range, phi_range, rho_range)

    spherical_coordinates = np.stack([theta_values, phi_values, rho_values], axis=3)

    spherical_coordinates = spherical_coordinates.reshape(theta_segment_count * phi_segment_count * rho_segment_count, 3)

    plane_points = [
        (rho * math.sin(theta) * math.cos(phi), rho * math.sin(theta) * math.sin(phi), rho * math.cos(theta))
        for theta, phi, rho in spherical_coordinates
    ]

    plane_points = np.asarray(plane_points)

    plane_normals = plane_points / np.sqrt(np.einsum("...i,...i", plane_points, plane_points))[..., None]

    plane_ids = np.arange(len(plane_points))

    return plane_points, plane_normals, plane_ids


def plane_intersects_bbox(plane_points, plane_normals, bbox_min, bbox_max):
    # Generate the 8 corners of the bounding box
    bbox_corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ]
    )

    plane_point_to_corner_vectors = bbox_corners - plane_points[:, np.newaxis]

    # distances = np.einsum("ijk, mk->ij", plane_point_to_corner_vectors, plane_normals) Works but is slow for some reason

    mask = []

    for index, vector in enumerate(plane_point_to_corner_vectors):
        distance = np.asarray(np.einsum("ij, j->i", vector, plane_normals[index]))

        mask.append(not (np.any(distance <= 0) and np.any(distance >= 0)))

    return mask


def generate_pointcloud_chunks(lazfile, chunk_size):
    mean_x = statistics.mean(lazfile.x)
    mean_y = statistics.mean(lazfile.y)
    mean_z = statistics.mean(lazfile.z)

    h = lazfile.header
    extent = [*h.min, *h.max]

    extent = extent - np.array([mean_x, mean_y, mean_z, mean_x, mean_y, mean_z])

    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()

    rng = np.random.default_rng()

    rng.shuffle(points)

    centered_points = points - np.array([mean_x, mean_y, mean_z])

    x_chunk_count = max(round(abs(extent[0] - extent[3]) / chunk_size), 1)
    x_chunks = np.linspace(extent[0], extent[3], x_chunk_count)

    y_chunk_count = max(round(abs(extent[1] - extent[4]) / chunk_size), 1)
    y_chunks = np.linspace(extent[1], extent[4], y_chunk_count)

    chunk_lists = [[[] for _ in range(x_chunk_count)] for _ in range(y_chunk_count)]

    for index, centered_point in enumerate(centered_points):
        x_index = -1
        y_index = -1

        for x in range(0, x_chunk_count - 1, 1):
            if x_chunks[x] <= centered_point[0] and centered_point[0] < x_chunks[x + 1]:
                x_index = x

                for y in range(0, y_chunk_count - 1, 1):
                    if y_chunks[y] <= centered_point[1] and centered_point[1] < y_chunks[y + 1]:
                        y_index = y

                        break

                break

        chunk_lists[y_index][x_index].append(centered_point)

    return chunk_lists


def plane_cleaning(points_with_ids, params):
    ids = points_with_ids[:, -1]

    points = points_with_ids[:, :3]

    tree = cKDTree(points)

    max_distance = params["cleaning distance"]
    minimum_points_needed = params["cleaning neighbors"]

    neighbors_distances, neighbors_indexes = tree.query(points, k=minimum_points_needed + 1, distance_upper_bound=max_distance, workers=5)

    neighbors_lists = list(zip(neighbors_distances, neighbors_indexes))

    for neighbor_values in neighbors_lists:
        distances = neighbor_values[0]
        indexes = neighbor_values[1]

        if distances[-1] == np.inf:
            cutoff = np.where(distances == np.inf)[0][0]

            distances = distances[:cutoff]
            indexes = indexes[:cutoff]

        if len(distances) <= 1:
            continue

        current_point_index = indexes[0]
        current_point_plane_id = ids[current_point_index]

        selected_neighbors_ids = [ids[x] for x in indexes[1:]]

        occurence_counter = Counter(selected_neighbors_ids)

        sorted_occurences = occurence_counter.most_common()

        if sorted_occurences[0][0] == current_point_plane_id:
            continue

        if sorted_occurences[0][0] != current_point_plane_id:
            try:
                current_plane_id_occurences = next(x for x in sorted_occurences if x[0] == current_point_plane_id)

            except:
                points_with_ids[current_point_index][3] = sorted_occurences[0][0]
                continue

            difference_factor = sorted_occurences[0][1] / current_plane_id_occurences[1]

            if difference_factor >= 2:
                points_with_ids[current_point_index][3] = sorted_occurences[0][0]

    return points_with_ids


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
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    tree = cKDTree(points)

    distances, indexes = tree.query([[0, 0, 0.1], [1.1, 0, 0]], k=10, distance_upper_bound=10)

    print(distances)
    print(indexes)

    print(np.where(distances[0] == 10000))

    # point_count = 150000

    # points = np.random.random_sample((point_count, 3))

    # ids = np.arange(point_count)

    # points_with_ids = np.c_[points, ids]

    # tree = cKDTree(points)

    # t1 = time.time()

    # result = tree.query_ball_point(points, 0.2, workers=5)

    # t2 = time.time()

    # print(t2 - t1)

    # print(result[5][:40])
