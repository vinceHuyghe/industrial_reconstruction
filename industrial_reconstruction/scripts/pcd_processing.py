import pathlib
from typing import List, Tuple

import numpy as np
import open3d as o3d
import yaml

# -----------------
# utility functions
# -----------------


def read_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd_path = pathlib.Path(path)
    if not pcd_path.exists():
        raise FileNotFoundError(f"File {pcd_path} does not exist")
    return o3d.io.read_point_cloud(str(pcd_path), remove_nan_points=True,
                                   remove_infinite_points=True)


def read_pcd_t(path: str) -> o3d.t.geometry.PointCloud:
    pcd_path = pathlib.Path(path)
    if not pcd_path.exists():
        raise FileNotFoundError(f"File {pcd_path} does not exist")
    return o3d.t.io.read_point_cloud(str(pcd_path))


def read_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh_path = pathlib.Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"File {mesh_path} does not exist")
    return o3d.io.read_triangle_mesh(str(mesh_path))


def statistical_outlier_removal(pcd: o3d.geometry.PointCloud,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0,
                                ) -> List[o3d.geometry.PointCloud]:

    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    return [inlier_cloud, outlier_cloud]


def radius_outlier_removal(pcd: o3d.geometry.PointCloud,
                           nb_points: int = 16,
                           radius=0.05) -> List[o3d.geometry.PointCloud]:

    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    return [inlier_cloud, outlier_cloud]


def segment_plane(pcd: o3d.geometry.PointCloud,
                  distance_threshold: float = 0.01
                  ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, dict]:
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    p_eq = {'a': str(a), 'b': str(b), 'c': str(c), 'd': str(d)}
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud, p_eq


def segment_planes(pcd: o3d.geometry.PointCloud,
                   iterations: int = 3,
                   distance_threshold: float = 0.05):
    planes = []
    log = {}
    outliers = pcd

    for i in range(iterations):

        plane, outliers, pq = segment_plane(outliers, distance_threshold)
        planes.append(plane)
        log['plane_' + str(i)] = pq

    return planes, outliers, log


def segment_plane_t(pcd: o3d.t.geometry.PointCloud,
                    distance_threshold: float = 0.01
                    ) -> Tuple[o3d.t.geometry.PointCloud, o3d.t.geometry.PointCloud,
                               dict]:
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model.cpu().numpy().tolist()
    p_eq = {'a': str(a), 'b': str(b), 'c': str(c), 'd': str(d)}
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud, p_eq


def segment_clusters(pcd: o3d.geometry.PointCloud,
                     eps: float = 0.02,
                     min_points: int = 100) -> Tuple[List[o3d.geometry.PointCloud],
                                                     dict]:

    clusters = []
    cls = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=True))
    num_clusters = len(np.unique(cls))

    for i in range(num_clusters):
        cluster = pcd.select_by_index(np.where(cls == i)[0])
        clusters.append(cluster)

    clusters.sort(key=lambda x: len(x.points), reverse=True)

    log = {'num_clusters': num_clusters, 'cluster_len': [
        len(c.points) for c in clusters]}

    return clusters, log


def segment_clusters_t(pcd: o3d.t.geometry.PointCloud,
                       eps: float = 0.02,
                       min_points: int = 400
                       ) -> Tuple[List[o3d.t.geometry.PointCloud], dict]:
    clusters = []
    cls = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=True))
    num_clusters = len(np.unique(cls))
    cluster_sizes = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_sizes[i] = np.where(cls == i)[0].shape[0]
    sorted_clusters = np.argsort(cluster_sizes)[::-1]

    log = {'num_clusters': num_clusters,
           'cluster_len': cluster_sizes[sorted_clusters].tolist()}
    print(log)

    for i in range(num_clusters):
        cluster = pcd.select_by_index(
            o3d.core.Tensor(np.where(cls == sorted_clusters[i])[0]).cuda())
        clusters.append(cluster)

    return clusters, log


def point_normal_from_plane_eq(plane_eq: dict) -> Tuple[np.ndarray, np.ndarray]:
    a = float(plane_eq['a'])
    b = float(plane_eq['b'])
    c = float(plane_eq['c'])
    d = float(plane_eq['d'])

    point = np.array([0, 0, -d / c])
    normal = np.array([a, b, c])

    return point, normal


def get_largest_triangle_cluster(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh = mesh.remove_unreferenced_vertices()

    return mesh


def save_log(report: dict, path: str) -> bool:
    with open(path, 'w') as file:
        documents = yaml.dump(report, file)
    return True
