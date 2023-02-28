#!/usr/bin/env python3
import pathlib
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pcd_processing import (get_largest_triangle_cluster,
                            point_normal_from_plane_eq, read_pcd, read_pcd_t,
                            save_log, segment_clusters, segment_clusters_t,
                            segment_plane, segment_plane_t,
                            statistical_outlier_removal)

C_MAP = plt.get_cmap("tab20")
SHARD_QUATERNION = [1.0, 0.0, 0.0, 0.0]


def segment_shards(input_path: str,
                   path_output: str,
                   output: bool = False,
                   num_shards: int = 9,
                   vis: bool = False,
                   ground_plane_threshold: float = 0.008,
                   cluster_eps: float = 0.02,
                   cluster_min_points: int = 200,
                   cluster_plane_threshold: float = 0.002
                   ):

    # read point cloud
    # ----------------
    pcd = read_pcd(input_path)
    if not pcd:
        return
    print('pcd read successfully')

    if vis:
        o3d.visualization.draw_geometries([pcd])

    # segment ground plane
    # ----------------
    plane, pcd, pq = segment_plane(pcd, ground_plane_threshold)
    if output:
        log = {'shards': {'ground_plane': {}}}
        log['shards']['ground_plane'] = pq
    print('ground plane segmented successfully')

    if vis:
        plane.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, plane])

    # cluster
    # ----------------
    clusters, l = segment_clusters(
        pcd, eps=cluster_eps, min_points=cluster_min_points)
    print(
        f'clusters segmented successfully, found {l["num_clusters"]} clusters with lengths {l["cluster_len"]}')

    clusters = clusters[:num_shards]

    if output:
        log['shards'].update({'num_shards': num_shards})

    if vis:
        clusters_colored = []
        for i, cluster in enumerate(clusters):
            clusters_colored.append(deepcopy(cluster).paint_uniform_color(
                C_MAP(i / (l['num_clusters']))[:3]))
        o3d.visualization.draw_geometries(clusters_colored)

    # plane segmentation
    # ----------------
    cl_planes = []
    outliers = []
    for i, cluster in enumerate(clusters):
        cl_plane, outlier, l = segment_plane(cluster, cluster_plane_threshold)
        cl_planes.append(cl_plane)
        outliers.append(outlier.paint_uniform_color([1, 0, 0]))
        if output:
            log['shards'].update({f'shard_{i}': {}})
            log['shards'][f'shard_{i}'].update({'plane': l})

    if vis:
        o3d.visualization.draw_geometries(cl_planes + outliers)

    # planes center
    # ----------------
    centers = []
    for i, plane in enumerate(cl_planes):
        center = plane.get_center()
        centers.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=center))
        if output:
            log['shards'][f'shard_{i}'].update({'pick': {}})
            log['shards'][f'shard_{i}'].update({'place': None})
            log['shards'][f'shard_{i}']['pick'].update(
                {'position': {'x': str(center[0]), 'y': str(center[1]), 'z': str(center[2])}})
            log['shards'][f'shard_{i}']['pick'].update({'quaternion': {'x': str(SHARD_QUATERNION[0]), 'y': str(
                SHARD_QUATERNION[1]), 'z': str(SHARD_QUATERNION[2]), 'w': str(SHARD_QUATERNION[3])}})

    if vis:
        o3d.visualization.draw_geometries(centers + clusters)

    # outlier removal
    # ----------------
    outliers = []
    for idx, cluster in enumerate(clusters):
        cluster, outlier = statistical_outlier_removal(cluster)
        cluster = cluster.remove_duplicated_points()
        cluster = cluster.remove_non_finite_points()

        outliers.append(outlier.paint_uniform_color([1, 0, 0]))

    if vis:
        o3d.visualization.draw_geometries(clusters + outliers)

    # build mesh
    # ----------------
    meshes = []
    point, normal = point_normal_from_plane_eq(pq)
    print('\nbuilding meshes')
    for idx, cluster in enumerate(clusters):
        print(f'building mesh for shard {idx}')

        # # poisson mesh
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     cluster, depth=9)
        # densities = np.asarray(densities)
        # vertices_to_remove = densities < np.quantile(densities, 0.01)
        # mesh.remove_vertices_by_mask(vertices_to_remove)
        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # mesh = mesh.clip_plane(point, normal)
        # mesh = mesh.to_legacy()

        # ball pivot mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cluster,
                                                                               o3d.utility.DoubleVector(
                                                                                   [0.002]))

        # # mesh cleaning
        mesh = get_largest_triangle_cluster(mesh)
        mesh = mesh.remove_degenerate_triangles()
        # mesh.compute_vertex_normals()
        # mesh.compute_triangle_normals()
        # mesh.orient_triangles()
        # mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

        meshes.append(mesh)

    # if vis:
    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)

    # # write output to folder
    # # ----------------
    if output:

        directory_name = f'/segmented_shards_{datetime.now().strftime("%d_%m_%H_%M_%S")}'
        path_output = path_output + directory_name

        p_output = pathlib.Path(path_output)
        if not p_output.exists():
            p_output.mkdir(parents=False, exist_ok=False)

        for idx, mesh in enumerate(meshes):
            mesh_name = f'shard_{idx}.ply'
            o3d.io.write_triangle_mesh(
                f'{str(p_output)}/{mesh_name}', mesh)

            log['shards'][f'shard_{idx}'].update(
                {'mesh_path': mesh_name})

        save_log(log, f'{str(p_output)}/log.yaml')


def segment_shards_cuda(input_path: str,
                        path_output: str,
                        output: bool = False,
                        num_shards: int = 9,
                        vis: bool = False,
                        write_meshes: bool = False,
                        logging: bool = False,
                        ground_plane_threshold: float = 0.008,
                        cluster_eps: float = 0.02,
                        cluster_min_points: int = 200,
                        cluster_plane_threshold: float = 0.002
                        ):

    pcd = read_pcd_t(input_path)
    if not pcd:
        return

    # set device
    pcd = pcd.cuda()

    if vis:
        o3d.visualization.draw_geometries([pcd.to_legacy()])

    # ground plane segmentation
    # ----------------
    inliers, outliers, pq = segment_plane_t(pcd, ground_plane_threshold)

    if output:
        log = {'shards': {'ground_plane': {}}}
        log['shards']['ground_plane'] = pq

    if vis:
        o3d.visualization.draw_geometries(
            [inliers.paint_uniform_color([1.0, 0.0, 0.0]).to_legacy(), outliers.to_legacy()])

    # cluster segmentation
    # ----------------

    # tensor segmentation is to slow
    # clusters, l = segment_clusters_t(outliers, cluster_eps, cluster_min_points)

    clusters, l = segment_clusters(
        outliers.to_legacy(), cluster_eps, cluster_min_points)
    clusters = clusters[:num_shards]
    clusters = [o3d.t.geometry.PointCloud.from_legacy(
        cluster) for cluster in clusters]

    if output:
        log['shards'].update({'num_shards': num_shards})

    if vis:
        clusters_colored = []
        for i, cluster in enumerate(clusters):
            clusters_colored.append(cluster.clone().paint_uniform_color(
                C_MAP(i / (l['num_clusters']))[:3]).to_legacy())
        o3d.visualization.draw_geometries(clusters_colored)

    # compute boundary
    # ----------------
    boundaries = []
    for idx, cluster in enumerate(clusters):
        cluster.estimate_normals(radius=0.1, max_nn=30)
        boundary, mask = cluster.compute_boundary_points(max_nn=30, radius=0.1)
        # boundary, outlier, pq = segment_plane_t(boundary, 0.002)
        boundaries.append(boundary)
        

        
        cluster_boundary = cluster.select_by_mask(mask).paint_uniform_color([np.float32(1.0), np.float32(0.0), np.float32(0.0)])
        cluster = cluster.select_by_mask(mask, invert=True)
        clusters[idx] = cluster.append(cluster_boundary)
        


        # o3d.visualization.draw_geometries([cluster.to_legacy()])

    if vis:
        boundaries = [boundary.to_legacy().paint_uniform_color([1, 0, 0])
                      for boundary in boundaries]
        o3d.visualization.draw_geometries(boundaries)
                

    # plane segmentation
    # ----------------
    cl_planes = []
    outliers = []
    for i, cluster in enumerate(clusters):
        cl_plane, outlier, l = segment_plane_t(cluster, 0.002)
        cl_planes.append(cl_plane)
        outliers.append(outlier)
        if output:
            log['shards'].update({f'shard_{i}': {}})
            log['shards'][f'shard_{i}'].update({'plane': l})

    if vis:
        outliers = [outlier.to_legacy().paint_uniform_color([1, 0, 0])
                    for outlier in outliers]
        cl_planes = [cl_plane.to_legacy()
                     for cl_plane in cl_planes]
        o3d.visualization.draw_geometries(cl_planes + outliers)

    # planes center
    # ----------------
    centers = []
    for i, plane in enumerate(cl_planes):
        center = plane.get_center()
        centers.append(o3d.t.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=center))
        if output:
            log['shards'][f'shard_{i}'].update({'pick': {}})
            log['shards'][f'shard_{i}'].update({'place': None})
            log['shards'][f'shard_{i}']['pick'].update(
                {'position': {'x': str(center[0]), 'y': str(center[1]), 'z': str(center[2])}})
            log['shards'][f'shard_{i}']['pick'].update({'quaternion': {'x': str(SHARD_QUATERNION[0]), 'y': str(
                SHARD_QUATERNION[1]), 'z': str(SHARD_QUATERNION[2]), 'w': str(SHARD_QUATERNION[3])}})

    if vis:
        centers = [center.to_legacy() for center in centers]
        o3d.visualization.draw_geometries(centers + clusters_colored)

    # outlier removal
    # ----------------
    # outliers = []
    # for cluster in clusters:
    #     cluster = cluster.remove_radius_outliers(
    #         nb_points=16, search_radius=0.02)
    #     cluster = cluster[0].remove_duplicated_points()
    #     cluster = cluster[0].remove_non_finite_points()

    # build mesh
    # ----------------

    point, normal = point_normal_from_plane_eq(pq)
    meshes = []
    print('\nbuilding meshes')
    for idx, cluster in enumerate(clusters):
        print(f'building mesh shard {idx}')

        # poisson mesh
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     cluster.to_legacy(), depth=9)
        # densities = np.asarray(densities)
        # vertices_to_remove = densities < np.quantile(densities, 0.01)
        # mesh.remove_vertices_by_mask(vertices_to_remove)
        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # mesh = mesh.cuda()
        # mesh = mesh.clip_plane(point, normal)
        # mesh = mesh.to_legacy()

        # ball pivot mesh
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.1, max_nn=30))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cluster.to_legacy(),
                                                                               o3d.utility.DoubleVector(
                                                                                   [0.01, 0.02]))

        # # mesh cleaning
        # mesh = get_largest_triangle_cluster(mesh)
        # mesh = mesh.remove_degenerate_triangles()
        # mesh.compute_vertex_normals()
        # mesh.compute_triangle_normals()
        # mesh.orient_triangles()
        # mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

        meshes.append(mesh)

    o3d.visualization.draw_geometries(meshes)

    # # write output to folder
    # # ----------------

    if output:

        directory_name = f'/segmented_shards_{datetime.now().strftime("%d_%m_%H_%M_%S")}'
        path_output = path_output + directory_name

        p_output = pathlib.Path(path_output)
        if not p_output.exists():
            p_output.mkdir(parents=False, exist_ok=False)

        for idx, mesh in enumerate(meshes):
            mesh_name = f'shard_{idx}.ply'
            o3d.io.write_triangle_mesh(
                f'{str(p_output)}/{mesh_name}', mesh)

            log['shards'][f'shard_{idx}'].update(
                {'mesh_path': mesh_name})

        save_log(log, f'{str(p_output)}/log.yaml')


if __name__ == '__main__':

    # segment_shards(input_path='/home/v/02_24_15_20.ply',
    #                path_output='/home/v/',
    #                output=True,
    #                num_shards=11,
    #                vis=True,
    #                ground_plane_threshold=0.002,
    #                cluster_eps=0.01,
    #                cluster_min_points=200,
    #                cluster_plane_threshold=0.002
    #                )

    # segment_shards(input_path='/home/v/pcd/linear.ply',
    #                path_output='/home/v/',
    #                output=True,
    #                num_shards=10,
    #                vis=True,
    #                ground_plane_threshold=0.002,
    #                cluster_eps=0.01,
    #                cluster_min_points=200,
    #                cluster_plane_threshold=0.002
    #    )

    segment_shards_cuda(input_path='/home/v/pcd/shards.ply',
                     path_output='/home/v/',
                     output=True,
                     num_shards=9,
                     vis=True,
                     ground_plane_threshold=0.008,
                     cluster_eps=0.02,
                     cluster_min_points=300,
                     cluster_plane_threshold=0.002)

    # segment_shards_cuda(input_path='/home/v/pcd/linear.ply',
    #                     path_output='/home/v/',
    #                     output=False,
    #                     num_shards=10,
    #                     vis=True,
    #                     ground_plane_threshold=0.002,
    #                     cluster_eps=0.01,
    #                     cluster_min_points=300,
    #                     cluster_plane_threshold=0.002)

    # segment_shards_cuda(input_path='/home/v/02_23_18_43.ply',
    #                     path_output='/home/v/',
    #                     output=True,
    #                     num_shards=11,
    #                     vis=True,
    #                     ground_plane_threshold=0.002,
    #                     cluster_eps=0.01,
    #                     cluster_min_points=200,
    #                     cluster_plane_threshold=0.002)
