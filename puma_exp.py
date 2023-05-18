#!/usr/bin/env python3

import glob
import os
from collections import deque
from pathlib import Path
import numpy as np
import open3d as o3d
import pykitti

from poisson import create_mesh_from_map
from utils import (
    get_progress_bar,
    load_config_from_yaml,
    # load_kitti_gt_poses,
    print_progress,
    preprocess,
)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", action="store", dest="config", default="./data/kk.ply", help="config file")
parser.add_argument("-d", "--dataset", action="store", dest="dataset", default="./data/kitti", type=str, help="dataset")
parser.add_argument("-s", "--start", action="store", dest="start", default=0, type=int, help="start")
parser.add_argument("-e", "--end", action="store", dest="end", default=-1, type=int, help="end")
parser.add_argument("-seq", "--sequence", action="store", dest="sequence", default="00", type=str, help="sequence")

FLAGS = parser.parse_args()

def main(config, dataset, start, end, sequence):
    """Similar to the slam/puma_pipeline.py but uses GT poses intead of
    estimate the ego-motion of the vehivle. Build an incremental map using
    the same technique used in the original puma pipeline."""
    config = load_config_from_yaml(config)
    dataset = os.path.join(dataset, "")
    os.makedirs(config.out_dir, exist_ok=True)

    map_name = Path(dataset).parent.name
    map_name += "_" + sequence
    map_name += "_depth_" + str(config.depth)
    map_name += "_cropped" if config.min_density else ""
    # gt_poses = load_kitti_gt_poses(dataset, sequence)
    data = pykitti.odometry(dataset, sequence)
    
    gt_poses = data.poses
    map_name += "_gt"

    poses_file = map_name + ".txt"
    poses_file = os.path.join(config.out_dir, poses_file)
    print("Results will be saved to", poses_file)

    scans = os.path.join(dataset, "sequences", sequence, "velodyne", "")
    scan_names = sorted(glob.glob(scans + "*.ply"))

    # Create data containers to store the map
    mesh = o3d.geometry.TriangleMesh()

    # Create a circular buffer, the same way we do in the C++ implementation
    local_map = deque(maxlen=config.acc_frame_count)
    global_mesh = o3d.geometry.TriangleMesh()

    poses = [np.eye(4, 4, dtype=np.float64)]

    # Start the mapping pipeline
    scan_count = 0
    map_count = 0
    step = config.step
    pbar = get_progress_bar(start, end, step=step)
    for idx in pbar:
        str_size = print_progress(pbar, idx, end)
        scan = preprocess(o3d.io.read_point_cloud(scan_names[idx]), config)
        poses.append(gt_poses[idx])
        scan.transform(poses[-1])
        local_map.append(scan)

        scan_count += 1
        if scan_count >= config.acc_frame_count or idx > end - step - 1:            
            scan_count = 0
            msg = "[scan #{}] Running PSR over local_map".format(idx)
            pbar.set_description(msg.rjust(str_size))
            mesh, _ = create_mesh_from_map(
                local_map, config.depth, config.n_threads, config.min_density
            )

        map_count += 1
        if map_count >= config.acc_map_count or idx > end - step - 1:
            map_count = 0
            global_mesh += mesh
            global_mesh = global_mesh.remove_duplicated_triangles()
            global_mesh = global_mesh.remove_duplicated_vertices()

    # Save map to file
    mesh_map_file = os.path.join(config.out_dir, map_name + ".ply")
    print("Saving Map to", mesh_map_file)
    o3d.io.write_triangle_mesh(mesh_map_file, global_mesh)

if __name__ == "__main__":
    config = FLAGS.config
    dataset = FLAGS.dataset
    start = FLAGS.start
    end = FLAGS.end
    # n_scans = FLAGS.n_scans
    sequence = FLAGS.sequence
    main(config, dataset, start, end,  sequence)
