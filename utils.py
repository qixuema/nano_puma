from easydict import EasyDict
import yaml
from tqdm import trange
from copy import deepcopy
import open3d as o3d
import numpy as np


def load_config_from_yaml(path):
    """Returns an EasyDict from the given path to a config.yml file.

    The path to the config file can be specified with an absolute
    path(e.g. ../../path_to_config/config.yml) or using a relative path
    to the current git directory(e.g config/config.yml). This function
    will try to load first from the absolute path and fallback to the
    git repo.
    """
    try:
        config_file = open(path)
    except FileNotFoundError:
        raise FileNotFoundError("{} file doesn't exist".format(path))

    # Returns any of the two possible config_file
    return EasyDict(yaml.safe_load(config_file))


def print_progress(pbar, idx, n_scans):
    msg = "[scan #{0}] Integrating scan #{0} of {1}".format(idx, n_scans)
    pbar.set_description(msg)
    return len(msg)

def get_progress_bar(first_scan, last_scan, step=1):
    return trange(
        first_scan,
        last_scan,
        step,
        unit=" scans",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    
def preprocess_cloud(
    pcd,
    voxel_size=0.25,
    max_nn=20,
    normals=None,
    downsample=False,
    crop=False,
    W = 1024,
    H = 64
):
    if downsample:
        cloud = pcd.voxel_down_sample(voxel_size) # 通过 voxel size 来控制点云降采样
    else:
        cloud = deepcopy(pcd)
    
    if crop: # 如果使用半径来裁剪点云
        # get depth of all points
        max_range = 30
        input_points = np.asarray(cloud.points)
        depth = np.linalg.norm(input_points, axis=1)
        pc_idx = (depth > 0) & (depth < max_range)
        current_points = input_points[pc_idx]
        cloud.points = o3d.utility.Vector3dVector(current_points)
        

    params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
    cloud.estimate_normals(params)
    cloud.orient_normals_towards_camera_location()
    
    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config.voxel_size,
        config.max_nn,
        config.normals,
        config.downsample,
        config.crop,
    )
