import numpy as np
import torch
import open3d as o3d
from sfm3 import load_json

from visualization import visualization
from readc2w import load_cameras_from_json

def read_ply_points(ply_file):
    # 读取 ply 文件
    pcd = o3d.io.read_point_cloud(ply_file)

    # 获取点云的 3D 坐标
    points = np.asarray(pcd.points)

    return points


def generate_cameras_around_sphere_for_blender(center, radius, num_cameras):
    cameras = []
    for i in range(num_cameras):
        # 在球面上均匀分布相机位置
        phi = np.random.uniform(0, np.pi * 2)
        phi = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, np.pi)
        theta = np.arccos(1 - 2 * np.random.uniform(0, 1))
        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(theta)

        camera_pos = np.array([x, y, z])

        # 计算相机朝向球心的向量
        look_dir = camera_pos - center
        look_dir = look_dir / np.linalg.norm(look_dir)

        # 修正：直接处理相机位于极点情况
        if np.linalg.norm(look_dir - np.array([0, 0, 1])) < 1e-6 or np.linalg.norm(
                look_dir - np.array([0, 0, -1])) < 1e-6:
            world_up = np.array([1, 0, 0])
        else:
            world_up = np.array([0, 0, 1])

        right = np.cross(world_up, look_dir)
        if np.linalg.norm(right) < 1e-6:  # 如果right向量太小，选择一个默认值
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)

        up = np.cross(look_dir, right)
        up = up / np.linalg.norm(up)

        # 重构c2w矩阵
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = look_dir  # 在Blender中，Z轴默认指向上方
        c2w[:3, 3] = camera_pos

        cameras.append(c2w)

    return cameras


c2ws = generate_cameras_around_sphere_for_blender([0, 0, 0], 5, 120)
c2ws = torch.tensor(c2ws)
points_3d = read_ply_points("chair.ply")
c2ws = load_cameras_from_json('cameras.json')
c2ws = np.array(c2ws)
c2ws[:, :3, 1:3] *= -1
visualization(c2ws, points_3d, grad_color= False)