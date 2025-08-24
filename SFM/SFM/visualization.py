import sys
import os
import argparse
from scipy.linalg import svd

sys.path.append(os.path.join(sys.path[0], '../..'))
import json
import open3d as o3d
from vis_cam_traj import draw_camera_frustum_geometry

import torch
import numpy as np

# import quaternion

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=True, type=eval, choices=[True, False])

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--resize_ratio', type=int, default=8, help='lower the image resolution with this ratio')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_load_sorted', type=bool, default=False)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def generate_random_c2w_matrices(base_c2w, vertical_dist_coeff, parallel_dist_coeff, num_matrices):
    def move_camera(base_c2w, vertical_dist_coeff, parallel_dist_coeff):
        # 获取基准的旋转矩阵
        rotation_matrix = base_c2w[:3, :3]

        # 相机的前向向量（通常是Z轴负方向，这里根据惯例进行调整）
        forward_vector = -rotation_matrix[:, 2]
        # 相机的右向量和上向量
        right_vector = rotation_matrix[:, 0]
        up_vector = rotation_matrix[:, 1]

        # 计算垂直移动（沿相机朝向）
        vertical_movement = forward_vector * vertical_dist_coeff * np.random.uniform(-1, 1)

        # 计算平行移动（在相机的右向和上向平面内）
        parallel_movement = right_vector * parallel_dist_coeff * np.random.uniform(-1,
                                                                                   1) + up_vector * parallel_dist_coeff * np.random.uniform(
            -1, 1)

        # 更新位置
        new_position = base_c2w[:3, 3] + vertical_movement + parallel_movement

        # 构造新的c2w矩阵
        new_c2w = np.zeros_like(base_c2w)
        new_c2w[:3, :3] = rotation_matrix
        new_c2w[:3, 3] = new_position
        new_c2w[3, 3] = 1

        return new_c2w

    # 生成随机c2w矩阵列表
    random_c2w_matrices = [move_camera(base_c2w, vertical_dist_coeff, parallel_dist_coeff) for _ in range(num_matrices)]

    return random_c2w_matrices


def generate_nearby_quaternions(base_quaternion, angle_range=np.radians(10), num_samples=20):
    """
    生成与给定四元数接近的四元数列表。

    :param base_quaternion: 基准四元数
    :param angle_range: 每个轴上的旋转角度范围（以弧度为单位）
    :param num_samples: 要生成的近邻四元数的数量
    :return: 与基准四元数接近的四元数列表
    """
    nearby_quaternions = []
    for _ in range(num_samples):
        # 随机选择一个轴
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        # 随机选择一个角度
        angle = np.random.uniform(-angle_range, angle_range)
        # 生成旋转四元数
        rotation_quaternion = quaternion.from_rotation_vector(axis * angle)
        # 应用旋转
        new_quaternion = rotation_quaternion * base_quaternion
        nearby_quaternions.append(new_quaternion)
    return nearby_quaternions


def generate_complete_c2w_matrices(base_c2w, vertical_dist_coeff, parallel_dist_coeff, angle_range, num_matrices):
    def move_camera(base_c2w, vertical_dist_coeff, parallel_dist_coeff):
        rotation_matrix = base_c2w[:3, :3]
        forward_vector = -rotation_matrix[:, 2]
        right_vector = rotation_matrix[:, 0]
        up_vector = rotation_matrix[:, 1]

        vertical_movement = forward_vector * vertical_dist_coeff * np.random.uniform(-1, 1)
        parallel_movement = right_vector * parallel_dist_coeff * np.random.uniform(-1,
                                                                                   1) + up_vector * parallel_dist_coeff * np.random.uniform(
            -1, 1)

        new_position = base_c2w[:3, 3] + vertical_movement + parallel_movement

        new_c2w = np.zeros_like(base_c2w)
        new_c2w[:3, :3] = rotation_matrix
        new_c2w[:3, 3] = new_position
        new_c2w[3, 3] = 1

        return new_c2w

    def generate_nearby_quaternions(base_quaternion, angle_range, num_samples):
        nearby_quaternions = []
        for _ in range(num_samples):
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(-angle_range, angle_range)
            rotation_quaternion = quaternion.from_rotation_vector(axis * angle)
            new_quaternion = rotation_quaternion * base_quaternion
            nearby_quaternions.append(new_quaternion)
        return nearby_quaternions

    # Generate c2w matrices with different positions
    position_varied_c2w = [move_camera(base_c2w, vertical_dist_coeff, parallel_dist_coeff) for _ in range(num_matrices)]

    complete_c2w_matrices = []
    for c2w in position_varied_c2w:
        base_rotation_matrix = c2w[:3, :3]
        base_quaternion = quaternion.from_rotation_matrix(base_rotation_matrix)
        # Generate quaternions for nearby rotations
        nearby_quats = generate_nearby_quaternions(base_quaternion, angle_range, num_matrices)
        for quat in nearby_quats:
            rotation_matrix = quaternion.as_rotation_matrix(quat)
            # Construct new c2w matrix with updated rotation
            new_c2w = np.zeros_like(c2w)
            new_c2w[:3, :3] = rotation_matrix
            new_c2w[:3, 3] = c2w[:3, 3]  # Keep the position unchanged from position_varied_c2w
            new_c2w[3, 3] = 1
            complete_c2w_matrices.append(new_c2w)

    return complete_c2w_matrices


def interpolate_camera_poses(frame_ids, Rs, Ts, total_frames):
    # 将帧编号转换为整数并获取每个编号对应的索引
    indices = [int(frame_id.replace('frame', '')) for frame_id in frame_ids]
    indices = np.array(indices) / total_frames  # 归一化索引

    # 创建插值所需的新索引（对于所有帧）
    new_indices = np.linspace(0.00847457627118644, 1, total_frames)

    # 对于旋转的插值，先将旋转矩阵转换为四元数
    quaternions = [R.from_matrix(r).as_quat() for r in Rs]
    quaternions = np.array(quaternions)

    # 使用Slerp进行旋转的插值
    slerp = Slerp(indices, R.from_quat(quaternions))
    interpolated_Rs = slerp(new_indices)

    # 对平移向量T使用CubicSpline进行插值
    spline = CubicSpline(indices, Ts, axis=0)
    interpolated_Ts = spline(new_indices)

    # 将插值后的四元数旋转转换回旋转矩阵
    interpolated_Rs = [r.as_matrix() for r in interpolated_Rs]

    return interpolated_Rs, interpolated_Ts


def sort_frames_and_poses(frame_ids, c2w_):
    # 提取帧编号为整数
    frame_numbers = [int(frame_id.replace('frame', '')) for frame_id in frame_ids]

    # 使用帧编号进行排序，同时保留排序后的索引
    sorted_indices = sorted(range(len(frame_numbers)), key=lambda k: frame_numbers[k])

    # 根据排序后的索引重新排序frame_ids和c2w_
    sorted_frame_ids = [frame_ids[i] for i in sorted_indices]
    sorted_c2w_ = [c2w_[i] for i in sorted_indices]

    return sorted_frame_ids, sorted_c2w_


def generate_color_gradient(start_color, end_color, n):
    """
    生成一个从 start_color 到 end_color 的颜色渐变数组。

    :param start_color: 起始颜色，格式为 [R, G, B]
    :param end_color: 结束颜色，格式为 [R, G, B]
    :param n: 渐变颜色的数量
    :return: 颜色渐变数组，形状为 (n, 3)
    """
    # 将输入颜色转换为 np.array 并标准化到 [0, 1]
    start_color = np.array(start_color, dtype=np.float32) / 255
    end_color = np.array(end_color, dtype=np.float32) / 255

    # 生成颜色渐变
    gradient = np.linspace(start_color, end_color, n)

    return gradient


def create_json(camera_transforms, file_paths, camera_angle_x, output_file):
    """
    创建一个指定格式的JSON文件。

    :param camera_transforms: 相机的c2w转换矩阵列表。
    :param file_paths: 相应的文件路径列表。
    :param camera_angle_x: 相机的X轴视角。
    :param output_file: 输出JSON文件的路径。
    """
    # 确保输入列表长度匹配
    if len(camera_transforms) != len(file_paths):
        raise ValueError("camera_transforms 和 file_paths 的长度必须匹配。")

    # 构建frames数据
    frames_data = []
    for transform_matrix, file_path in zip(camera_transforms, file_paths):
        frame_data = {
            "file_path": 'train/' + file_path,
            "transform_matrix": transform_matrix,
        }
        frames_data.append(frame_data)

    # 构建最终的字典
    data = {
        "camera_angle_x": camera_angle_x,
        "frames": frames_data,
    }

    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def load_c2ws(json_file_path):
    """
    从 JSON 格式数据中读取相机到世界坐标的变换矩阵，并返回一个 n*4*4 的 ndarray。

    参数:
    json_data (str): JSON 格式的字符串，包含相机的变换矩阵数据。

    返回:
    ndarray: 形状为 (n, 4, 4) 的 numpy 数组，包含所有帧的 c2w 矩阵。
    """
    # 解析 JSON 数据
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)  # Directly load from file
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {json_file_path} is not a valid JSON file.")
        return None

    # 获取帧列表
    frames = data['frames']

    # 创建一个空的列表来存储变换矩阵
    c2ws = []

    # 遍历每一帧，提取变换矩阵
    for frame in frames:
        # 将当前帧的变换矩阵添加到列表中
        c2ws.append(frame['transform_matrix'])

    # 将列表转换为 numpy 数组
    c2ws_array = np.array(c2ws)

    return c2ws_array


def generate_cameras_around_sphere_for_blender(center, radius, num_cameras):
    cameras = []
    for i in range(num_cameras):
        # 在球面上均匀分布相机位置
        phi = np.random.uniform(0, np.pi * 2)
        theta = np.random.uniform(0, np.pi)
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





def generate_c2w_around(base_c2w, n, max_angle, max_roll):
    new_c2w_matrices = []

    # 获取相机到原点的距离
    distance = np.linalg.norm(base_c2w[:3, 3])
    base_direction = base_c2w[:3, 3] / distance  # 归一化的方向向量
    base_up = base_c2w[:3, 1]

    # 生成 n 个随机旋转矩阵
    random_axes = np.random.randn(n, 3)
    random_axes /= np.linalg.norm(random_axes, axis=1)[:, np.newaxis]
    random_angles = np.random.uniform(0, max_angle, n)
    rotations = R.from_rotvec((random_angles[:, np.newaxis] * random_axes))

    # 生成新的方向向量，并添加扰动
    new_directions = rotations.apply(base_direction)
    perturbation = np.random.randn(n, 3) / 100
    perturbed_directions = new_directions + perturbation
    perturbed_directions /= np.linalg.norm(perturbed_directions, axis=1)[:, np.newaxis]

    # 确定相机位置：保持在原点为球心，distance 为半径的球上
    new_positions = perturbed_directions * distance

    for new_position, new_direction in zip(new_positions, perturbed_directions):
        # 计算新的旋转矩阵，使相机指向扰动方向
        z_axis = new_direction / np.linalg.norm(new_direction)
        up = base_up
        if np.abs(np.dot(z_axis, up)) > 0.99:
            up = base_c2w[:3, 0]  # 防止与z轴平行，选择一个不同的up方向
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # 更新旋转矩阵
        R_camera = np.vstack([x_axis, y_axis, z_axis]).T

        # 添加翻滚旋转
        random_roll = np.random.uniform(-max_roll, max_roll)
        roll_rotation = R.from_euler('z', random_roll).as_matrix()
        R_camera = R_camera @ roll_rotation

        new_c2w = np.eye(4)
        new_c2w[:3, :3] = R_camera
        new_c2w[:3, 3] = new_position

        new_c2w_matrices.append(new_c2w)

    return new_c2w_matrices



def draw_plane(plane_point, plane_normal):
    # 创建一个网格来表示平面
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(x, y)
    z = (-plane_normal[0] * x - plane_normal[1] * y + np.dot(plane_normal, plane_point)) / plane_normal[2]

    # 创建顶点和三角形
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    triangles = []
    for i in range(9):
        for j in range(9):
            triangles.append([i * 10 + j, i * 10 + j + 1, (i + 1) * 10 + j])
            triangles.append([i * 10 + j + 1, (i + 1) * 10 + j + 1, (i + 1) * 10 + j])

    # 创建Mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 绘制Mesh
    return mesh


def filter_cameras_by_distance(c2w_list, c2w, n, m):
    """
    过滤出与给定c2w距离大于n的c2w矩阵列表。

    参数:
    c2w_list: 一个包含多个c2w矩阵的列表。
    c2w: 一个参考的c2w矩阵。
    n: 距离阈值。

    返回:
    过滤后的c2w矩阵列表。
    """
    w2cs = [np.linalg.inv(c2w) for c2w in c2w_list]
    w2c = np.linalg.inv(c2w)
    filtered_cameras = []

    # 提取参考c2w矩阵的位置部分
    ref_pos = w2c[:3, 3]

    for i, cam in enumerate(w2cs):
        # 提取当前c2w矩阵的位置部分
        cam_pos = cam[:3, 3]
        # 计算两个位置之间的欧式距离
        distance = np.linalg.norm(ref_pos - cam_pos)
        # 如果距离大于阈值，则添加到结果列表中
        if distance > n and distance < m:
            filtered_cameras.append(c2w_list[i])

    return filtered_cameras


def calculate_camera_distances_to_plane(c2w_matrices, plane_normal, plane_point):
    """
    Calculate distances of cameras from a plane defined by a normal vector and a point on the plane.

    Args:
    - c2w_matrices: A list of camera-to-world transformation matrices (4x4 numpy arrays).
    - plane_normal: The normal vector of the plane (should be a normalized 3-element numpy array).
    - plane_point: A point on the plane (3-element numpy array).

    Returns:
    - distances: A list of distances of each camera from the plane.
    """
    distances = []
    for c2w in c2w_matrices:
        camera_position = c2w[:3, 3]  # Extract the camera position from c2w matrix
        vector_to_plane = camera_position - plane_point  # Vector from camera to a point on the plane
        distance = abs(np.dot(plane_normal, vector_to_plane))  # Calculate the distance
        distances.append(distance)
    return distances


def determine_plane_from_cameras(c2w_matrices):
    # Extract camera directions from c2w matrices
    camera_directions = np.array(
        [-c2w[:3, 2] for c2w in c2w_matrices])  # Assuming the camera looks along the negative z-axis

    # Solve the linear system for the plane normal
    _, _, Vh = svd(camera_directions, full_matrices=False)
    plane_normal = Vh[-1, :]  # The solution is the last singular vector

    # Calculate the centroid of the camera positions as a point on the plane
    camera_positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])
    plane_point = np.mean(camera_positions, axis=0)

    return plane_normal, plane_point


def point_to_camera_line_distance(c2w_matrices, point):
    """
    计算三维点到一系列相机朝向直线的距离。

    参数:
    - c2w_matrices: 相机的camera-to-world矩阵列表。
    - point: 三维空间中的点，形状为(3,)的numpy数组。

    返回:
    - distances: 与c2w_matrices列表中的每个相机对应的点到直线的距离列表。
    """
    distances = []
    for c2w in c2w_matrices:
        # 相机位置
        camera_pos = c2w[:3, 3]
        # 相机朝向，假设相机的前向在相机坐标系中是Z轴负方向
        camera_forward = -c2w[:3, 2]  # 取Z轴负方向

        # 计算点P到相机朝向直线的距离
        distance = np.linalg.norm(np.cross(point - camera_pos, camera_forward)) / np.linalg.norm(camera_forward)
        distances.append(distance)

    return distances


def write_camera_pose_to_txt(c2w_matrices, file_name='stamped_traj_estimate.txt'):
    """
    将相机的世界到相机矩阵 (c2w) 写入一个文本文件，格式为：
    # timestamp tx ty tz qx qy qz qw

    参数:
    - c2w_matrices: 相机的世界到相机矩阵列表。
    - file_name: 输出文本文件的名称。
    """
    # 打开文件准备写入
    with open(file_name, 'w') as file:
        file.write("# timestamp tx ty tz qx qy qz qw\n")

        for i, mat in enumerate(c2w_matrices):
            # 生成时间戳示例（这里简单使用索引，实际应用中需要根据情况生成或提供）
            timestamp = f"1.403636580013555{i:03}e+09"

            # 提取平移向量
            t = mat[:3, 3]

            # 提取旋转矩阵并转换为四元数
            r = R.from_matrix(mat[:3, :3])
            q = r.as_quat()  # 四元数格式为 (x, y, z, w)

            # 写入文件
            file.write(f"{timestamp} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

    print(f"文件 '{file_name}' 已生成。")


def move_camera_along_direction(base_c2w, distance):
    """
    Move the camera along its viewing direction by a specified distance.

    Args:
    - base_c2w: The base camera-to-world transformation matrix (4x4 numpy array).
    - distance: The distance to move the camera along its viewing direction.
                Positive values move the camera forward, negative values move it backward.

    Returns:
    - new_c2w: The new camera-to-world transformation matrix after moving the camera.
    """
    # Extract the rotation matrix and position vector from the base c2w matrix
    rotation_matrix = base_c2w[:3, :3]
    position_vector = base_c2w[:3, 3]

    # The viewing direction is the negative Z axis of the camera's coordinate system
    viewing_direction = -rotation_matrix[:, 2]

    # Calculate the new position by moving along the viewing direction
    new_position = position_vector + distance * viewing_direction

    # Construct the new c2w matrix with the updated position
    new_c2w = np.zeros_like(base_c2w)
    new_c2w[:3, :3] = rotation_matrix  # Keep the original rotation
    new_c2w[:3, 3] = new_position  # Update the position
    new_c2w[3, 3] = 1  # Set the bottom-right element to 1

    return new_c2w


def main():
    # cam = np.genfromtxt('data_css/camera.csv', delimiter=',')
    # cam = cam.reshape(-1, 3, 4)
    # cam[:, :3, :3] = -cam[:, :3, :3]
    #
    # pose3 = np.load('data_css/pose3.npy').reshape(1, 3, 4)
    #
    # cam = np.concatenate([cam, pose3], 0)
    # print(cam)
    # cam = np.concatenate([cam, np.tile(np.array([[0, 0, 0, 1]]), (len(cam), 1, 1))], 1)
    # # cam = np.linalg.inv(cam)
    # points = np.genfromtxt('data_css/points.csv', delimiter=',')
    # visualization(cam, points)

    cam = np.load('c2ws_mm.npy')
    visualization(cam, None)
    # frame_ids = [f"frame{i:03}" for i in range(0, 140, 10)]
    # frame_ids = sorted(frame_ids)
    # create_json(cam.tolist(), frame_ids, 0.08040715440186917, 'css.json')


def visualization(camera, points_3d, grad_color=True):
    if camera.shape[1] == 3:
        camera = np.concatenate([camera, np.tile(np.array([[0, 0, 0, 1]]), (len(camera), 1, 1))], 1)

    c2ws_est = camera
    points_3d = points_3d

    '''Frustum properties'''
    ff = 500
    fs = 2

    random_color = False
    if random_color:
        pass
    else:
        frustum_color_r = np.array([[249, 65, 68]], dtype=np.float32) / 255
        frustum_color_g = np.array([[65, 249, 68]], dtype=np.float32) / 255
        #
        # frustum_color_g = np.tile(frustum_color_g, (len(c2ws_est)//2, 1))
        frustum_color = np.tile(frustum_color_r, (len(c2ws_est), 1))
        # frustum_color = np.concatenate([frustum_color_g, frustum_color_r], 0)
        # frustum_color_g = np.tile(frustum_color_g, (len(c2ws_est)//2, 1))
        # frustum_color = np.concatenate([frustum_color_r, frustum_color_g], 0)
        # frustum_color = np.concatenate([np.tile(frustum_color_r, (len(c2ws_est)-1, 1)), frustum_color_g],0)
        #
        if grad_color:
            frustum_color = generate_color_gradient([91, 155, 233], [250, 60, 68], len(c2ws_est))
        else:
            frustum_color = np.tile(frustum_color_r, (len(c2ws_est), 1))
        # frustum_color2 = generate_color_gradient([249, 65, 68], [249, 65, 68], len(c2ws_est)//2)
        # frustum_color = np.concatenate([frustum_color1, frustum_color2], 0)
        # frustum_color = np.tile(frustum_color_g, (len(c2ws_est), 1))
    '''Get frustums'''
    frustum_est_list = draw_camera_frustum_geometry(c2ws_est, 100, 100,
                                                    ff, ff,
                                                    fs, frustum_color)

    # geometry_to_draw = []
    # geometry_to_draw.append(frustum_est_list)
    # # geometry_to_draw.append(draw_plane(plane_point, plane_normal))
    # o3d.visualization.draw_geometries(geometry_to_draw)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 设置背景颜色为黑色
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # 添加要绘制的几何体
    colors = np.array([[0.45, 0.9, 1] for _ in range(len(points_3d))])
    if points_3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
    vis.add_geometry(frustum_est_list)
    # vis.add_geometry(coordinate_frame)
    # 运行Visualizer
    vis.run()

if __name__ == '__main__':
    with torch.no_grad():
        main()


