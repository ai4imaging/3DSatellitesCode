#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
import lpips
from scene.cameras import Camera
from scene.dataset_readers import c2w
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getRt, get_c2w

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.other_cams = []
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.lpips_loss = lpips.LPIPS(net="alex").cuda()

    def lpips_loss(self, img1, img2):
        return self.lpips_loss(img1, img2)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        frame_ids = [cam.image_name for cam in self.train_cameras[1.0]]

        R = [np.transpose(cam.R) for cam in self.train_cameras[1.0]]
        T = [cam.T for cam in self.train_cameras[1.0]]
        c2ws = c2w(R, T)
        sorted_frame_ids, c2ws = sort_frames_and_poses(frame_ids, c2ws)
        np.save(os.path.join(point_cloud_path, "c2ws.npy"), c2ws)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getAllCameras(self, cams, scale=1.0):
        cameras = cams
        cameras.sort(key=lambda x: x.image_name)
        ids = [cam.image_name for cam in cameras]
        R = [np.transpose(cam.R) for cam in cameras]
        T = [cam.T for cam in cameras]
        Rs, Ts = interpolate_camera_poses(ids, R, T, 138)
        cams = []
        ids = range(0, 138)
        for i in ids:
            image_name = "frame{:03d}".format(i)
            image_path = os.path.join('data/test2/images/', image_name + '.jpg')
            image = Image.open(image_path)
            image = PILtoTorch(image, (360, 360))
            cam = Camera(i, Rs[i].transpose(), Ts[i], 0.084628, 0.084628, image, None, image_name,
                         i)
            cams.append(cam)
        self.train_cameras[scale] = cams
        return cams

    def expandCameras(self, flag = False, scale=1.0):
        cameras = self.train_cameras[scale]
        cameras.sort(key=lambda x: x.image_name)
        ids = [cam.image_name for cam in cameras]
        R = [np.transpose(cam.R) for cam in cameras]
        T = [cam.T for cam in cameras]
        Rs, Ts = interpolate_camera_poses(ids, R, T, 138)
        cams = [cam for cam in self.train_cameras[scale]]
        ids = [60, 70]
        for i in ids:
            image_name = "frame{:03d}".format(i)
            image_path = os.path.join('data/test2/images/', image_name + '.jpg')
            image = Image.open(image_path)
            image = PILtoTorch(image, (360, 360))
            cam = Camera(i, Rs[i].transpose(), Ts[i], cameras[0].FoVx, cameras[0].FoVy, image, None, image_name, i)
            cams.append(cam)
        self.train_cameras[scale] = cams

    def setTrainCameras(self, cameras, scale=1.0):
        self.train_cameras[scale] = cameras

    def replaceCameras(self, before, after):
        self.train_cameras[1.0].remove(before)
        self.train_cameras[1.0].append(after)

    def init_fov(self, dis):

        info = [(cam.image_name, cam.FoVx) for cam in self.train_cameras[1.0]]
        info.sort(key=lambda x: x[0])
        print(info)
        print([info[i + 1][1] - info[i][1] for i in range(len(info) - 1)])
        fovs = [cam.FoVx for cam in self.train_cameras[1.0]]
        trimmed_numbers = sorted(fovs)[2:-2]
        fov = sum(trimmed_numbers) / len(trimmed_numbers)
        print('init fov', fov)
        for i, cam in enumerate(self.train_cameras[1.0]):
            cam.FoVx = fov
            cam.FoVy = fov
            c2w_ = get_c2w(cam.R, cam.T)
            c2w_ = move_camera_along_direction(c2w_, dis[i])
            R, T = getRt(c2w_)
            cam.R = R
            cam.T = T

    def init_other_cams(self):
        cams = np.load('data/cams.npy')
        print('Len cams', len(cams))
        cm = self.train_cameras[1.0][0]
        for i in range(len(cams)):
            R, T = getRt(cams[i])
            cam = Camera(0, R, T, cm.FoVx, cm.FoVx, cm.original_image,
                         None, cm.image_name, cm.uid)
            self.other_cams.append(cam)
        print('init other cams', len(self.other_cams))
    def get_other_cams(self):
        return self.other_cams

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def updateTrainCameras(self, cameras):
        self.train_cameras[1.0] = cameras

def sort_frames_and_poses(frame_ids, c2w_):
    # 提取帧编号为整数
    frame_numbers = [int(frame_id.replace('frame', '')) for frame_id in frame_ids]

    # 使用帧编号进行排序，同时保留排序后的索引
    sorted_indices = sorted(range(len(frame_numbers)), key=lambda k: frame_numbers[k])

    # 根据排序后的索引重新排序frame_ids和c2w_
    sorted_frame_ids = [frame_ids[i] for i in sorted_indices]
    sorted_c2w_ = [c2w_[i] for i in sorted_indices]

    return sorted_frame_ids, sorted_c2w_


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


def interpolate_camera_poses(frame_ids, Rs, Ts, total_frames):
    # 将帧编号转换为整数并获取每个编号对应的索引
    indices = [int(frame_id.replace('frame', '')) for frame_id in frame_ids]
    indices[0] = 0
    indices = np.array(indices) / max(indices)  # 归一化索引

    # 创建插值所需的新索引（对于所有帧）
    new_indices = np.linspace(0, 1, total_frames)

    # 对于旋转的插值，先将旋转矩阵转换为四元数
    quaternions = [R.from_matrix(r).as_quat() for r in Rs]
    quaternions = np.array(quaternions)

    # 使用Slerp进行旋转的插值
    slerp = Slerp(indices, R.from_quat(quaternions))
    interpolated_Rs = slerp(new_indices)

    # 对平移向量T使用CubicSpline进行插值
    # spline = CubicSpline(indices, Ts, axis=0)
    # interpolated_Ts = spline(new_indices)
    interp = interp1d(indices, Ts, axis=0, kind='quadratic')  # 这里可以根据需要选择插值的类型，如'linear', 'quadratic', 'cubic'等
    interpolated_Ts = interp(new_indices)
    # 将插值后的四元数旋转转换回旋转矩阵
    interpolated_Rs = [r.as_matrix() for r in interpolated_Rs]

    return interpolated_Rs, interpolated_Ts