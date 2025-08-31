import time

import numpy as np
import pickle
import json
import cv2
import open3d as o3d
from bundle_adjustment import bundle_adjustment
from visualization import visualization
from bundle_adjustment import project as ba_project


class SFM:
    def __init__(self):
        self.K = np.array([[4000, 0, 180], [0, 4000, 180], [0, 0, 1]], dtype=np.float64)
        with open('data_css/matches.pkl', 'rb') as file:
            self.matches = pickle.load(file)
        with open('data_css/keypoints.pkl', 'rb') as file:
            self.key_points = pickle.load(file)
        cam = np.genfromtxt('data_css/camera.csv', delimiter=',', dtype=np.float64)
        self.camera = cam.reshape(-1, 3, 4)
        # self.camera[:, :3, 1:3] = -self.camera[:, :3, 1:3]
        # self.camera = to_c2w(self.camera)
        self.points_3d = np.genfromtxt('data_css/points.csv', delimiter=',', dtype=np.float64)

        self.n_cameras = self.camera.shape[0]
        self.n_points = self.points_3d.shape[0]
        corr = np.genfromtxt('data_css/corr.csv', delimiter=',')
        corr = corr.transpose()
        self.points_2d = corr.reshape(self.n_points, self.n_cameras, 2)

        co = corr.reshape(-1, 3, 2)
        self.map_3d_2d = dict()
        self.map_2d_3d = dict()
        self.vis = np.ones((self.n_points, self.n_cameras))
        for i, point in enumerate(co):
            for frame in range(point.shape[0]):
                pt = point[frame]
                for j, key in enumerate(self.key_points[frame]):
                    if pt[0] == key[0] and pt[1] == key[1]:
                        self.map_3d_2d[(i, frame)] = j
                        self.map_2d_3d[(frame, j)] = i
                        break
    def ransac_filter(self, matches1, matches2, K, threshold=3.0, confidence=0.999):
        """
        使用RANSAC算法过滤外点匹配点。

        参数:
            matches1 (ndarray): 第一组匹配点，形状为 (N, 2)。
            matches2 (ndarray): 第二组匹配点，形状为 (N, 2)。
            K (ndarray): 相机的内参矩阵，形状为 (3, 3)。
            threshold (float): RANSAC 内点判定的误差阈值。
            confidence (float): RANSAC 的置信度。

        返回:
            inliers_matches1 (ndarray): 过滤后的第一组内点匹配点。
            inliers_matches2 (ndarray): 过滤后的第二组内点匹配点。
            mask (ndarray): 内点掩码，值为 1 表示内点，0 表示外点。
        """
        # 将匹配点转换为齐次坐标
        points1 = np.hstack((matches1, np.ones((matches1.shape[0], 1))))
        points2 = np.hstack((matches2, np.ones((matches2.shape[0], 1))))

        # 使用OpenCV的findFundamentalMat进行RANSAC过滤
        F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_RANSAC, ransacReprojThreshold=threshold,
                                         confidence=confidence)

        # 根据mask提取内点
        inliers_matches1 = matches1[mask.ravel() == 1]
        inliers_matches2 = matches2[mask.ravel() == 1]

        return inliers_matches1, inliers_matches2, mask
    def get_matches(self, m1,m2):
        for item in self.matches:
            if item['pair'] == (m1, m2):
                match = item['matches']
                res = self.ransac_filter(self.key_points[m1][match[:, 0]], self.key_points[m2][match[:, 1]], self.K)
                match = match[res[2].ravel() == 1]
                return match

    def visualize(self):
        visualization( to_c2w(self.camera), self.points_3d)

    def triangulate_points(self, R1, t1, K1, R2, t2, K2, pts1, pts2):
        P1 = K1 @ np.hstack((R1, t1.reshape(-1, 1)))
        P2 = K2 @ np.hstack((R2, t2.reshape(-1, 1)))
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3D = pts4D[:3] / pts4D[3]
        return pts3D.T

    def triangulate(self, m1, m2):
        tri_points_2d = []
        tri_points_2d_points = []

        matches = self.get_matches(m1, m2)
        for rela in matches:
            tmp_points = []
            if (m1, rela[0]) not in self.map_2d_3d:
                tmp_points.append(self.key_points[m1][rela[0]])
                tmp_points.append(self.key_points[m2][rela[1]])
                mh = self.get_matches(m1-1, m1)
                # for it in mh:
                #     if it[1] == rela[0]:
                #         tmp.append((m1-1, it[0]))
                #         break
                tri_points_2d.append(rela)
                tri_points_2d_points.append(tmp_points)
        points_2ds = np.array(tri_points_2d_points, dtype=np.float32)
        np.save('data_css/points_2ds.npy', points_2ds)
        p_2d_1 = points_2ds[:, 0]
        p_2d_2 = points_2ds[:, 1]
        R1 = self.camera[m1][:3, :3]
        t1 = self.camera[m1][:3, 3]
        R2 = self.camera[m2][:3, :3]
        t2 = self.camera[m2][:3, 3]
        points_3d = self.triangulate_points(R1, t1, self.K, R2, t2, self.K, p_2d_1, p_2d_2)
        # visualization(self.camera[m1:], np.concatenate([points_3d, self.points_3d], 0))


        vis = np.zeros((len(tri_points_2d), self.n_cameras))
        con_p2d = np.zeros((len(tri_points_2d), self.n_cameras, 2))
        for i, rela in enumerate(tri_points_2d):
            self.map_2d_3d[(m1, rela[0])] = self.n_points + i
            self.map_2d_3d[(m2, rela[1])] = self.n_points + i
            self.map_3d_2d[(self.n_points + i, m1)] = rela[0]
            self.map_3d_2d[(self.n_points + i, m2)] = rela[1]
            vis[i][m1] = 1
            vis[i][m2] = 1
            con_p2d[i][m1] = self.key_points[m1][rela[0]]
            con_p2d[i][m2] = self.key_points[m2][rela[1]]
        self.points_2d = np.concatenate([self.points_2d, con_p2d], 0)
        self.vis = np.concatenate([self.vis, vis], 0)
        self.n_points += points_3d.shape[0]
        self.points_3d = np.concatenate([self.points_3d, points_3d], 0)
        print(points_2ds.shape)


    def add_camera(self, m1, m2):
        p3d = []
        p2d = []
        p3d_id = []
        matches = self.get_matches(m1, m2)
        for rela in matches:
            if (m1, rela[0]) not in self.map_2d_3d:
                continue
            p3d_id.append(self.map_2d_3d[(m1, rela[0])])
            p3d.append(self.points_3d[self.map_2d_3d[(m1, rela[0])]])
            p2d.append(self.key_points[m2][rela[1]])
            self.map_2d_3d[(m2, rela[1])] = self.map_2d_3d[(m1, rela[0])]
            self.map_3d_2d[(self.map_2d_3d[(m1, rela[0])], m2)] = rela[1]

        con_p2d = np.zeros((self.n_points, 1, 2))
        vis = np.zeros((self.n_points, 1))
        for item, p2 in zip(p3d_id, p2d):
            con_p2d[item] = p2
            vis[item] = 1

        if m1 == 4:
            print('5')

        p2d = np.concatenate([self.points_2d, con_p2d], axis=1)
        vis = np.concatenate([self.vis, vis], axis=1)
        cam = np.concatenate([self.camera, self.camera[-1][None, ...]], axis=0)
        # visualization(self.camera, self.points_3d)
        print('Optimizing',m1,m2)
        start_time = time.time()
        optimized_camera, optimized_points_3d = bundle_adjustment(self.n_cameras + 1, self.n_points, cam, self.points_3d, self.K, p2d, vis)
        end_time = time.time()
        print('Optimized time', m1, m2, end_time-start_time)

        # visualization(optimized_camera, optimized_points_3d)
        if m2==8:
            print('9')
        self.camera = optimized_camera
        self.n_cameras += 1
        self.points_3d = optimized_points_3d
        self.points_2d = p2d
        self.vis = vis
        np.save('data_css/optimized_camera.npy', optimized_camera)
        np.save('data_css/optimized_points_3d.npy', optimized_points_3d)
        if m1 == 4:
            self.triangulate(m1, m2)
        print('Optimized completed',m1,m2)

    def test_ba(self,frame):
        camera = np.tile(self.camera[frame], (self.points_3d.shape[0], 1, 1))
        n_cameras = camera.shape[0]
        cam_r = camera[:, :3, :3]
        cam_t = camera[:, :3, 3]
        rotation_vectors = np.zeros((n_cameras, 3))
        for i in range(n_cameras):
            rotation_vectors[i] = cv2.Rodrigues(cam_r[i])[0].squeeze(-1)
        camera_params = np.concatenate([rotation_vectors, cam_t], axis=-1)
        points_proj = ba_project(self.points_3d, camera_params, self.K)
        print(points_proj, points_proj.shape)

def test_ba_project():
    K = np.array([[4000, 0, 180], [0, 4000, 180], [0, 0, 1]], dtype=np.float64)
    camera = np.load('data_css/optimized_camera.npy')
    points_3d = np.load('data_css/optimized_points_3d.npy')
    print(camera.shape, points_3d.shape)
    camera = to_c2w(camera)
    frame = 1
    camera = np.tile(camera[frame], (points_3d.shape[-1], 1, 1))
    n_cameras = camera.shape[0]
    cam_r = camera[:, :3, :3]
    cam_t = camera[:, :3, 3]
    rotation_vectors = np.zeros((n_cameras, 3))
    for i in range(n_cameras):
        rotation_vectors[i] = cv2.Rodrigues(cam_r[i])[0].squeeze(-1)
    camera_params = np.concatenate([rotation_vectors, cam_t], axis=-1)
    points_proj = ba_project(points_3d,camera_params, K)
    print(points_proj, points_proj.shape)

def main():
    sfm = SFM()
    # sfm.test_ba(1)
    for i in range(2, 13):
        sfm.add_camera(i, i+1)
    sfm.visualize()


def to_c2w(R_t_batch):
    """
    Converts an n*3*4 R_t matrix batch to an n*4*4 camera-to-world (c2w) matrix batch.

    Parameters:
    R_t_batch (numpy.ndarray): n*3*4 matrix, where each 3x4 matrix contains the rotation matrix (R)
                               and the translation vector (t) for n cameras.

    Returns:
    numpy.ndarray: n*4*4 matrix batch representing the camera-to-world (c2w) transformation
                   for n cameras.
    """
    n = R_t_batch.shape[0]  # Number of cameras
    c2w_batch = np.zeros((n, 4, 4))  # Initialize an empty n*4*4 matrix

    for i in range(n):
        R_t = R_t_batch[i]  # Extract the i-th camera's 3x4 matrix
        R = R_t[:, :3]      # Rotation matrix (3x3)
        t = R_t[:, 3]       # Translation vector (3x1)

        # Compute the inverse of R (which is the transpose for a rotation matrix)
        R_inv = R.T

        # Compute the new translation vector
        t_new = -np.dot(R_inv, t)

        # Construct the 4x4 c2w matrix for this camera
        c2w = np.eye(4)     # Initialize a 4x4 identity matrix
        c2w[:3, :3] = R_inv # Set the rotation part
        c2w[:3, 3] = t_new  # Set the translation part

        # Store in the batch
        c2w_batch[i] = c2w
    c2w_batch[:, :3, 1:3] = -c2w_batch[:, :3, 1:3]
    return c2w_batch


def compute_camera_direction_angle(c2w1, c2w2, in_degrees=True):
    """
    计算两个相机朝向（Z轴负方向）之间的夹角。

    参数：
        c2w1: np.ndarray，shape=(4, 4)，第一个相机的 c2w 矩阵
        c2w2: np.ndarray，shape=(4, 4)，第二个相机的 c2w 矩阵
        in_degrees: 是否返回角度（True）或弧度（False）

    返回：
        angle: 两个相机朝向之间的夹角
    """
    # 提取朝向向量：第三列是相机 Z 轴方向（正方向），view direction 通常为 -Z
    dir1 = -c2w1[:3, 2]
    dir2 = -c2w2[:3, 2]

    # 单位化
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)

    # 计算夹角
    dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0)  # 避免数值误差导致 >1 或 <-1
    angle_rad = np.arccos(dot_product)

    if in_degrees:
        return np.degrees(angle_rad)
    return angle_rad


def test():
    camera = np.load('data_css/optimized_camera.npy')
    points_3d = np.load('data_css/optimized_points_3d.npy')
    c2ws = np.load('data_css/c2ws_fined.npy')
    cameras = to_c2w(camera)
    # cameras = np.concatenate([c2ws, cameras], 0)
    camera_t = cameras[:, :3, 3]
    point_mean = np.mean(points_3d, axis=0)

    distance = np.linalg.norm(point_mean - camera_t, axis=1)
    print(point_mean, distance)
    print(compute_camera_direction_angle(cameras[0], cameras[-1]))
    visualization(cameras, points_3d)
    frame_ids = [f"frame{i:03}" for i in range(0, 140, 10)]
    frame_ids = sorted(frame_ids)
    create_json(cameras.tolist(), frame_ids, 0.08040715440186917, 'data_css/transforms_train.json')
    create_json(cameras.tolist(), frame_ids, 0.08040715440186917, 'data_css/transforms_test.json')
    create_json(cameras.tolist(), frame_ids, 0.08040715440186917, 'data_css/transforms_val.json')

def pose_init():
    camera = np.load('data_css/optimized_camera.npy')
    points_3d = np.load('data_css/optimized_points_3d.npy')
    cameras = to_c2w(camera)
    for i in range(len(cameras)):
        if i>5:
            visualization(cameras[:i+1], points_3d, 'media/init' + str(i))
        else:
            visualization(cameras[:i + 1], points_3d[::2], 'media/init' + str(i))


def pose_refine():
    camera = np.load('data_css/optimized_camera.npy')
    points_3d = np.load('data_css/optimized_points_3d.npy')
    cameras = to_c2w(camera)

    noise_level = 0.2  # 控制噪声的大小，越大越无序
    noisy_cameras = cameras
    for i in range(10):
        # 为每个相机添加噪声
        noisy_cameras = add_noise_to_cameras(noisy_cameras, noise_level)

        # 渲染带噪声的相机位置
        visualization(noisy_cameras, points_3d, 'media/refine' + str(i))


def add_noise_to_cameras(cameras, noise_level):
    """
    Adds noise to the camera-to-world matrices.

    Parameters:
    cameras (numpy.ndarray): The camera-to-world matrices.
    noise_level (float): The level of noise to add to the camera positions.

    Returns:
    numpy.ndarray: The new batch of camera-to-world matrices with added noise.
    """
    noisy_cameras = cameras.copy()

    # 随机生成噪声（可调节的高斯噪声）
    noise_translation = np.random.normal(0, noise_level, size=noisy_cameras[:, :3, 3].shape)
    noise_rotation = np.random.normal(0, noise_level/10, size=noisy_cameras[:, :3, :3].shape)

    # 添加噪声到相机的平移向量
    noisy_cameras[:, :3, 3] += noise_translation

    # 添加噪声到相机的旋转矩阵（可以按需使用旋转噪声）
    for i in range(noisy_cameras.shape[0]):
        # 使用噪声扰动旋转矩阵（以小的旋转噪声为例）
        R = noisy_cameras[i, :3, :3]
        noisy_rotation = R + noise_rotation[i]

        # 正常化旋转矩阵（可以通过SVD或其他方法保持其正交性）
        U, _, Vt = np.linalg.svd(noisy_rotation)
        noisy_cameras[i, :3, :3] = np.dot(U, Vt)  # 保持正交矩阵

    return noisy_cameras


def triangulate_points(R1, t1, K1, R2, t2, K2, pts1, pts2):
    print(pts1.shape,pts2.shape)
    P1 = K1 @ np.hstack((R1, t1.reshape(-1, 1)))
    P2 = K2 @ np.hstack((R2, t2.reshape(-1, 1)))
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.T


def distance_point_to_line(c2w, point):
    """
    计算从相机原点朝向与c2w矩阵的朝向相同的直线到三维点的距离。

    参数:
    c2w : 4x4 numpy array
        从相机坐标系到世界坐标系的变换矩阵
    point : 1x3 numpy array
        三维点的坐标

    返回:
    distance : float
        直线到三维点的距离
    """
    # 提取相机的世界坐标原点（c2w的最后一列表示相机位置）
    camera_origin = c2w[:3, 3]

    # 提取 c2w 矩阵的 z 轴方向（朝向向量，通常是第三列）
    direction_vector = c2w[:3, 2]

    # 计算点 P 到相机原点 O 的向量
    point_to_origin = point - camera_origin

    # 计算点到直线的距离
    distance = np.linalg.norm(np.cross(point_to_origin, direction_vector)) / np.linalg.norm(direction_vector)

    return distance


def create_image_with_points_and_lines(points1, points2, save_path='output_image.png'):
    """
    生成360*720的图片，其中包含两张360*360的图片，两组二维点分别在两侧，
    并在两组点之间连线。

    参数:
    points1: n*2的ndarray，第一组二维坐标，放在左侧
    points2: n*2的ndarray，第二组二维坐标，放在右侧
    save_path: 保存图片的路径，默认为'output_image.png'
    """
    # 创建空白图片，大小为360*720，RGB图像
    img = np.ones((360, 720, 3), dtype=np.uint8) * 255  # 白色背景

    # 绘制第一组点到左侧的360x360区域
    for point in points1:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < 360 and 0 <= y < 360:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)  # 蓝色点

    # 绘制第二组点到右侧的360x360区域
    for point in points2:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < 360 and 0 <= y < 360:
            cv2.circle(img, (x + 360, y), 3, (0, 0, 255), -1)  # 红色点

    # 绘制点对之间的连线
    for p1, p2 in zip(points1, points2):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]) + 360, int(p2[1])  # 右侧图片点的x坐标需要加上360的偏移量
        if (0 <= x1 < 360 and 0 <= y1 < 360) and (360 <= x2 < 720 and 0 <= y2 < 360):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色连线

    # 保存图片
    cv2.imwrite(save_path, img)
    print(f"图片已保存为: {save_path}")



def load_json(input_file):
    """
    从指定的JSON文件中加载数据。

    :param input_file: 输入JSON文件的路径。
    :return: 一个包含c2w转换矩阵的列表和相机的X轴视角。
    """
    # 读取JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 提取数据
    camera_transforms = [frame['transform_matrix'] for frame in data['frames']]
    camera_angle_x = data['camera_angle_x']

    return np.array(camera_transforms)

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


def project(points, camera, K):
    if camera.shape[0] == 3:
        camera = np.concatenate([camera, np.array([[0, 0, 0, 1]])], axis=0)
    point_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # 扩展为4x1的齐次坐标
    point_cam = camera @ point_h.T
    point_proj = K @ point_cam[:3]
    points_proj = point_proj[:2] / point_proj[2]

    return points_proj.T


def test_projection():
    K = np.array([[4000, 0, 180], [0, 4000, 180], [0, 0, 1]], dtype=np.float64)
    cameras = np.load('data_css/optimized_camera.npy')
    points_3d = np.load('data_css/optimized_points_3d.npy')
    frame = 0
    project_2d = project(points_3d, cameras[frame], K)

    return
    c2ws = np.concatenate([cameras, np.tile(np.array([[[0, 0, 0, 1]]]), (cameras.shape[0], 1, 1))], axis=1)
    frame_ids = [f"frame{i:03}" for i in range(0, 140, 10)]
    frame_ids = sorted(frame_ids)
    create_json(c2ws.tolist(), frame_ids, 0.08040715440186917, 'data_css/transformer_train.json')


def test_lego():
    c2ws = load_json('data_css/transforms_train_lego.json')
    # c2ws[:, :3, 1:3] *= -1
    visualization(np.array(c2ws), None, grad_color=False)


if __name__ == '__main__':
    # pose_init()
    # pose_refine()
    # test_lego()
    main()
    test()
    # test_ba_project()

# retval, rvec, tvec = cv2.solvePnP(points_3d, corr[:,4:6], np.array(K), None)
# imagePoints, _ = cv2.projectPoints(points_3d, rvec, tvec, np.array(K, dtype=np.float32), None)





def show_camera(rvec, tvec, points_3d):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # camera_position = -rotation_matrix.T @ tvec

    pose = np.concatenate([rotation_matrix, tvec], axis=1)
    np.save('data_css/pose3.npy', pose)

    # 创建 Open3D 的点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # 创建相机的坐标系表示
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tvec.flatten())

    # 创建相机到点云的连线
    lines = [[0, i + 1] for i in range(len(points_3d))]
    line_points = np.vstack([tvec.flatten(), points_3d])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 创建 Open3D 可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加几何体到可视化器
    vis.add_geometry(pcd)
    vis.add_geometry(camera_frame)
    vis.add_geometry(line_set)

    # 运行可视化器
    vis.run()
    vis.destroy_window()


def show2d_3d(p_2d, p_3d):
    points_3d_on_xy = np.hstack([p_2d, np.ones((p_2d.shape[0], 1))])

    # 创建 3D 点云对象（p_3d）
    pcd_3d = o3d.geometry.PointCloud()
    pcd_3d.points = o3d.utility.Vector3dVector(p_3d)
    pcd_3d.paint_uniform_color([0, 1, 0])  # 设置 3D 点的颜色为绿色

    # 创建 2D 点在 XY 平面的 3D 表示（points_3d_on_xy）
    pcd_2d_on_xy = o3d.geometry.PointCloud()
    pcd_2d_on_xy.points = o3d.utility.Vector3dVector(points_3d_on_xy)
    pcd_2d_on_xy.paint_uniform_color([1, 0, 0])  # 设置投影点的颜色为红色

    # 创建 3D 点与其在 XY 平面上投影点之间的连线
    lines = [[i, i + len(p_3d)] for i in range(len(p_3d))]
    line_points = np.vstack([p_3d, points_3d_on_xy])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    colors = [[0, 0, 1] for _ in range(len(lines))]  # 设置连线的颜色为蓝色
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 创建 Open3D 可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加 3D 点云、XY 平面的投影点云和连线到可视化器
    vis.add_geometry(pcd_3d)
    vis.add_geometry(pcd_2d_on_xy)
    vis.add_geometry(line_set)

    # 运行可视化器
    vis.run()
    vis.destroy_window()


