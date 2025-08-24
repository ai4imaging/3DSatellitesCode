import cv2
import numpy as np
from scipy.optimize import least_squares



def c2w_to_w2c(c2w):
    # 检查输入是否为 (n, 3, 4) 或 (n, 4, 4) 的形状
    if c2w.ndim != 3 or (c2w.shape[1:] != (3, 4) and c2w.shape[1:] != (4, 4)):
        raise ValueError("输入的 c2w 必须是形状为 (n, 3, 4) 或 (n, 4, 4) 的数组")

    # 如果是 (n, 3, 4)，扩展为 (n, 4, 4)
    if c2w.shape[1:] == (3, 4):
        n = c2w.shape[0]
        c2w_homogeneous = np.zeros((n, 4, 4))
        c2w_homogeneous[:, :3, :4] = c2w
        c2w_homogeneous[:, 3, 3] = 1  # 设置最后一行为 [0, 0, 0, 1]
    else:
        c2w_homogeneous = c2w  # 如果已经是 (n, 4, 4)，直接使用

    # 计算逆矩阵，得到 w2c
    w2c_homogeneous = np.linalg.inv(c2w_homogeneous)

    # 返回 (n, 3, 4) 的结果
    return w2c_homogeneous[:, :3, :4]


def w2c_to_c2w(w2c):
    # 检查输入是否为 (n, 3, 4) 或 (n, 4, 4) 的形状
    if w2c.ndim != 3 or (w2c.shape[1:] != (3, 4) and w2c.shape[1:] != (4, 4)):
        raise ValueError("输入的 w2c 必须是形状为 (n, 3, 4) 或 (n, 4, 4) 的数组")

    # 如果是 (n, 3, 4)，扩展为 (n, 4, 4)
    if w2c.shape[1:] == (3, 4):
        n = w2c.shape[0]
        w2c_homogeneous = np.zeros((n, 4, 4))
        w2c_homogeneous[:, :3, :4] = w2c
        w2c_homogeneous[:, 3, 3] = 1  # 设置最后一行为 [0, 0, 0, 1]
    else:
        w2c_homogeneous = w2c  # 如果已经是 (n, 4, 4)，直接使用

    # 计算逆矩阵，得到 c2w
    c2w_homogeneous = np.linalg.inv(w2c_homogeneous)

    # 返回 (n, 3, 4) 的结果
    return c2w_homogeneous[:, :3, :4]

def project(points, camera_params, K):
    """将3D点投影到2D平面"""
    points_proj = np.zeros((len(points), 2))

    for i, (point, cam_param) in enumerate(zip(points, camera_params)):
        rvec = cam_param[:3]  # 提取Rodrigues向量
        tvec = cam_param[3:]  # 提取平移向量
        R, _ = cv2.Rodrigues(rvec)
        w2c = np.hstack((R, tvec.reshape(3, 1)))
        c2w = np.linalg.inv(np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0))
        point_h = np.append(point, 1)  # 扩展为4x1的齐次坐标
        point_cam = c2w @ point_h
        point_proj = K @ point_cam[:3]
        points_proj[i] = point_proj[:2] / point_proj[2]

    return points_proj


def rotation_matrix(rvec):
    """根据欧拉角计算旋转矩阵"""
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return np.eye(3)
    rvec = rvec / theta
    K = np.array([
        [0, -rvec[2], rvec[1]],
        [rvec[2], 0, -rvec[0]],
        [-rvec[1], rvec[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def residuals_before(params, n_cameras, n_points, points_2d, visibility, K):
    """计算残差，即重投影误差"""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

    points_3d_lists = []
    camera__params_lists = []
    points_2d_lists = []
    for i in range(n_points):
        for j in range(n_cameras):
            if visibility[i, j] !=0:
                points_3d_lists.append(points_3d[i])
                camera__params_lists.append(camera_params[j])
                points_2d_lists.append(points_2d[i, j])
    point_proj = project(points_3d_lists, camera__params_lists, K)
    residuals = point_proj - np.array(points_2d_lists)
    # trans_error = camera__params_lists[0][:, 3] + camera__params_lists[2][:, 3] - camera__params_lists[1][:, 3]
    # return np.hstack((residuals.ravel(), trans_error.ravel()/100))
    return residuals.ravel()




def residuals(params, n_cameras, n_points, points_2d, visibility, K, cam_origin, direction, dis):
    camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
    position = params[n_cameras * 3:(n_cameras*4)-1].reshape((-1, 1))
    position = np.concatenate([np.zeros((1, 1)), position], axis=0)
    positions = []
    for pos in position:
        positions.append(point_from_distance_and_direction(pos, direction, cam_origin))
    positions = np.array(positions)
    camera_params = np.concatenate([camera_params, positions], axis=-1)
    points_3d = params[(n_cameras*4)-1:].reshape((n_points, 3))

    points_3d_lists = []
    camera__params_lists = []
    points_2d_lists = []
    for i in range(n_points):
        for j in range(n_cameras):
            if visibility[i, j]:
                points_3d_lists.append(points_3d[i])
                camera__params_lists.append(camera_params[j])
                points_2d_lists.append(points_2d[i, j])

    # 计算重投影误差
    point_proj = project(points_3d_lists, camera__params_lists, K)
    reprojection_residuals = (point_proj - np.array(points_2d_lists)).ravel()*10


    # position_error = position[1:] - position[:-1]
    # dir = position[1]-position[0]
    # dir = dir / np.linalg.norm(dir)
    # position_error = np.maximum(0, -dir * position_error*1000000)
    # # print(position_error, position, position_error.shape)
    # position_error = position_error.ravel()
    #
    # distance = np.linalg.norm(position[1:] - position[:-1], axis=1)
    # distance[distance <= dis] = 1000
    # distance[distance > dis] = 0
    #
    # error = np.hstack((reprojection_residuals, position_error, distance))
    return reprojection_residuals


def calculate_positions_on_line(direction, point1, point2):
    # 将输入的三个点转换为numpy数组
    p1 = np.array(point1)
    p2 = np.array(point2)

    # 计算 p2 和 p3 在 p1 的方向向量上的投影
    d2 = np.dot(p2 - p1, direction)  # p2 到 p1 的距离

    return d2


def point_from_distance_and_direction(distance, direction, origin=(0, 0, 0)):
    # 将方向向量单位化
    direction = np.array(direction)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("方向向量不能为零向量")

    direction_unit = direction / direction_norm  # 单位方向向量
    origin = np.array(origin)  # 将起点坐标转换为numpy数组

    # 计算目标点的坐标
    point = origin + distance * direction_unit

    return tuple(point)  # 返回三维坐标作为一个元组



def bundle_adjustment(n_cameras, n_points, camera, points_3d, K, points_2d, visibility):
    """使用最小二乘法进行优化"""
    camera = w2c_to_c2w(camera)
    cam_r = camera[:, :3, :3]
    cam_t = camera[:, :3, 3]
    rotation_vectors = np.zeros((n_cameras, 3))
    for i in range(n_cameras):
        rotation_vectors[i] = cv2.Rodrigues(cam_r[i])[0].squeeze(-1)
    direction = cam_t[-1] - cam_t[0]
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    position = []
    for i in range(1, n_cameras):
        position.append(calculate_positions_on_line(direction_unit, cam_t[0], cam_t[i]))
    position = np.array(position)
    x0 = np.hstack((rotation_vectors.ravel(), position.ravel(), points_3d.ravel()))
    result = least_squares(residuals, x0, args=(n_cameras, n_points, points_2d, visibility, K, cam_t[0], direction, np.linalg.norm(cam_t[0]-cam_t[1])/3))

    params = result.x
    camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
    position = params[n_cameras * 3:(n_cameras*4)-1].reshape((-1, 1))
    position = np.concatenate([np.zeros((1, 1)), position], axis=0)
    positions = []
    for pos in position:
        positions.append(point_from_distance_and_direction(pos, direction, cam_t[0]))
    positions = np.array(positions)
    camera_params = np.concatenate([camera_params, positions], axis=-1)
    points_3d = params[(n_cameras*4)-1:].reshape((n_points, 3))
    print("Final residuals:", result.fun.mean())
    vec = camera_params.reshape((n_cameras, 6))
    rvec = vec[:, :3]
    tvec = vec[:, 3:]
    Rs = np.zeros((n_cameras, 3, 3))
    for i in range(n_cameras):
        Rs[i], _ = cv2.Rodrigues(rvec[i])
    c2w = np.concatenate([Rs, tvec.reshape(-1, 3, 1)], -1)
    w2c = c2w_to_w2c(c2w)
    return w2c, points_3d


# # 将所有参数展开为一个长向量
# x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
#
# # 使用最小二乘法进行优化
# result = least_squares(residuals, x0, args=(n_cameras, n_points, points_2d, visibility))
#
# # 提取优化后的相机参数和3D点
# optimized_camera_params = result.x[:n_cameras * 6].reshape((n_cameras, 6))
# optimized_points_3d = result.x[n_cameras * 6:].reshape((n_points, 3))

def main():
    optimized_camera_params, optimized_points_3d = bundle_adjustment(n_cameras, n_points, camera_params, points_3d,
                                                                     K, points_2d, visibility)
    print("优化后的相机参数:\n", optimized_camera_params)
    print("优化后的3D点:\n", optimized_points_3d)
