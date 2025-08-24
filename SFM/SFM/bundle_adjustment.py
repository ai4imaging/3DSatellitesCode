import cv2
import numpy as np
from scipy.optimize import least_squares



def project(points, camera_params, K):
    """将3D点投影到2D平面"""
    points_proj = np.zeros((len(points), 2))

    for i, (point, cam_param) in enumerate(zip(points, camera_params)):
        rvec = cam_param[:3]  # 提取Rodrigues向量
        tvec = cam_param[3:]  # 提取平移向量
        R, _ = cv2.Rodrigues(rvec)
        c2w = np.hstack((R, tvec.reshape(3, 1)))
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


def residuals(params, n_cameras, n_points, points_2d, visibility, K, lambda_distance=0.01):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

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
    reprojection_residuals = (point_proj - np.array(points_2d_lists)).ravel()

    # 添加相机位置正则化项
    position_regularization = []
    for i in range(n_cameras-1):
        t1 = camera_params[i, 3:]
        t2 = camera_params[i+1, 3:]
        distance = np.linalg.norm(t1 - t2)
        position_regularization.append(lambda_distance / (distance + 1e-6))  # 防止除0

    # 合并残差
    total_residuals = np.hstack((reprojection_residuals, position_regularization))
    return total_residuals


def residuals_line_var(params, n_cameras, n_points, points_2d, visibility, K, lambda_distance=0.01, lambda_line=0.01): #simu1
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

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
    reprojection_residuals = (point_proj - np.array(points_2d_lists)).ravel()

    # 添加相机位置正则化项，确保相机保持在一条直线上
    position_regularization = []
    t_start = camera_params[0, 3:]  # 第一台相机的位置
    t_end = camera_params[-1, 3:]   # 最后一台相机的位置
    line_direction = t_end - t_start
    line_direction /= np.linalg.norm(line_direction)  # 单位化

    for i in range(1, n_cameras - 1):
        t_i = camera_params[i, 3:]
        # 计算t_i到直线的距离
        projection = t_start + np.dot(t_i - t_start, line_direction) * line_direction
        distance_to_line = np.linalg.norm(t_i - projection)
        position_regularization.append(lambda_line * distance_to_line)

    # 相邻相机位置的距离正则化
    for i in range(n_cameras - 1):
        t1 = camera_params[i, 3:]
        t2 = camera_params[i + 1, 3:]
        distance = np.linalg.norm(t1 - t2)
        position_regularization.append(lambda_distance / (distance + 1e-6))  # 防止除0

    # var = np.var(points_3d, axis=0)
    #
    # # 合并残差
    total_residuals = np.hstack((reprojection_residuals, position_regularization))
    return total_residuals


def residuals_line(params, n_cameras, n_points, points_2d, visibility, K, lambda_distance=0.01, lambda_line=0.01, lambda_angle=0.1): #simu1
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

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
    reprojection_residuals = (point_proj - np.array(points_2d_lists)).ravel()

    # 添加相机位置正则化项，确保相机保持在一条直线上
    position_regularization = []
    t_start = camera_params[0, 3:]  # 第一台相机的位置
    t_end = camera_params[-1, 3:]   # 最后一台相机的位置
    line_direction = t_end - t_start
    line_direction /= np.linalg.norm(line_direction)  # 单位化

    for i in range(1, n_cameras - 1):
        t_i = camera_params[i, 3:]
        # 计算t_i到直线的距离
        projection = t_start + np.dot(t_i - t_start, line_direction) * line_direction
        distance_to_line = np.linalg.norm(t_i - projection)
        position_regularization.append(lambda_line * distance_to_line)

    # 相邻相机位置的距离正则化
    for i in range(n_cameras - 1):
        t1 = camera_params[i, 3:]
        t2 = camera_params[i + 1, 3:]
        distance = np.linalg.norm(t1 - t2)
        position_regularization.append(lambda_distance / (distance + 1e-6))  # 防止除0

        # 添加朝向夹角正则化项
        angle_regularization = []
        min_cos_angle = np.cos(np.radians(30))  # 50°的余弦值

        for i in range(n_cameras):
            rotation_vector = camera_params[i, :3]
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            camera_z_axis = rotation_matrix[:, 2]  # Z轴方向

            # 计算相机到连线方向的向量
            camera_position = camera_params[i, 3:]
            line_vector = (t_end - camera_position) if i < n_cameras - 1 else (camera_position - t_start)
            line_vector /= np.linalg.norm(line_vector)  # 单位化

            # 计算夹角的余弦值
            cos_angle = np.dot(camera_z_axis, line_vector)
            if cos_angle < min_cos_angle:  # 如果夹角小于50°
                angle_regularization.append(lambda_angle * (min_cos_angle - cos_angle))
            else:
                angle_regularization.append(0)

    # 合并残差
    total_residuals = np.hstack((reprojection_residuals, position_regularization, angle_regularization))
    return total_residuals


def residuals_angle(params, n_cameras, n_points, points_2d, visibility, K, lambda_line=0.01, lambda_angle=0.1): #simu1
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

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
    reprojection_residuals = (point_proj - np.array(points_2d_lists)).ravel()

    # 添加相机位置正则化项，确保相机保持在一条直线上
    position_regularization = []
    t_start = camera_params[0, 3:]  # 第一台相机的位置
    t_end = camera_params[-1, 3:]   # 最后一台相机的位置
    line_direction = t_end - t_start
    line_direction /= np.linalg.norm(line_direction)  # 单位化

    # for i in range(1, n_cameras - 1):
    #     t_i = camera_params[i, 3:]
    #     # 计算t_i到直线的距离
    #     projection = t_start + np.dot(t_i - t_start, line_direction) * line_direction
    #     distance_to_line = np.linalg.norm(t_i - projection)
    #     position_regularization.append(lambda_line * distance_to_line)

    # 相邻相机位置的距离正则化
    for i in range(n_cameras - 1):
        # 添加朝向夹角正则化项
        angle_regularization = []
        min_cos_angle = np.cos(np.radians(50))  # 50°的余弦值

        for i in range(n_cameras):
            rotation_vector = camera_params[i, :3]
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            camera_z_axis = rotation_matrix[:, 2]  # Z轴方向

            # 计算相机到连线方向的向量
            camera_position = camera_params[i, 3:]
            line_vector = (t_end - camera_position) if i < n_cameras - 1 else (camera_position - t_start)
            line_vector /= np.linalg.norm(line_vector)  # 单位化

            # 计算夹角的余弦值
            cos_angle = np.dot(camera_z_axis, line_vector)
            if cos_angle < min_cos_angle:  # 如果夹角小于50°
                angle_regularization.append(lambda_angle * (min_cos_angle - cos_angle))
            else:
                angle_regularization.append(0)

    # 合并残差
    total_residuals = np.hstack((reprojection_residuals, position_regularization, angle_regularization))
    return total_residuals


def residuals_dis(params, n_cameras, n_points, points_2d, visibility, K, distance):
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

    # distance
    point_mean = np.mean(points_3d, axis=0)
    camera_t = camera_params[:, 3:]
    point_mean = np.tile(point_mean, (n_cameras, 1))
    distance_residuals = 10 * abs(np.linalg.norm(camera_t - point_mean, axis=1)-distance[:n_cameras])
    residuals = np.hstack((residuals.ravel(), distance_residuals.ravel()))
    return residuals



def bundle_adjustment(n_cameras, n_points, camera, points_3d, K, points_2d, visibility):
    """使用最小二乘法进行优化"""
    cam_r = camera[:, :3, :3]
    cam_t = camera[:, :3, 3]
    rotation_vectors = np.zeros((n_cameras, 3))
    for i in range(n_cameras):
        rotation_vectors[i] = cv2.Rodrigues(cam_r[i])[0].squeeze(-1)
    camera_params = np.concatenate([rotation_vectors, cam_t], axis=-1)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    result = least_squares(residuals_before, x0, args=(n_cameras, n_points, points_2d, visibility, K))
    print("Final residuals:", result.fun.mean())
    vec = result.x[:n_cameras * 6].reshape((n_cameras, 6))
    rvec = vec[:, :3]
    tvec = vec[:, 3:]
    Rs = np.zeros((n_cameras, 3, 3))
    for i in range(n_cameras):
        Rs[i], _ = cv2.Rodrigues(rvec[i])
    c2w = np.concatenate([Rs, tvec.reshape(-1, 3, 1)], -1)

    return c2w, result.x[n_cameras * 6:].reshape((n_points, 3))


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
