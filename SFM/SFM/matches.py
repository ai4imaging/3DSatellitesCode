import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_matches(i, j):
    """从pkl文件中加载匹配信息"""
    with open('data_iss/matches.pkl', 'rb') as file:
        matches = pickle.load(file)
    for item in matches:
        if item['pair'] == (i, j):
            return item['matches']
    return None

def points_to_keypoints(points):
    """将二维点坐标转换为cv2.KeyPoint类型"""
    return [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in points]

def draw_matches(img1, img2, pts1, pts2, matchesMask=None):
    """可视化特征点匹配关系"""
    # 创建空白画布，用于拼接两张图片
    img_out = np.hstack((img1, img2))

    # 绘制匹配关系
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if matchesMask is not None and not matchesMask[i]:
            continue

        # 将pt1和pt2转换为整数
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2_shifted = (int(pt2[0] + img1.shape[1]), int(pt2[1]))  # 偏移第二张图像的坐标
        color = tuple(np.random.randint(0, 2, 3).tolist())
        color_green = (0, 200, 0)
        cv2.circle(img_out, pt1, 2, color, -1)
        cv2.circle(img_out, pt2_shifted, 2, color_green, -1)
        cv2.line(img_out, pt1, pt2_shifted, color_green, 1)

    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def are_rotations_opposite(R1, R2):
    # 计算相对旋转矩阵
    R_relative = R1.T @ R2

    # 计算相对旋转矩阵的迹
    trace_R_relative = np.trace(R_relative)

    # 计算旋转角度
    theta = np.arccos((trace_R_relative - 1) / 2)

    # 通过旋转角度判断是否相同或相反
    theta_deg = np.degrees(theta)
    print(f"Relative rotation angle: {theta_deg} degrees")

    # if np.isclose(theta_deg, 0, atol=1e-2):
    #     print("Rotations are the same")
    # elif np.isclose(theta_deg, 180, atol=1e-2):
    #     return "Rotations are opposite"
    # else:
    #     return "Rotations are different"


def main():
    # 读取关键点和匹配信息
    with open('data_iss/keypoints.pkl', 'rb') as file:
        keypoints = pickle.load(file)
    K = np.array([[4000, 0, 180], [0, 4000, 180], [0, 0, 1]])


    # 获取第0张和第1张图像的匹配对
    match01 = get_matches(0, 1)

    # 获取匹配的特征点位置
    pts1 = keypoints[0][match01[:, 0]]  # 图像0的特征点
    pts2 = keypoints[1][match01[:, 1]]  # 图像1的特征点

    E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
    retval, R01, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    # print(R01, t)
    # # 使用自定义函数绘制匹配结果
    # draw_matches(img0, img1, pts1, pts2)
    R_prev = np.linalg.inv(R01)
    for i in range(1, 10):
        matchs = get_matches(i, i+1)
        pts1 = keypoints[i][matchs[:, 0]]
        pts2 = keypoints[i+1][matchs[:, 1]]
        E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        R = np.linalg.inv(R)
        are_rotations_opposite(R_prev, R)
        R_prev = R

    # match12 = get_matches(1, 2)
    # pts1 = keypoints[1][match12[:, 0]]
    # pts2 = keypoints[2][match12[:, 1]]
    # E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
    # F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)[0]
    # retval, R12, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    # print(R12, t)
    # are_rotations_opposite(R01, R12)


def test_css():
    pose = np.load("data_iss/optimized_camera.npy")
    pose = np.linalg.inv(pose[:,:3,:3])
    Rs = pose[:, :3]
    for i in range(1, len(Rs)):
        are_rotations_opposite(Rs[i-1], Rs[i])


def test_iss():
    pose = np.load("data_iss/optimized_camera.npy")
    pose = np.linalg.inv(pose[:,:3,:3])
    Rs = pose[:, :3]
    for i in range(1, len(Rs)):
        are_rotations_opposite(Rs[i-1], Rs[i])


if __name__ == '__main__':
    test_iss()
