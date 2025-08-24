import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt



with open('matches.pkl', 'rb') as file:
    matches = pickle.load(file)

with open('keypoints.pkl', 'rb') as file:
    keypoints = pickle.load(file)


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


def ransac_filter(matches1, matches2, K, threshold=3.0, confidence=0.999):
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


def get_matches(m1, m2):
    for item in matches:
        if item['pair'] == (m1, m2):
            return item['matches']


K = np.array([[4122, 0, 180], [0, 4122, 180], [0, 0, 1]])
match = get_matches(5, 6)
keypoints5 = keypoints[7]
keypoints6 = keypoints[8]

img1 = cv2.imread('simu2/images/frame070.jpg')
img2 = cv2.imread('simu2/images/frame080.jpg')

draw_matches(img1, img2, keypoints5[match[:, 0]], keypoints6[match[:, 1]])
res = ransac_filter(keypoints5[match[:, 0]], keypoints6[match[:, 1]], K)
print(res[2], match[res[2]])
match = match[res[2].ravel() == 1]
draw_matches(img1, img2, res[0], res[1])

res = ransac_filter(keypoints5[match[:, 0]], keypoints6[match[:, 1]], K)
print(res[2], match[res[2]])
match = match[res[2].ravel() == 1]
draw_matches(img1, img2, res[0], res[1])

res = ransac_filter(keypoints5[match[:, 0]], keypoints6[match[:, 1]], K)
print(res[2], match[res[2]])
match = match[res[2].ravel() == 1]
draw_matches(img1, img2, res[0], res[1])


res = ransac_filter(keypoints5[match[:, 0]], keypoints6[match[:, 1]], K)
print(res[2], match[res[2]])
match = match[res[2].ravel() == 1]
draw_matches(img1, img2, res[0], res[1])