import pickle
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def draw_matches_three(img1, img2, img3, pts1, pts2, pts3, matchesMask=None):
    """可视化三张图片的特征点匹配关系"""
    # 创建空白画布，用于拼接三张图片
    img_out = np.hstack((img1, img2, img3))

    # 绘制第一张和第二张图片的匹配关系
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if matchesMask is not None and not matchesMask[i]:
            continue

        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2_shifted = (int(pt2[0] + img1.shape[1]), int(pt2[1]))  # 偏移第二张图像的坐标
        color = tuple(np.random.randint(0, 2, 3).tolist())
        color_green = (0, 0, 200)

        # 在第一张和第二张图像上绘制匹配点和连线
        cv2.circle(img_out, pt1, 2, color, -1)
        cv2.circle(img_out, pt2_shifted, 2, color_green, -1)
        cv2.line(img_out, pt1, pt2_shifted, color_green, 1)

    # 绘制第二张和第三张图片的匹配关系
    for i, (pt2, pt3) in enumerate(zip(pts2, pts3)):
        if matchesMask is not None and not matchesMask[i]:
            continue

        pt2_shifted = (int(pt2[0] + img1.shape[1]), int(pt2[1]))  # 偏移第二张图像的坐标
        pt3_shifted = (int(pt3[0] + img1.shape[1] + img2.shape[1]), int(pt3[1]))  # 偏移第三张图像的坐标
        color = tuple(np.random.randint(0, 2, 3).tolist())
        color_green = (0, 0, 200)

        # 在第二张和第三张图像上绘制匹配点和连线
        cv2.circle(img_out, pt2_shifted, 2, color, -1)
        cv2.circle(img_out, pt3_shifted, 2, color_green, -1)
        cv2.line(img_out, pt2_shifted, pt3_shifted, color_green, 1)
    cv2.imwrite('match_images/matches.png', img_out)
    # 显示拼接后的图片
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

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
        color_green = (0, 0, 200)
        cv2.circle(img_out, pt1, 2, color, -1)
        cv2.circle(img_out, pt2_shifted, 2, color_green, -1)
        cv2.line(img_out, pt1, pt2_shifted, color_green, 1)
    cv2.imwrite('match_images/matches.png', img_out)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def convert_points_to_keypoints(points):
    """
    将n*2格式的特征点坐标转换为cv2.KeyPoint格式
    """
    keypoints = []
    for point in points:
        keypoint = cv2.KeyPoint(x=point[0], y=point[1], size=1)  # 设置size为1
        keypoints.append(keypoint)
    return keypoints

# features = pickle.load(open('data_iss/features.pkl', 'rb'))
# key_points = pickle.load(open('data_iss/keypoints.pkl', 'rb'))
# key_points_flipped = pickle.load(open('iss_flip/keypoints.pkl', 'rb'))
#
# points2 = np.concatenate([key_points[0], key_points_flipped[0][:, ::-1]], axis=0)
# points1 = key_points[11]
# descriptors2 = features[0]['descriptors'].transpose().astype(np.float32)
# descriptors1 = features[11]['descriptors'].transpose().astype(np.float32)
# descriptors2_flipped = features[11]['descriptors'].transpose().astype(np.float32)
# descriptors2 = np.concatenate([descriptors2, descriptors2_flipped])
# keypoints1 = convert_points_to_keypoints(points1)
# keypoints2 = convert_points_to_keypoints(points2)
#
#
#
#
# # 3. 使用 BFMatcher 进行特征点匹配
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # 使用 L2 范数，适合 SIFT 或其他浮点型描述符
# matches = bf.knnMatch(descriptors1, descriptors2, k=8)
#
#
# # 5. 可视化匹配结果
# img1 = cv2.imread('images/frame110.png')  # 读取图像1
# img2 = cv2.imread('images/frame000.png')  # 读取图像1


# matches = matches[2:]
#
# show_matches = 8
#
# pts1 = np.zeros((show_matches*2, 2))
# pts2 = np.zeros((show_matches*2, 2))
# for id in range(show_matches//4):
#     for i, match in enumerate(matches[id]):
#         pts1[id*8 + i] = keypoints1[match.queryIdx].pt
#         pts2[id*8 + i] = keypoints2[match.trainIdx].pt



def draw_tow():

    a,b,c = 0,1

    img1 = cv2.imread('css_processed/frame000.png')  # 读取图像1
    img2 = cv2.imread('css_processed/frame010.png')  # 读取图像1
    img3 = cv2.imread('css_processed/frame020.png')  # 读取图像1

    features = pickle.load(open('data_css/features.pkl', 'rb'))
    mateches = pickle.load(open('data_css/matches.pkl', 'rb'))
    keypoints = pickle.load(open('data_css/keypoints.pkl', 'rb'))
    for item in mateches:
        if item["pair"] == (a, b):
            match = item['matches']
            break

    pts1 = []
    pts2 = []

    for i, match in enumerate(match):
        pts1.append(keypoints[a][match[0]])
        pts2.append(keypoints[b][match[1]])
    draw_matches(img1, img2, pts1, pts2)


def draw_three():
    corr = np.genfromtxt('data_css/corr.csv', delimiter=',')
    corr = corr.transpose()
    points_2d = corr.reshape(-1, 3, 2)
    print(points_2d.shape)

    img1 = cv2.imread('css_processed/frame000.png')  # 读取图像1
    img2 = cv2.imread('css_processed/frame010.png')  # 读取图像1
    img3 = cv2.imread('css_processed/frame020.png')  # 读取图像1

    draw_matches_three(img1, img2, img3, points_2d[:,0], points_2d[:,1], points_2d[:,2])


if __name__ == '__main__':
    draw_three()
    # draw_matches(img1, img2, pts1, pts2)
    # draw_matches_three(img1, img2, img3, pts1, pts2, pts3)