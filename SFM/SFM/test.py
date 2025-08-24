# 1.拿到两张图的所有特征点
# 2.随机生成一个匹配关系
# 3.可视化
# 4.使用ransac算法计算本质矩阵
# 5.使用本质矩阵过滤掉错误的匹配点
# 6.可视化

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


def gen_matches(pts1, pts2, num_matches):
    """
    随机生成匹配关系
    """
    idx1 = np.random.choice(pts1.shape[0], num_matches)
    idx2 = np.random.choice(pts2.shape[0], num_matches)

    return pts1[idx1], pts2[idx2]


def ransac(pts1, pts2, threshold=1):
    K = np.array([[4000, 0, 300], [0, 4000, 300], [0, 0, 1]])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=threshold)
    mask = mask.ravel().astype(bool)
    return pts1[mask], pts2[mask]



def main():
    # 读取关键点和匹配信息
    with open('data_iss/keypoints.pkl', 'rb') as file:
        keypoints = pickle.load(file)
    K = np.array([[4000, 0, 180], [0, 4000, 180], [0, 0, 1]])
    pts1 = keypoints[0]
    pts2 = keypoints[11]
    img1 = cv2.imread('images/frame000.png')
    img2 = cv2.imread('images/frame110.png')
    pts1, pts2 = gen_matches(pts1, pts2, 60)
    draw_matches(img1, img2, pts1, pts2)
    pts1, pts2 = ransac(pts1, pts2)
    draw_matches(img1, img2, pts1, pts2)


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


def flip_image(input_path, output_path):
    # 读取图片
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # 左右翻转图片
    flipped_image = cv2.flip(image, 1)  # 参数 1 表示左右翻转，0 表示上下翻转

    # 保存翻转后的图片
    cv2.imwrite(output_path, flipped_image)
    print(f"Flipped image saved to {output_path}")




def process_flip():
    keypoints = pickle.load(open('iss_flip/keypoints.pkl', 'rb'))
    features = pickle.load(open('iss_flip/features.pkl', 'rb'))
    img1 = cv2.imread('images/frame000.png')
    img2 = cv2.imread('images/frame110_flipped.png')

    pass
    # draw_matches()



if __name__ == '__main__':
    flip_image('images/frame000.png', 'images/frame000_flipped.png')