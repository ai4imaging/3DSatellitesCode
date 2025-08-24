import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

K = np.array([[4122, 0, 180],
              [0, 4122, 180],
              [0, 0, 1]])
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


with open('simu3/matches.pkl', 'rb') as file:
    matches = pickle.load(file)

with open('simu3/keypoints.pkl', 'rb') as file:
    key_points = pickle.load(file)





match01=None
match12=None
match02=None

a,b,c = 0,1,2

for item in matches:
    print('pair', item['pair'])
    if (item['pair']==(a,b)):
        match=item['matches']
        res = ransac_filter(key_points[a][match[:, 0]], key_points[b][match[:, 1]], K)
        match01 = match[res[2].ravel() == 1]
        # res = ransac_filter(key_points[a][match[:, 0]], key_points[b][match[:, 1]], K)
        # match01 = match[res[2].ravel() == 1]
    if (item['pair']==(b,c)):
        match=item['matches']
        res = ransac_filter(key_points[b][match[:, 0]], key_points[c][match[:, 1]], K)
        match12 = match[res[2].ravel() == 1]
        # res = ransac_filter(key_points[b][match[:, 0]], key_points[c][match[:, 1]], K)
        # match12 = match[res[2].ravel() == 1]
    if (item['pair']==(a,c)):
        match=item['matches']
        res = ransac_filter(key_points[a][match[:, 0]], key_points[c][match[:, 1]], K)
        match02 = match[res[2].ravel() == 1]
        # res = ransac_filter(key_points[a][match[:, 0]], key_points[c][match[:, 1]], K)
        # match02 = match[res[2].ravel() == 1]

points01 = np.concatenate([key_points[a][match01[:,0]], key_points[b][match01[:,1]]],-1)
points12 = np.concatenate([key_points[b][match12[:,0]], key_points[c][match12[:,1]]],-1)
points02 = np.concatenate([key_points[a][match02[:,0]], key_points[c][match02[:,1]]],-1)


with open('./01.txt', 'w') as file:
    for item in points01:
        # if abs(item[0]-item[2])>10 or abs(item[1]-item[3])>20:
        #     continue
        file.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n')

with open('./12.txt', 'w') as file:
    for item in points12:
        # if abs(item[0]-item[2])>10 or abs(item[1]-item[3])>20:
        #     continue
        file.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n')

with open('./02.txt', 'w') as file:
    for item in points02:
        # if abs(item[0]-item[2])>10 or abs(item[1]-item[3])>20:
        #     continue
        file.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n')



draw_matches(cv2.imread('simu3/images/frame000.jpg'), cv2.imread('simu3/images/frame010.jpg'), key_points[a][match01[:,0]], key_points[b][match01[:,1]])
draw_matches(cv2.imread('simu3/images/frame010.jpg'), cv2.imread('simu3/images/frame020.jpg'), key_points[b][match12[:,0]], key_points[c][match12[:,1]])
draw_matches(cv2.imread('simu3/images/frame000.jpg'), cv2.imread('simu3/images/frame020.jpg'), key_points[a][match02[:,0]], key_points[c][match02[:,1]])

# import cv2
# import numpy as np
# from plyfile import PlyData, PlyElement
#
# fx = fy = 4000
# cx = cy = 180
#
# K = np.array([[fx,  0, cx],
#               [ 0, fy, cy],
#               [ 0,  0,  1]])
#
#
# # 假设你有以下两个图像中的匹配点
# # pts1 和 pts2 是 (N, 2) 的 numpy 数组，分别表示图像1和图像2中的N个匹配点
# # pts1 = np.array([[x1_1, y1_1], [x1_2, y1_2], ...])
# # pts2 = np.array([[x2_1, y2_1], [x2_2, y2_2], ...])
# pts1 = points01[:,:2]
# pts2 = points01[:,2:]
# # 使用内参矩阵校正匹配点坐标（去畸变）
# pts1_normalized = cv2.undistortPoints(pts1.astype(np.float32).reshape(-1, 1, 2), K, None)
# pts2_normalized = cv2.undistortPoints(pts2.astype(np.float32).reshape(-1, 1, 2), K, None)
#
# # 计算基础矩阵 F 或本质矩阵 E
# E, mask = cv2.findEssentialMat(pts1_normalized, pts2_normalized, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#
# # 从本质矩阵 E 中恢复相机姿态（R 和 t）
# _, R, t, mask = cv2.recoverPose(E, pts1_normalized, pts2_normalized, K)
#
# # 构造投影矩阵
# # 假设第一台相机位于世界坐标系的原点，投影矩阵为 [I | 0]
# proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
#
# # 第二台相机的投影矩阵为 [R | t]
# proj_matrix2 = np.hstack((R, t))
#
# # 使用投影矩阵和匹配点进行三角化
# pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1_normalized, pts2_normalized)
#
# # 将三维点从齐次坐标转换为非齐次坐标
# pts3D = pts4D_hom[:3] / pts4D_hom[3]
# pts3D = pts3D.T  # 转置以得到 (N, 3) 的形状
#
# # 输出三维点坐标
# print("三维点坐标:\n", pts3D)
#
#
# camera_positions = np.array([[0.0, 0.0, 0.0],
#                              t.T[0]])
# pts3D = np.concatenate([pts3D, camera_positions], axis=0)
# # 定义三维点的 PLY 格式
# vertex = np.array([tuple(point) for point in pts3D],
#                   dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#
# # 相机位置的 PLY 格式，添加颜色信息
# camera_vertices = np.array([(cam_pos[0], cam_pos[1], cam_pos[2], 255, 0, 0) for cam_pos in camera_positions],
#                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
#
# # 创建 PLY 元素
# vertex_element = PlyElement.describe(vertex, 'vertex')
# camera_element = PlyElement.describe(camera_vertices, 'camera')  # 使用 'vertex' 以兼容点格式
#
# # 将点和相机位置写入 PLY 文件
# ply_data = PlyData([vertex_element, camera_element])
# ply_data.write('output.ply')
#
# print("3D points and camera positions saved to 'output.ply'")