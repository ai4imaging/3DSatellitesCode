import pickle

import numpy as np

with open('matches.pkl', 'rb') as file:
    matches = pickle.load(file)

with open('keypoints.pkl', 'rb') as file:
    keypoints = pickle.load(file)

print(matches)
print(keypoints)

match01=None
match12=None
match02=None

a,b,c = 0,1,2

for item in matches:
    if (item['pair']==(a,b)):
        match01=item['matches']
    if (item['pair']==(b,c)):
        match12=item['matches']
    if (item['pair']==(a,c)):
        match02=item['matches']

points01 = np.concatenate([keypoints[a][match01[:,0]], keypoints[b][match01[:,1]]],-1)
points12 = np.concatenate([keypoints[b][match12[:,0]], keypoints[c][match12[:,1]]],-1)
points02 = np.concatenate([keypoints[a][match02[:,0]], keypoints[c][match02[:,1]]],-1)


with open('01.txt', 'w') as file:
    for item in points01:
        file.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n')

with open('12.txt', 'w') as file:
    for item in points12:
        file.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n')

with open('02.txt', 'w') as file:
    for item in points02:
        file.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n')



import cv2
import numpy as np
from plyfile import PlyData, PlyElement

fx = fy = 4000
cx = cy = 180

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])


# 假设你有以下两个图像中的匹配点
# pts1 和 pts2 是 (N, 2) 的 numpy 数组，分别表示图像1和图像2中的N个匹配点
# pts1 = np.array([[x1_1, y1_1], [x1_2, y1_2], ...])
# pts2 = np.array([[x2_1, y2_1], [x2_2, y2_2], ...])
pts1 = points01[:,:2]
pts2 = points01[:,2:]
# 使用内参矩阵校正匹配点坐标（去畸变）
pts1_normalized = cv2.undistortPoints(pts1.astype(np.float32).reshape(-1, 1, 2), K, None)
pts2_normalized = cv2.undistortPoints(pts2.astype(np.float32).reshape(-1, 1, 2), K, None)

# 计算基础矩阵 F 或本质矩阵 E
E, mask = cv2.findEssentialMat(pts1_normalized, pts2_normalized, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# 从本质矩阵 E 中恢复相机姿态（R 和 t）
_, R, t, mask = cv2.recoverPose(E, pts1_normalized, pts2_normalized, K)

# 构造投影矩阵
# 假设第一台相机位于世界坐标系的原点，投影矩阵为 [I | 0]
proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))

# 第二台相机的投影矩阵为 [R | t]
proj_matrix2 = np.hstack((R, t))

# 使用投影矩阵和匹配点进行三角化
pts4D_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1_normalized, pts2_normalized)

# 将三维点从齐次坐标转换为非齐次坐标
pts3D = pts4D_hom[:3] / pts4D_hom[3]
pts3D = pts3D.T  # 转置以得到 (N, 3) 的形状

# 输出三维点坐标
print("三维点坐标:\n", pts3D)


camera_positions = np.array([[0.0, 0.0, 0.0],
                             t.T[0]])
pts3D = np.concatenate([pts3D, camera_positions], axis=0)
# 定义三维点的 PLY 格式
vertex = np.array([tuple(point) for point in pts3D],
                  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# 相机位置的 PLY 格式，添加颜色信息
camera_vertices = np.array([(cam_pos[0], cam_pos[1], cam_pos[2], 255, 0, 0) for cam_pos in camera_positions],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

# 创建 PLY 元素
vertex_element = PlyElement.describe(vertex, 'vertex')
camera_element = PlyElement.describe(camera_vertices, 'camera')  # 使用 'vertex' 以兼容点格式

# 将点和相机位置写入 PLY 文件
ply_data = PlyData([vertex_element, camera_element])
ply_data.write('output.ply')

print("3D points and camera positions saved to 'output.ply'")