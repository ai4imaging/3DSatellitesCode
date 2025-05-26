import time

import numpy as np
import pickle
import cv2
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

def main():
    sfm = SFM()
    for i in range(2, 13):
        sfm.add_camera(i, i+1)
    sfm.visualize()


if __name__ == '__main__':
    main()