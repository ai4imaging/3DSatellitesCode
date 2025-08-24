import numpy as np
import cv2
import pickle


def compute_c2w_matrices(R_list, T_list):
    """
    Compute camera-to-world (C2W) transformation matrices from lists of rotation matrices and translation vectors.

    Parameters:
    R_list (list of numpy.ndarray): List of 3x3 rotation matrices.
    T_list (list of numpy.ndarray): List of 3x1 translation vectors.

    Returns:
    list of numpy.ndarray: List of 4x4 camera-to-world transformation matrices.
    """
    c2w_list = []

    for R, T in zip(R_list, T_list):
        # Create a 4x4 identity matrix
        c2w = np.eye(4)

        # Set the top-left 3x3 part to be the transpose of R
        c2w[:3, :3] = R.T

        # Set the top-right 3x1 part to be the negative transpose of R times T, flattened to 1D array
        c2w[:3, 3] = (-R.T @ T).flatten()

        # Append the computed C2W matrix to the list
        c2w_list.append(c2w)

    return c2w_list


with open('matches.pkl', 'rb') as file:
    matches = pickle.load(file)
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)

n = len(features)

K = np.array([
    [4410, 0, 360],
    [0, 4410, 360],
    [0, 0, 1]
], dtype=np.float32)

for p in matches:
    print(p['pair'])

pts_pair = dict()
essential_matrices = np.zeros((n, n, 3, 3))
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        matches_ij = None
        for pair in matches:
            if pair['pair'] == (i, j):
                matches_ij = pair['matches']
                break
        if matches_ij is None:
            continue
        features_i = features[i]['keypoints']
        features_j = features[j]['keypoints']

        pts_i = features_i[matches_ij[:, 0]].astype(np.float32)
        pts_j = features_j[matches_ij[:, 1]].astype(np.float32)
        pts_pair[(i, j)] = (pts_i, pts_j)
        print('start', pts_i.shape, pts_j.shape)
        if len(pts_i) <= 5:
            continue
        E, mask = cv2.findEssentialMat(pts_i, pts_j, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        print('end', E.shape, mask.shape)
        essential_matrices[i, j] = E

R = [np.eye(3)]
t = [np.zeros((3, 1))]

for i in range(1, n):
    E = essential_matrices[0][i]  # 获取第 0 对图像和第 i 对图像的本质矩阵
    if (0, i) in pts_pair:
        points1, points2 = pts_pair[(0, i)]  # 获取匹配点对
    else:
        continue

    # 从本质矩阵中恢复相对位姿
    _, R_rel, t_rel, _ = cv2.recoverPose(E, points1, points2, K)

    # 累积相对位姿以获得绝对位姿
    R_abs = R[-1] @ R_rel
    t_abs = t[-1] + (R[-1] @ t_rel)

    R.append(R_abs)
    t.append(t_abs)

for i in range(len(R)):
    print(f"Camera {i} pose:")
    print("Rotation:\n", R[i])
    print("Translation:\n", t[i])

c2w_list = compute_c2w_matrices(R, t)
np.save('c2ws.npy', np.array(c2w_list))
print(len(c2w_list))