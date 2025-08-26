from pathlib import Path
import pickle
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils import viz_3d
import cv2


images = Path('datasets/simulation1/data')
outputs = Path('outputs/sfm/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['NN-superpoint']
# matcher_conf = match_features.confs['superglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=3)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

print(sfm_dir, images, sfm_pairs, feature_path, match_path)
# opts = dict(camera_model='SIMPLE_RADIAL', camera_params=','.join(map(str, (4410, 180, 180, 0))))
# opts = dict(camera_model='SIMPLE_RADIAL', camera_params=','.join(map(str, (5863, 256, 256, 0))))
# model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, image_options=opts)
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
#


######################### create 01 02 12
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


with open('matches.pkl', 'rb') as file:
    matches = pickle.load(file)

with open('keypoints.pkl', 'rb') as file:
    key_points = pickle.load(file)

import numpy as np

K = np.array([[4122, 0, 180],
              [0, 4122, 180],
              [0, 0, 1]])


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




# fig = viz_3d.init_figure()
# viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
# fig.show()


