import numpy as np
import pickle
import cv2


with open('data_css/matches.pkl', 'rb') as file:
    matches = pickle.load(file)

with open('data_css/keypoints.pkl', 'rb') as file:
    keypoints = pickle.load(file)


i1 = np.genfromtxt('data_css/i1.csv', delimiter=',') - 1
i2 = np.genfromtxt('data_css/i2.csv', delimiter=',') - 1


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

print(match01[i1.astype(int)])
print(match12[i2.astype(int)])
l, r = match01[i1.astype(int)][:,0], match12[i2.astype(int)][:,1]
points_2d = []
lr = np.concatenate([l[:,None],r[:,None]],-1)
for it in lr:
    if it in match02:
        points_2d.append(it)

print(len(points_2d))

points_3d = np.genfromtxt('data_css/points.csv', delimiter=',')
inlines = np.genfromtxt('data_css/inliners.csv', delimiter=',', dtype=np.int32) - 1

relations = dict()
to3d = dict()

p2d = []

ct = 0
cct = 0
points = []
for item1 in match01:
    for item2 in match12:
        if item1[1]==item2[0]:
            ct = ct+1
            print('?', keypoints[0][item1[0]], keypoints[1][item1[1]], keypoints[2][item2[1]])
            if [item1[0], item2[1]] in match02.tolist():
                p2d.append(keypoints[2][item2[1]])
                points.append([keypoints[a][item1[0]],keypoints[b][item1[1]],keypoints[c][item2[1]]])
                # relations[cct] = [{0 : item1[0]}, {1 : item1[1]}, {2 : item2[1]}]
                id = inlines[cct]
                relations[(id, 0)] = item1[0]
                relations[(id, 1)] = item1[1]
                relations[(id, 2)] = item2[1]
                to3d[(0, item1[0])] = id
                to3d[(1, item1[1])] = id
                to3d[(2, item2[1])] = id
                cct = cct + 1

match23=None
for item in matches:
    if (item['pair']==(2,3)):
        match23=item['matches']

# p_3d = []
# p_2d = []
# K = [[4000, 0, 180], [0, 4000, 180], [0, 0, 1]]
# if match23 is not None:
#     for item in match23:
#         p2_id = item[0]
#         p3_id = item[1]
#
#         for keys in to3d:
#             if keys[0] == 2 and keys[1] == p2_id:
#                 p_3d.append(points_3d[to3d[keys]])
#                 p_2d.append(keypoints[3][p3_id])
#
# p_3d = np.array(p_3d, dtype=np.float32)
# p_2d = np.array(p_2d, dtype=np.float32)
#
# retval, rvec, tvec = cv2.solvePnP(p_3d, p_2d, np.array(K), None)
# R = cv2.Rodrigues(rvec)[0]
# T = tvec
#
# pose = np.concatenate([R, T], axis=1)
# np.save('data/pose3.npy', pose)
# print(pose.shape)
# print(retval, rvec, tvec, R, T)
# print(R.shape, T.shape)


p3d = []

K = [[4000, 0, 180], [0, 4000, 180], [0, 0, 1]]
# for keys in to3d:
#     if keys[0]==2:
#         # p2d.append(keypoints[2][keys[1]])
#         p3d.append(points_3d[to3d[keys]])
p3d = points_3d[inlines]

p_3d = np.array(p3d, dtype=np.float32)
p_2d = np.array(p2d, dtype=np.float32)

retval, rvec, tvec = cv2.solvePnP(p_3d, p_2d, np.array(K), None)
R = cv2.Rodrigues(rvec)[0]
T = tvec

pose = np.concatenate([R, T], axis=1)
np.save('data_css/pose3.npy', pose)
print('p_3d', p_3d)

print(pose.shape)
print(R, T)
print(R.shape, T.shape)

np.save('p3d.npy', p_3d)


corresp = np.genfromtxt('data_css/corresp.csv', delimiter=',')

p2d_ = corresp[:,inlines][4:6]
p2d_ = np.transpose(p2d_)
print('p2d_', np.concatenate([p2d, p2d_], -1))

