from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils import viz_3d

images = Path('datasets/simu1')
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
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()
