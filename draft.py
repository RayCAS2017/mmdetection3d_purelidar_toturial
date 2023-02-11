import mmcv
import open3d as o3d
import numpy as np
import os

# pkl = './toy_kitti/kitti_dbinfos_train.pkl'
#
# info1 = mmcv.load(pkl, file_format='pkl')
#
# pkl2 = './toy_kitti/dbinfos_train.pkl'
# info2 = mmcv.load(pkl2, file_format='pkl')
#
# pkl3 = './toy_kitti/kitti_infos_train.pkl'
# info3 = mmcv.load(pkl3, file_format='pkl')
#
# pkl4 = './data/minikitti/kitti_infos_val.pkl'
# info4 = mmcv.load(pkl4, file_format='pkl')
#
# pkl5 = './data/minikitti_sustech/train_annotation.pkl'
# info5 = mmcv.load(pkl5, file_format='pkl')

pcd_file = './demo/data/ouster/206.pcd'
pcd = o3d.io.read_point_cloud(pcd_file)
points = np.array(pcd.points)
points.tofile(pcd_file.replace('.pcd', '.bin'))
pass
