from argparse import ArgumentParser
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
import open3d as o3d
import numpy as np
import os


def read_pcd(filepath):
    lidar = []
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)


# def convert(pcdfolder, binfolder):
#     current_path = os.getcwd()
#     ori_path = os.path.join(current_path, pcdfolder)
#     file_list = os.listdir(ori_path)
#     des_path = os.path.join(current_path, binfolder)
#     if os.path.exists(des_path):
#         pass
#     else:
#         os.makedirs(des_path)
#     for file in file_list:
#         (filename, extension) = os.path.splitext(file)
#         velodyne_file = os.path.join(ori_path, filename) + '.pcd'
#         pl = read_pcd(velodyne_file)
#         pl = pl.reshape(-1, 4).astype(np.float32)
#         velodyne_file_new = os.path.join(des_path, filename) + '.bin'
#         pl.tofile(velodyne_file_new)


def pcd2bin(pcd_file):
    # pcd = o3d.io.read_point_cloud(pcd_file, format='pcd')
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.array(pcd.points)
    # points = []
    # points_start = 0
    # with open(pcd_file, 'r') as f:
    #     line = f.readline().strip()
    #     while line:
    #         linestr = line.split(" ")
    #         if linestr[0] == 'DATA':
    #             data_format = linestr[1]
    #             points_start = 1
    #         if points_start == 1 and len(linestr) == 4:
    #             if data_format == 'ascii':
    #                 linestr_convert = list(map(float, linestr))
    #                 points.append(linestr_convert)
    #             elif data_format == 'binary':
    #
    #         line = f.readline().strip()


    pass


def bin2pcd(bin_file, pcd_file='', load_dim=4, use_dim=4):
    LoadPoints = LoadPointsFromFile(coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim)
    data = dict(pts_filename=bin_file)
    points = LoadPoints(data)['points']
    pts = points.tensor

    # 写文件句柄
    if pcd_file is None:
        handle = open(bin_file.replace('.bin', '.pcd'), 'a')
    else:
        handle = open(pcd_file, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    if load_dim == 4:
        handle.write(
            '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1')
    elif load_dim == 3:
        handle.write(
            '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')

    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')
    # handle.write('\nDATA binary')

    # 依次写入点
    for i in range(point_num):
        if load_dim == 4:
            string = '\n%f %f %f %f' % (pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3])
            handle.write(string.encode('bytes'))
            # handle.write('\n%b %b %b %b' % (points[i, 0], points[i, 1], points[i, 2], points[i, 3]))
        elif load_dim==3:
            handle.write('\n%f %f %f' % (pts[i, 0], pts[i, 1], pts[i, 2]))
            # handle.write('\n%b %b %b' % (points[i, 0], points[i, 1], points[i, 2]))
    handle.close()


def bin2pcd_folder(bin_folder, pcd_folder):
    for i in os.listdir(bin_folder):
        bin_file = os.path.join(bin_folder, i)
        pcd_file = os.path.join(pcd_folder, i.replace('.bin', '.pcd'))
        bin2pcd(bin_file, pcd_file)

def main():
    parser = ArgumentParser()
    parser.add_argument('pts_file', help='Point cloud file')
    parser.add_argument('--mode', type=int, default=0, help='0:bin2pcd, 1:pcd2bin, 2:bin2pcd_folder')
    args = parser.parse_args()
    if args.mode == 0:
        bin2pcd(args.pts_file)
    elif args.mode == 1:
        pcd2bin(args.pts_file)
    elif args.mode == 2:
        bins_folder = args.pts_file
        pcds_folder = bins_folder.replace('bins', 'pcds')
        bin2pcd_folder(bins_folder, pcds_folder)


if __name__ == '__main__':
    main()