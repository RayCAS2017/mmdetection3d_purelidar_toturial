from argparse import ArgumentParser
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
import torch


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


# 传入点云对象
def _points2pcd(points, PCD_FILE_PATH):

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    # handle.write(
    #     '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        handle.write('\n%f %f %f %f' % (points[i, 0], points[i, 1], points[i, 2], points[i, 3]))
    handle.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument(
        '--angle', type=float, default=0.0, help='the rotaed angle')
    parser.add_argument(
        '--axis', type=int, default=0, help='The axis to be rotated')

    args = parser.parse_args()

    LoadPoints = LoadPointsFromFile(coord_type='LIDAR', load_dim=4, use_dim=4)
    data = dict(pts_filename=args.pcd)

    result = LoadPoints(data)
    points = result['points']
    points_tensor = points.tensor[:,:3]
    points1 = points_tensor[(points_tensor[:,0] > 0.1) &
        (points_tensor[:,1] < 0.1) & (points_tensor[:,2] < 0.1) &
        (points_tensor[:,1] > -0.1) & (points_tensor[:,2] > -0.1)
    ]
    points1_mean = torch.mean(points1,0)

    points2 = points_tensor[(points_tensor[:,0] > points1_mean[0]-1.2) & (points_tensor[:,0] < points1_mean[0]-1) &
        (points_tensor[:,1] < 0.5) &(points_tensor[:,1] > -0.5)
    ]

    points2_mean = torch.mean(points2, 0)

    angle = torch.atan(points2_mean[2]/(points1_mean[0]-points2_mean[0]))

    points.rotate(rotation=-angle, axis=args.axis)


    #
    if points is not None:
        # _write_obj(points.tensor, args.pcd.replace('.bin', '_rotated.obj'))
        _points2pcd(points.tensor, args.pcd.replace('.bin', '_rotated.pcd'))
    pass


if __name__ == '__main__':
    main()