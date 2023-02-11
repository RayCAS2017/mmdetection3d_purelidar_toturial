import argparse
from pathlib import Path
from concurrent import futures as futures
import os
import numpy as np
import json
import pickle
from tools.create_gt_database_custom import create_groundtruth_database_custom


def _read_sampleset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def _get_lidar_path(samples, idx, lidar_path=None):
    if lidar_path is not None:
        return os.path.join(lidar_path, samples[idx] + '.bin')
    else:
        return os.path.join('bins', samples[idx] + '.bin')


def _get_label_info_sustech(root_path, samples, idx, label_path=None):
    if label_path is not None:
        label_rel_path = os.path.join(label_path, samples[idx] + '.json')
    else:
        label_rel_path = os.path.join('labels', samples[idx] + '.json')

    label_path = root_path / label_rel_path
    label_str = ""
    with open(label_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            label_str = label_str + line.replace(" ","").replace("\t","").strip()
    label_info = json.loads(label_str)
    num_objs = len(label_info)
    gt_bboxes = []
    gt_names = []
    for obj in label_info:
        bbox = [
            obj["psr"]["position"]["x"],obj["psr"]["position"]["y"], obj["psr"]["position"]["z"],
            obj["psr"]["scale"]["x"], obj["psr"]["scale"]["y"], obj["psr"]["scale"]["z"],
            obj["psr"]["rotation"]["z"]
        ]
        gt_bboxes.append(bbox)
        obj_name = obj["obj_type"]
        gt_names.append(obj_name)
    annos = {}
    annos['box_type_3d'] = 'LiDAR'
    annos['gt_bboxes_3d'] = np.array([i for i in gt_bboxes])
    annos['gt_names'] = gt_names
    annos['group_ids'] = np.arange(num_objs, dtype=np.int32)

    return annos


def _get_label_info_kitti(root_path, samples, idx, label_path=None):
    if label_path is not None:
        label_rel_path = os.path.join(label_path, samples[idx] + '.txt')
    else:
        label_rel_path = os.path.join('labels', samples[idx] + '.txt')

    label_path = root_path / label_rel_path
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_info(path,
             samples=[],
             sample_ids=[],
             num_worker=8,
             label_path=None,
             label_format=0,
             lidar_path=None
             ):

    root_path = Path(path)
    if not isinstance(sample_ids, list):
        sample_ids = list(range(sample_ids))

    if label_format == 0:
        _get_label_info = _get_label_info_sustech
    elif label_format == 1:
        _get_label_info = _get_label_info_kitti


    def map_func(idx):
        info = {}
        info['sample_idx'] = idx
        info['lidar_points'] = {'lidar_path': _get_lidar_path(samples, idx, lidar_path=lidar_path)}
        info['annos'] = _get_label_info(root_path, samples, idx, label_path=label_path)

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        lable_infos = executor.map(map_func, sample_ids)

    return list(lable_infos)


def create_custom_info_file(data_path, save_path=None, relative_path=True, class_names=None, args=None):
    sampleset_folder = Path(data_path) / args.sample_sets
    train_samples = _read_sampleset_file(str(sampleset_folder / 'train.txt'))
    val_samples = _read_sampleset_file(str(sampleset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    infos_train = get_info(
        data_path,
        samples=train_samples,
        sample_ids=len(train_samples),
        label_path=args.label_path,
        lidar_path=args.lidar_path,
        label_format= args
    )
    infos_val = get_info(
        data_path,
        samples=val_samples,
        sample_ids=len(val_samples),
        label_path=args.label_path,
        lidar_path=args.lidar_path
    )

    infos_train_save_path = save_path / 'train_annotation.pkl'
    infos_val_save_path = save_path / 'val_annotation.pkl'

    if not os.path.exists(infos_train_save_path):
        with open(infos_train_save_path, 'wb') as f:
            pickle.dump(infos_train, f)

    if not os.path.exists(infos_val_save_path):
        with open(infos_val_save_path, 'wb') as f:
            pickle.dump(infos_val, f)

    create_groundtruth_database_custom(
        'Custom3DDataset',
        data_path,
        info_path=infos_train_save_path,
        relative_path=False,
        classes=class_names
    )


def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./toy_kitti',
        help='specify the root path of dataset')

    parser.add_argument(
        '--lidar-path',
        type=str,
        default='bins',
        help='the path of lidar bin files')

    parser.add_argument(
        '--label-path',
        type=str,
        default='labels',
        help='the path of label files')

    parser.add_argument(
        '--class-file',
        type=str,
        default='classnames.txt',
        help='the txt file of class_names')

    parser.add_argument(
        '--label-format',
        type=int,
        default=0,
        help='0: SUSTechPoint; 1: kitti')

    parser.add_argument(
        '--sample-sets',
        type=str,
        default='SampleSets',
        help='the sample sets')

    args = parser.parse_args()

    class_names_txt = os.path.join(args.root_path, args.class_file)
    class_names = []
    with open(class_names_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            class_names.append(line.strip())

    create_custom_info_file(args.root_path, class_names=class_names, args=args)


if __name__ == "__main__":
    main()
