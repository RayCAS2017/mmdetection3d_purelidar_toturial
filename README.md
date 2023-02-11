<div align="center">
     <b><font size="10">利用MMDetection3D训练纯点云数据集</font></b> 
</div>

### 引言
利用MMDet3D训练一个纯点云数据的一种方法是先将标注文件转成kitti格式，再基于MMDet3D的KittyDataset类进行训练。该方法虽然简单，不需要对MMDet3D进行任何的增改，但kitti数据的3D框坐标系是相机坐标系，不是lidar坐标系，因此还需要额外构建lidar和相机的内外参矩阵，将3D点云标注工具标注的3D框转到相机坐标系，整个流程显得特别的冗余。该repo利用lidar标注文件，不需要转化为kitti格式，即可对纯点云数据进行训练。

### 点云标注工具
lidar点云标注工具采用的是[SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS.git)

![SUSTechPOINTS_UI](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/main-ui.png)

其标签格式如下

![SUSTechPOINTS_lable](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/lidar_label.jpg)


### 步骤：

#### 1、数据集准备


#### 2、生成中间文件
添加的脚本为：
```
./tools/create_data_custom.py
./tools/create_gt_database_custom.py
```
生成的文件有：
```
train_annotaion.pkl
val_annotation.pkl
dbinfos_train.pkl
gt_database
```
其中， train_annotaion.pkl和val_annotation.pkl记录的信息格式为:

![pickl_info](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/plk_info.jpg)

dbinfos_train.pkl和gt_database是截取的gt 3d框信息和数据，用于数据增强。


#### 3、构建PureLidarDataset
写一个继承CustomDataset的数据集PureLidarDataset，主要是修改了CustomDataset中的评估方法

1）添加
```
./mmdet3d/datasets/purelidar_dataset.py
```
需要在./mmdet3d/datasets/__init__.py中申明可见
![dataset_init_](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/dataset_init.png)

2)添加
```
./mmdet3d/core/evaluation/purelidar_eval.py
```
需要在/mmdet3d/core/evaluation/__init__.py中申明可见
![eval_init_](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/eval_init.jpg)

#### 4、配置文件
1）configs/_base_/datasets/minikitti-3d-3class_custom.py

![config_base](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/config_base.jpg)

2)configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_minikitti-3d-3class_custom.py

![config_model](https://github.com/RayCAS2017/mmdetection3d_purelidar_toturial/raw/main/assets/config_model.jpg)

#### 5、训练
```
./tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_minikitti-3d-3class_custom.py --work-dir outputs/pointpillars_minikitti_custom_debug --gpu-id 0
```



