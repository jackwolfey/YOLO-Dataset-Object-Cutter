# YOLO-Dataset-Object-Cutter

### YOLO目标检测数据集目标小图裁剪工具

### A tool to cut YOLO series dataset's object crop image

#### 使用方法

#### Usage

```shell
python yolo_object_cutter.py --dataset-path ./your_dataset_path --save-path ./saved_dir
```

or

```shell
python yolo_object_cutter.py -src ./your_dataset_path -dst ./saved_dir -w 4
```

支持多进程并行加速处理，进程数使用--workers(-w)控制

Support multiprocess, by passing --workers(-w)

#### 同时支持Pillow和OpenCV进行裁剪操作(优先Pillow)
#### Support both Pillow and OpenCV for cropping operations (prioritize Pillow)

适用于对于以此目录结构存储的数据集(yolov5官方的coco128数据集目录结构):

Applicable to datasets stored in this directory structure (YOLOv5 official coco128 dataset directory structure):
```
image_dataset
    - images
        - no1.jpg
        - no2.jpg
        ...
    - labels
        - no1.txt
        - no2.txt
        ...
```

使用此程序可以将数据集中每张图片标注的内容对应的小图裁剪下来并保存到目标文件夹
目标文件夹的目录结构为

This program allows you to crop the corresponding small images of each annotated image in the dataset and save them to the target folder
The directory structure of the target folder is

```
dst_dir
    - no1
      - no1_0_x1_y1_x2_y2.jpg
      - no1_1_x1_y1_x2_y2.jpg
      ...
    - no2
      - no2_0_x1_y1_x2_y2.jpg
      ...
      ...
```

(PS:小图命名规则为{文件名_类别索引_(目标框)左上角x坐标_左上角y坐标_右下角x坐标_右下角y坐标.图像扩展名})

(PS: The naming convention for small images is {file name_category index_(target box)top left corner x-coordinate_top left corner y-coordinate_bottom right corner x-coordinate_bottom right corner y-coordinate.image_extension})
