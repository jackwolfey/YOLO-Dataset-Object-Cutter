# -*- coding: utf-8 -*-
# Author: Wei Jia
# Project: yolov5
# File: yolo_object_cutter.py
"""
同时支持Pillow和OpenCV进行裁剪操作(优先Pillow)
适用于对于以此目录结构存储的数据集(yolov5官方的coco128数据集目录结构):
image_dataset
    - images
        - no1.jpg
        - no2.jpg
        ...
    - labels
        - no1.txt
        - no2.txt
        ...

使用此程序可以将数据集中每张图片标注的内容对应的小图裁剪下来并保存到目标文件夹
目标文件夹的目录结构为

dst_dir
    - no1
        - no1_0_x1_y1_x2_y2.jpg
        - no1_1_x1_y1_x2_y2.jpg
        ...
    - no2
        - no2_0_x1_y1_x2_y2.jpg
        ...
    ...

(PS:小图命名规则为{文件名_类别索引_(目标框)左上角x坐标_左上角y坐标_右下角x坐标_右下角y坐标.图像扩展名})
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

NO_PILLOW = False
try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:
    NO_PILLOW = True
    import cv2


def split_list(lst, n):
    if n <= 0:
        raise ValueError("The number of splits must be greater than 0")
    if n > len(lst):
        raise ValueError("The number of splits must be less than or equal to the length of the list")

    # Calculate the size of each chunk
    chunk_size = len(lst) // n
    remainder = len(lst) % n

    # Create the chunks
    chunks = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end

    return chunks


class YoloDatasetObjectCutter:
    def __init__(self, dataset_path, save_path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            raise ValueError(f"{dataset_path} is not exist or not a dir.")
        self._check_dataset()
        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=False)
        print(f"数据集路径:{self.dataset_path.absolute()}\n存储路径:{self.save_path.absolute()}")
        self.dataset_meta = self._get_dataset_meta()

    def _check_dataset(self):
        if not self.dataset_path.joinpath('images').exists():
            raise ValueError(f"{self.dataset_path.joinpath('images')} is not exist.")
        if not self.dataset_path.joinpath('labels').exists():
            raise ValueError(f"{self.dataset_path.joinpath('labels')} is not exist.")

    def _get_dataset_meta(self):
        meta = {}
        image_files = list(self.dataset_path.joinpath('images').iterdir())
        label_files_stem = [i.stem for i in self.dataset_path.joinpath('labels').iterdir() if i.suffix == '.txt']
        for i in image_files:
            if i.stem in label_files_stem:
                meta[i.stem] = (i, self.dataset_path.joinpath('labels', f'{i.stem}.txt'))
        return meta

    def save_result(self, image: Path, label: Path):
        cuts = self.cut_single(image, label)
        if cuts:
            save_cuts_path = self.save_path.joinpath(image.stem)
            save_cuts_path.mkdir(exist_ok=True)

            for c in cuts:
                file_name = f"{image.stem}_{'_'.join(list(map(lambda x: str(x), c[:5])))}{image.suffix}"
                if not NO_PILLOW:
                    c[-1].save(save_cuts_path.joinpath(file_name))
                else:
                    cv2.imwrite(str(save_cuts_path.joinpath(file_name)), c[-1])

    def save_result_batch(self, batch: list[tuple[Path, Path]], log):
        total = len(batch)
        processed = 0
        for i in batch:
            self.save_result(i[0], i[1])
            processed += 1
            if log:
                print(f"\r总共: {total}个, 已处理: {processed}个, "
                      f"已完成 {round((processed / total) * 100, 2)}%", end='')

    def run(self, workers: int = 1):
        if workers == 1:
            self.save_result_batch(list(self.dataset_meta.values()), log=True)
        elif workers > 1:
            chunks = split_list(list(self.dataset_meta.values()), workers)
            print(f"总共{workers}个进程正在进行处理，请稍后...")

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.save_result_batch, chuk, log=False) for chuk in chunks}
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    print(f"总共{workers}个进程处理, 进程:{id(future)}完成, "
                          f"已完成{completed}个进程")
        else:
            raise ValueError('invalid workers.')

    @classmethod
    def cut_single(cls, image: Path, label: Path):
        if not NO_PILLOW:
            image = Image.open(image)
            image = ImageOps.exif_transpose(image)
            imw, imh = image.size
        else:
            image = cv2.imread(str(image))
            imh, imw = image.shape[:2]
        with label.open('r', encoding='utf-8') as f:
            label = f.read().strip().splitlines()
        cuts = []
        for line in label:
            boxes = line.strip().split(' ')
            cls_idx = int(boxes[0])
            centerx, centery, w, h = map(lambda x: float(x), boxes[1:])
            centerx, w = centerx * imw, w * imw
            centery, h = centery * imh, h * imh
            x1, y1 = round(centerx - (w / 2)), round(centery - (h / 2))
            x2, y2 = round(centerx + (w / 2)), round(centery + (h / 2))
            if not NO_PILLOW:
                cut = image.crop((x1, y1, x2, y2))
            else:
                cut = image[y1:y2, x1:x2]
            cuts.append((cls_idx, x1, y1, x2, y2, cut))

        return cuts


def main(_args):
    cutter = YoloDatasetObjectCutter(_args.dataset_path, _args.save_path)
    cutter.run(workers=_args.workers)
    print("处理完毕")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo dataset object cutter')
    parser.add_argument('-src', '--dataset-path', type=str, help='dataset path')
    parser.add_argument('-dst', '--save-path', type=str, help='cut image save path')
    parser.add_argument('-w', '--workers', default=1, type=int, help='multiprocess workers')
    args = parser.parse_args()
    main(args)
