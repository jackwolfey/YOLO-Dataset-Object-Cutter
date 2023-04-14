"""
使用场景：
img_path文件夹内，存在*.jpg和*.txt文件，每张图片和label的文件名一一对应
调用run_batch即可生成裁剪后的图片到目标文件夹
"""

import cv2
import os


class ImgCutter():
    def __init__(self, img_path, save_path):
        self.img_path = img_path
        self.save_path = save_path
        self.img_li, self.label_li = self.get_img_and_label_list()

    def get_img_and_label_list(self):
        li = os.listdir(self.img_path)
        label_li = [self.img_path + i for i in li if i.endswith('.txt') and not i.startswith('class')]
        img_li = []
        for i in li:
            for j in label_li:
                if i[:-4] == j.split('/')[-1][:-4] and i.endswith('.jpg'):
                    img_li.append(img_path + i)
        img_li.sort()
        label_li.sort()
        return img_li, label_li

    # using yolo label
    def cut_img(self, image_path, label_path, count):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = img.shape
        h, w = size[0], size[1]
        print(h, w)
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.readline().strip()
        content = content.split(' ')[1:]
        content[0] = int(float(content[0]) * w)
        content[1] = int(float(content[1]) * h)
        content[2] = int(float(content[2]) * w)
        content[3] = int(float(content[3]) * h)
        x, y, w, h = content[0], content[1], content[2], content[3]
        x1, y1 = int(x - (w / 2)), int(y - (h / 2))
        x2, y2 = int(x + (w / 2)), int(y + (h / 2))
        print([x1, y1], [x2, y2])
        img = img[y1:y2, x1:x2]
        # cv2.imshow('1', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(self.save_path + str(count) + '.jpg', img)

    def run_batch(self):
        for idx in range(len(self.img_li)):
            self.cut_img(self.img_li[idx], self.label_li[idx], idx)


if __name__ == '__main__':
    img_path = './pics_rotated/'
    save_path = './bbox_pics_rotated/'
    cutter = ImgCutter(img_path, save_path)
    cutter.run_batch()
