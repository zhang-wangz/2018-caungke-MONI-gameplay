# -*- coding: utf-8 -*-
# import 进openCV的库
import cv2
import sys
import dlib
import cv2
import os
import glob
import numpy as np
import os.path
from PIL import Image
# img_file:图片的路径
# path_save:保存路径
# width：宽度
# height：长度

def img_resize(img_file, path_save, width=16,height=16):
    img = Image.open(img_file)
    new_image = img.resize((width,height),Image.BILINEAR)
    new_image.save(os.path.join(path_save,os.path.basename(img_file)))
if __name__ == '__main__':
    for jpgfile in glob.glob("/home/wz/Desktop/xunlian/*.bmp"):
        img_resize(jpgfile,"/home/wz/Desktop/xunlian/",20,20)
