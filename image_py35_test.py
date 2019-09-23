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
###调用电脑摄像头检测人脸并截图
def comparePersonData(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
        # diff += (v1 - v2)**2
    for i in range(len(data1)):
        diff += (data1[i] - data2[i])**2
    diff = np.sqrt(diff)
    print (diff)
    if(diff < 0.6):
        print "It's the same person"
    else:
        print "It's not the same person"

def savePersonData(face_rec_class, face_descriptor):
    if face_rec_class.name == None or face_descriptor == None:
        return
    filePath = face_rec_class.dataPath + face_rec_class.name + '.npy'
    vectors = np.array([])
    for i, num in enumerate(face_descriptor):
        vectors = np.append(vectors, num)
        # print(num)
    print('Saving files to :'+filePath)
    np.save(filePath, vectors)
    return vectors

def loadPersonData(face_rec_class, personName):
    if personName == None:
        return
    filePath = face_rec_class.dataPath + personName + '.npy'
    vectors = np.load(filePath)
    print(vectors)
    return vectors

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    #告诉OpenCV使用人脸识别分类器
    # classfier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classfier = cv2.CascadeClassifier('xml.xml')

    #识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像

        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.01, minNeighbors = 3, minSize = (212, 212))
        if len(faceRects) > 0:          #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect

                #将当前帧保存为图片
                # img_name = "%s/%d.jpg" % (path_name, num+1)
                # print(img_name)
                # image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                # cv2.imwrite(img_name, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                #
                # num += 1
                # if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                #     break

                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d/1 %d' % (num,len(faceRects)),(x + 30, y + 30), font, 1, (255,0,255),4)

                #超过指定最大保存数量结束程序
        # if num > (catch_pic_num): break

        #显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    # 连续截100张图像，存进image文件夹中
    # CatchPICFromVideo("get face", 0, 2, "C:\\Users\\Administrator\\Desktop\\face_recognition\\faces")
    CatchPICFromVideo("get_face", 0, 100, "/home/wz/Desktop/chuangke/simple/")
    for jpgfile in glob.glob("/home/wz/Desktop/chuangke/simple/*.jpg"):
        img_resize(jpgfile,"/home/wz/Desktop/chuangke/simple/",96,96)









# class face_recognition(object):
#     def __init__(self):
#         self.current_path = "/home/wz/Desktop/chuangke/face_recognition" # 获取当前路径 print self.current_path
#         self.predictor_path = self.current_path + "/model/shape_predictor_68_face_landmarks.dat"
#         self.face_rec_model_path = self.current_path + "/model/dlib_face_recognition_resnet_model_v1.dat"
#         self.faces_folder_path = self.current_path + "/faces/"
#         self.dataPath = self.current_path + "/data/"
#         self.detector = dlib.get_frontal_face_detector()
#         self.shape_predictor = dlib.shape_predictor(self.predictor_path)
#         self.face_rec_model = dlib.face_recognition_model_v1(self.face_rec_model_path)
#
#         self.name = None
#         self.img_bgr = None
#         self.img_rgb = None
#         self.detector = dlib.get_frontal_face_detector()
#         self.shape_predictor = dlib.shape_predictor(self.predictor_path)
#         self.face_rec_model = dlib.face_recognition_model_v1(self.face_rec_model_path)
#
#     def inputPerson(self, name='people', img_path=None):
#         if img_path == None:
#             print 'No file!\n'
#             return
#
#         # img_name += self.faces_folder_path + img_name
#         self.name = name
#         self.img_bgr = cv2.imread(self.current_path+img_path)
#         # opencv的bgr格式图片转换成rgb格式
#         b, g, r = cv2.split(self.img_bgr)
#         self.img_rgb = cv2.merge([r, g, b])
#
#     def create128DVectorSpace(self):
#         dets = self.detector(self.img_rgb, 1)
#         print("Number of faces detected: {}".format(len(dets)))
#         for index, face in enumerate(dets):
#             print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
#
#             shape = self.shape_predictor(self.img_rgb, face)
#             face_descriptor = self.face_rec_model.compute_face_descriptor(self.img_rgb, shape)
#             # print(face_descriptor)
#             # for i, num in enumerate(face_descriptor):
#             #   print(num)
#             #   print(type(num))
#
#             return face_descriptor


# import face_rec as fc
# face_rec = fc.face_recognition()   # 创建对象
# face_rec.inputPerson(name='1', img_path='/faces/1.jpg')  # name中写第一个人名字，img_name为图片名字，注意要放在faces文件夹中
# vector = face_rec.create128DVectorSpace()  # 提取128维向量，是dlib.vector类的对象
# person_data1 = fc.savePersonData(face_rec, vector )   # 将提取出的数据保存到data文件夹，为便于操作返回numpy数组，内容还是一样的
#
# # 导入第二张图片，并提取特征向量
# face_rec.inputPerson(name='2', img_path='/faces/2.jpg')
# vector = face_rec.create128DVectorSpace()  # 提取128维向量，是dlib.vector类的对象
# person_data2 = fc.savePersonData(face_rec, vector )
#
# # 计算欧式距离，判断是否是同一个人
# fc.comparePersonData(person_data1, person_data2)
# os.system('pause')
