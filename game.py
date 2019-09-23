#coding: utf8
import cocos
from cocos.sprite import Sprite
from pyaudio import PyAudio, paInt16
import struct
from pyaudioplay import PPX
from block import Block
import sys
import dlib
import cv2
import os
import glob
import numpy as np
import os.path
from PIL import Image
import time


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
                img_name = "%s/%d.jpg" % (path_name, num+1)
                print(img_name)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                num += 1
                if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                    break

                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                #显示当前捕捉到了多少人脸图片
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d/1 %d' % (num,len(faceRects)),(x + 30, y + 30), font, 1, (255,0,255),4)


        if num > (catch_pic_num):
            break

        #显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
        time.sleep(0.1)
            #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
class VoiceGame(cocos.layer.ColorLayer):
    is_event_handler = True

    def __init__(self):
        super(VoiceGame, self).__init__(255, 255, 255, 255, 800, 600)

        self.logo = cocos.sprite.Sprite('1.png')
        self.logo.position = 550, 400
        self.add(self.logo, 99999)

        # init voice
        self.NUM_SAMPLES = 1000  # pyAudio内部缓存的块的大小
        self.LEVEL = 3000  # 声音保存的阈值

        self.voicebar = Sprite('black.png', color=(0, 0, 255))
        self.voicebar.position = 20, 450
        self.voicebar.scale_y = 0.1
        self.voicebar.image_anchor = 0, 0
        self.add(self.voicebar)

        self.ppx = PPX()
        self.add(self.ppx)

        self.floor = cocos.cocosnode.CocosNode()
        self.add(self.floor)
        pos = 0, 100
        for i in range(100):
            b = Block(pos)
            self.floor.add(b)
            pos = b.x + b.width, b.height

        # 开启声音输入
        pa = PyAudio()
        SAMPLING_RATE = int(pa.get_device_info_by_index(0)['defaultSampleRate'])
        self.stream = pa.open(format=paInt16, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=self.NUM_SAMPLES)

        self.schedule(self.update)

    def on_mouse_press(self, x, y, buttons, modifiers):
        pass

    def collide(self):
        px = self.ppx.x - self.floor.x
        for b in self.floor.get_children():
            if b.x <= px + self.ppx.width * 0.8 and px + self.ppx.width * 0.2 <= b.x + b.width:
                if self.ppx.y < b.height:
                    self.ppx.land(b.height)
                    break

    def update(self, dt):
        # 读入NUM_SAMPLES个取样
        string_audio_data = self.stream.read(self.NUM_SAMPLES)
        k = max(struct.unpack('1000h', string_audio_data))
        # print k
        self.voicebar.scale_x = k / 10000.0
        if k > 1500:#3000
            self.floor.x -= min((k / 20.0), 150) * dt
        if k > 6000:#8000
            self.ppx.jump((k - 8000) / 1000.0)
        self.collide()

    def reset(self):
        self.floor.x = 0
if __name__ == '__main__':
    # 连续截100张图像，存进image文件夹中
    # CatchPICFromVideo("get face", 0, 2, "C:\\Users\\Administrator\\Desktop\\face_recognition\\faces")
    print('--------正在准备设备中----------')
    time.sleep(2)
    CatchPICFromVideo("get_face", 0, 20, "/home/wz/Desktop/chuangke/")
    print('-----------识别中-------------')
    time.sleep(3)
    # cut
    print('-------图像正在归一化----------')
    time.sleep(3)
    for jpgfile in glob.glob("/home/wz/Desktop/chuangke/*.jpg"):
        img_resize(jpgfile, "/home/wz/Desktop/chuangke/", 96, 96)
    #gameplay
    print('----------gameplay-------------')
    time.sleep(2)
    cocos.director.director.init(caption="play")
    cocos.director.director.run(cocos.scene.Scene(VoiceGame()))

