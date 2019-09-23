# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
import os
import glob

current_path = os.getcwd()	# 获取当前路径
predictor_path = current_path + "\\model\\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = current_path + "\\model\\dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = current_path + "\\faces\\"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

for img_path in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
	print("Processing file: {}".format(img_path))
	# opencv 读取图片，并显示
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	# opencv的bgr格式图片转换成rgb格式
	b, g, r = cv2.split(img)
	img2 = cv2.merge([r, g, b])
	
	dets = detector(img, 1)
	print("Number of faces detected: {}".format(len(dets)))
	
	for index, face in enumerate(dets):
		print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
		
		shape = shape_predictor(img2, face)
		for i, pt in enumerate(shape.parts()):
			#print('Part {}: {}'.format(i, pt))
			pt_pos = (pt.x, pt.y)
			cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
			#print(type(pt))
		#print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
		cv2.namedWindow(img_path+str(index), cv2.WINDOW_AUTOSIZE)
		cv2.imshow(img_path+str(index), img)
		
		face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)
		print(face_descriptor)
		
		
	
k = cv2.waitKey(0)
cv2.destroyAllWindows()
	