import numpy as np
import cv2
from mss import mss
from PIL import Image
import os
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import load_model

bounding_box = {'top': 200, 'left': 100, 'width': 500, 'height': 500}
model = load_model('weights/fromscratch.h5')
sct = mss()
face_haar_cascade = cv2.CascadeClassifier('haar.xml')

while True:
	sct_img = np.array(sct.grab(bounding_box))
	gray_img= cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
	faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
	
	for (x,y,w,h) in faces_detected:
		cv2.rectangle(sct_img,(x,y),(x+w,y+h),(0,0,0),thickness=3)
		roi_gray=gray_img[y:y+w,x:x+h]
		roi_gray=cv2.resize(roi_gray,(48,48))
		img_pixels = image.img_to_array(roi_gray)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255

		predictions = model.predict(img_pixels)

		max_index = np.argmax(predictions[0])

		emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
		predicted_emotion = emotions[max_index]
		predicted_probability = np.max(predictions[0])*100
		text_to_show = predicted_emotion + ' ' + str(int(predicted_probability)) + '%'	
		cv2.putText(sct_img, text_to_show, (int(x), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow('Emotion recognition', sct_img)

	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
