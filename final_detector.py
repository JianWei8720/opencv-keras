#coding=utf-8
import cv2 as cv
from CNN_train import Model
#from image_show import show_image
face_cascade = cv.CascadeClassifier('/home/jianwei/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_alt.xml')

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    model = Model()
    model.load()
    while cap.isOpened():
        _, image = cap.read()
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #灰度化图片 
        equalImage = cv.equalizeHist(grayImage) #直方图均衡化
        faces = face_cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        if len(faces) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for (x,y,w,h) in faces:
                #裁剪出人脸，单独保存成图片
                head = image[y-10:y+h,x:x+w]
                result = model.predict(head)
                if result == 0:  # boss
		    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
		    cv.putText(image, 'WeiJian', (x+5,y+5), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
 #                   show_image()
                else:
		    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
		    cv.putText(image, 'Others', (x+5,y+5), cv.FONT_HERSHEY_SIMPLEX, 1, (0,120,255), 1)
	    cv.imshow("Find me", image)
        key = cv.waitKey(40)
        if key == 27 or key == ord('q'): #如果按ESC或q键，退出
            break

    cap.release()
    cv.destroyAllWindows()
