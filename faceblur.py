import cv2
import os

path = os.getcwd()
if not os.path.isdir("Haar"):
    print("A pasta não existe")
else:
    print("a pasta existe")

cascade = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")


def face_blur(gray, frame):
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi = frame[y: y+h, x: x+w]
        blur = cv2.GaussianBlur(roi, (101, 101), 0)
        frame[y: y + h, x: x + w] = blur
    return frame


video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurs = face_blur(gray, frame)
    cv2.imshow("Hello", blurs)
    if cv2.waitKey(1) == 32:
        break

video.release()
cv2.destroyAllWindows()
