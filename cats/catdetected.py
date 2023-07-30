import argparse
import cv2

#только если в пути к фалу нет не латиницы
"""""ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help = "path to the input image")
ap.add_argument("-c", "--cascade", default="haarcascade_frontalcatface.xml", help="path to  cat detector haar cascade")
args = vars(ap.parse_args())"""""

image_path = "cats2.jpg"
Cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = Cat_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75,75))

for (i, (x, y, w, h)) in enumerate(rects):
    cv2.rectangle(image, (x, y), (x+w, y+h), (150, 0, 209), 2)
    cv2.putText(image, "Cat #{}".format(i+1), (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 0, 209), 2)

cv2.imshow("Cat Faces", image)
cv2.waitKey(0)