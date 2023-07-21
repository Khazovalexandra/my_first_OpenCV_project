import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')      #лицо
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')                       #глаза
glasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')  #очки
smile_cascade = cv2.CascadeClassifier('haarcascade_smile1.xml')                 #улыбка

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    eye = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
    #glasses = glasses_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=1)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 110, 180), 2)
    for(sx, sy, sw, sh) in smiles:
        cv2.rectangle(frame, (sx,sy), (sx+sw,sy+sh), (255,165,0), 2)
    #for(gx, gy, gw, gh) in glasses:
        #cv2.rectangle(frame, (gx,gy), (gx+gw,gy+gh), (139,58,58), 2)
    if len(smiles) != 0:
            cv2.putText(frame, "Smiling", (sx,sy-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Not smiling", (sx,sy-30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()