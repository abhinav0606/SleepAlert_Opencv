import cv2
from pygame import mixer
mixer.init()
sound=mixer.Sound('alarm.wav')
faces=cv2.CascadeClassifier("face.xml")
eyes=cv2.CascadeClassifier("close_eye.xml")
score=0
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=3)
        gray_face=gray[y:y+h,x:x+w]
        color_face=frame[y:y+h,x:x+w]
        eye=eyes.detectMultiScale(gray_face,1.3,5)
        if eye is ():
            score=score+1

        else:
            for (a,b,c,d) in eye:
                cv2.rectangle(color_face,(a,b),(a+c,b+d),(100,100,100),thickness=3)
    cv2.putText(frame, "Score" + str(score), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    if score>10:
        sound.play()
        score=0
    cv2.imshow("Abhinav's Frame",frame)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()