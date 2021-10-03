import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_color)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
               h, w, c = img.shape
               cx, cy = int(lm.x*w), int(lm.y*h)
               print(id, cx, cy)
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime= cTime
    # cv2.putText(img,str(int(fps)), (10,70),cv2.QT_FONT_NORMAL, 1, (255,255,255), 2)
    cv2.imshow("Image",img)
    if cv2.waitKey(10)==ord('q'):
        break
