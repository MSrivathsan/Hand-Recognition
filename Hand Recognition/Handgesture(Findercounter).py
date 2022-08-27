import os
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)


curtime = 0
prevtime = 0

folderpath = "Fingerimages"
filelist = os.listdir(folderpath)
overlaylist = []
print(filelist)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.6, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

for imgpath in filelist:
    image = cv2.imread(f'{folderpath}/{imgpath}')
    overlaylist.append(image)

tipids = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmlist = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                px, py = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, px, py])

                if id in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
                    cv2.circle(img, (px,py), 17, (50,0,150), 3)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmlist) != 0:
        boolfing = []
        if lmlist[tipids[0]][1] < lmlist[tipids[0]-1][1]:
            boolfing.append(1)
        else:
            boolfing.append(0)

        for id in range(1, 5):
            if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
               boolfing.append(1)
            else:
                boolfing.append(0)

        totalfing= boolfing.count(1)

        if totalfing == 5:
            oh, ow, oc = overlaylist[2].shape
            img[0:oh, 0:ow] = overlaylist[2]
        if totalfing == 1:
            oh, ow, oc = overlaylist[5].shape
            img[0:oh, 0:ow] = overlaylist[5]
        if totalfing == 2:
            oh, ow, oc = overlaylist[3].shape
            img[0:oh, 0:ow] = overlaylist[3]
        if totalfing == 4:
            oh, ow, oc = overlaylist[1].shape
            img[0:oh, 0:ow] = overlaylist[1]
        if totalfing == 3:
            oh, ow, oc = overlaylist[4].shape
            img[0:oh, 0:ow] = overlaylist[4]

    curtime = time.time()
    fps = 1 / (curtime - prevtime)
    prevtime = curtime
    cv2.putText(img, 'FPS : ' + str(int(fps)), (1190, 30), cv2.FONT_HERSHEY_PLAIN, 1, (150, 50, 20), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)