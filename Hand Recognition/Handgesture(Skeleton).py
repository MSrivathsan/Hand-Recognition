import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.6, max_num_hands=3)                                 # Hands(static_image_mode,max_num_hands,min_detection_confidence, min_tracking_confidence) default values (False, 2, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils                     # points drawing utility provided by mediapipe

curtime = 0
prevtime = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       #Convert img into RGB for Hands() object which is hands
    results = hands.process(imgRGB)
                                                        #print(results.multi_hand_landmarks) can be used to know when the hand enters the frame

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape                                               #we are converting the default coordinate values of the 21 points with ids to pixel values
                px, py = int(lm.x*w), int(lm.y*h)
                                                                                  #print(id, px, py) will print the pixel coordinates of the 21 data points of the hand in the img
                if id in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
                    cv2.circle(img, (px,py), 17, (50,0,150), 3)                   #circle func plots circle around a specific pixel on the img with attributes (img, the coordintes of the pixel,radius of the circle, colour, thickness of the circle)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)         #we draw the 21 points on the displayed img and not the RGB img for each hand in the frame, we connect those 21 ponts usings an attribute called HAND_CONNECTIONS

    curtime = time.time()
    fps = 1/(curtime-prevtime)
    prevtime = curtime
    cv2.putText(img, 'FPS : '+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (150,50,20), 1)     #put text inserts a particular text on the display screen with atributes (type of output, the text, placement of the text, font, scale/size of the test, colour, thicknss)

    cv2.imshow("Image", img)                            #shows the image seen by the webcam
    cv2.waitKey(1)