

import cv2
from tqdm import tqdm

videoCaptureInput = cv2.VideoCapture(0)

c = 1
while c:
    success, frame = videoCaptureInput.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
