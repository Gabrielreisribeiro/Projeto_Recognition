from csv import list_dialects
import cv2
import matplotlib.pyplot as plt

from os import listdir, path, makedirs
from os.path import isfile, join
import os

import shutil

from sklearn.metrics import accuracy_score
from io import BytesIO
from IPython.display import clear_output, Image, display
from PIL import Image as Img

# variaveis
video = "faces/videos/V1-GAB.mp4"
# padronizando cor dos frames


def frameColor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (400, 700))
    return frame

# salvando frames dos videos


def FrameCapture(frame):
    video = cv2.VideoCapture(frame)
    count = 0
    success = 1
    namePath = frame[16:19]
    if not path.exists(namePath):
        makedirs(namePath)
    os.chdir(namePath)
    while success:
        success, image = video.read()
        image = frameColor(image)
        cv2.imwrite(namePath + "_%d.jpg" % count, image)

        count += 1


if __name__ == '__main__':
    FrameCapture(video)


# if (video.isOpened() == False):
#     print("Erro ao abrir video")

# while(video.isOpened()):

#     ret, frame = video.read()
#     if ret == True:

#         cv2.imshow('Frame', frame)

#         if cv2.waitKey(27) & 0xFF == 27:
#             break

#     else:
#         break

# video.release()

# cv2.destroyAllWindows()
