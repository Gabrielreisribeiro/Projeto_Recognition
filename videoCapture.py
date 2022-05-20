from csv import list_dialects
import cv2
import matplotlib.pyplot as pl
from matplotlib import pyplot as plt
import datetime

from os import listdir, path, makedirs
from os.path import isfile, join
import os
import re
import dlib


from sklearn.metrics import accuracy_score
from io import BytesIO
from IPython.display import clear_output, Image, display
from PIL import Image as Img


shape_predictor_filename = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_filename)
# variaveis
video = "faces/videos/V1-GAB.mp4"
Datavideo = cv2.VideoCapture(video)

# padronizando cor dos frames


def frameColor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (400, 700))
    return frame

# salvando frames dos videos


def FrameCapture(frame, Datavideo, thresholdFaces=6, stepFrame=10):
    detectedFacesCounter = 0
    faces = []
    count = 0000
    success = 1
    namePath = frame[16:19]
    if not path.exists(namePath):
        makedirs(namePath)
    os.chdir(namePath)
    while success:
        count += 1
        success, video = Datavideo.read()
        # acabou o vídeo ou detectou o limite de faces?
        if success == False or detectedFacesCounter >= thresholdFaces:
            return faces

        # "pula" alguns frames
        if count % stepFrame != 0:
            continue

        # detecta a face, o 1 indica superamostragem, pra achar faces mais facilmente
        detections = detector(video, 1)
        for i, detection in enumerate(detections):
            # salvar região das faces em uma pasta
            x, y = detection.left(), detection.top()
            w, h = detection.right() - detection.left(), detection.bottom() - detection.top()
            faceCrop = video[y:y + h, x:x + w]
            namefile = str(namePath + "_%d.jpg" % count)
            fileNumber = re.findall(r'\d', namefile)
            fileNumber = (''.join(fileNumber)).zfill(4)
            cv2.imwrite(namePath + '_' + fileNumber + ".jpg", faceCrop)

            # essa parte é só pra adicionar retângulo ao redor da face e marcações, pra exibir bonitinho depois
            cv2.rectangle(video, (x, y), (x + w, y + h), (255, 255, 0), 4)
            landmarks = predictor(video, detection)
            for i in range(0, 68):
                cv2.circle(video, (landmarks.part(i).x,
                           landmarks.part(i).y), 4, (0, 0, 255), -1)
            faces.append(video)
        detectedFacesCounter += len(detections)


if __name__ == '__main__':
    frameExpl = FrameCapture(video, Datavideo)

# # mostra algumas faces detectadas de exemplo
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
# i = 0
# for y in range(0, 2):
#     for x in range(0, 2):
#         axes[x, y].set_title('Face ' + str(i), fontsize=14)
#         frameExpl[i] = cv2.cvtColor(frameExpl[i], cv2.COLOR_BGR2RGB)
#         axes[x, y].imshow(frameExpl[i])
#         i += 1
#         plt.show()


def secondsVideo(video):
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    segundos = int(frames/fps)
    timeVideo = str(datetime.timedelta(seconds=segundos))
    timeVideo = int(timeVideo[5:7])
    return timeVideo, fps


def activateVideoSecunds():
    return secondsVideo(Datavideo)
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
