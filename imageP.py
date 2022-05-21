from os import makedirs
import shutil
import face_recognition as fr
from unittest.mock import _SentinelObject
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
from videoCapture import listdir, isfile, join, path, secondsVideo, activateVideoSecunds, os
# padronizando imagens


def generateEncodings(imgPath, name, knownEncodings, knownNames):
    for filename in os.listdir(imgPath):
        print(filename)
        img = fr.load_image_file(imgPath + filename)
        boxes = fr.face_locations(img, model='cnn')
        encodings = fr.face_encodings(img, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)


knownEncodings = []
knownNames = []

imgPath = "GABRIEL/"
faces_list = [i for i in listdir(imgPath) if isfile(join(imgPath, i))]
name = imgPath[0:7]
generateEncodings(imgPath, name, knownEncodings, knownNames)

imgPath = "KAREN/"
faces_list = [i for i in listdir(imgPath) if isfile(join(imgPath, i))]
name = imgPath[0:5]

generateEncodings(imgPath, name, knownEncodings, knownNames)

data_encoding = {"encodings": knownEncodings, "names": knownNames}

f = open("face_encodings", "wb")
f.write(pickle.dumps(data_encoding))
f.close()

# treinoPath = "imagens/treino/"
# testePath = "imagens/teste/"

# if not path.exists(treinoPath):
#     makedirs(treinoPath)


# if not path.exists(testePath):
#     makedirs(testePath)

# segundos = activateVideoSecunds()
# print(segundos)
# for arq in faces_list:
#     people = arq[0:3]
#     idpeople = arq[4:8]
#     value = int((segundos[0]*segundos[1])*0.07+1)

#     if int(idpeople) <= value:
#         shutil.copyfile(imgPath + arq, treinoPath + arq)
#     else:
#         shutil.copyfile(imgPath + arq, testePath + arq)


# Usando classificador Dlib detectando rostos


# def recognitionFace(val):
#     image = fr.load_image_file(val)
#     faces = fr.face_encodings(image)

#     if(len(faces) > 0):
#         return True, faces
#     return False, []


# def get_faces():
#     peopleRecognition = []
#     labelsRecognition = []

#     facesDetec = recognitionFace("imagens/treino/GAB_0000.jpg")
#     if(facesDetec[0]):
#         peopleRecognition.append(facesDetec[1][0])
#         labelsRecognition.append("Gabriel")

#     return peopleRecognition, labelsRecognition
