from os import makedirs
import shutil
import face_recognition as fr
from unittest.mock import _SentinelObject
import cv2
from matplotlib import pyplot as plt
import numpy as np
from videoCapture import listdir, isfile, join, path, secondsVideo, activateVideoSecunds
# padronizando imagens
imgPath = "GAB/"
faces_list = [i for i in listdir(imgPath) if isfile(join(imgPath, i))]

treinoPath = "imagens/treino/"
testePath = "imagens/teste/"

if not path.exists(treinoPath):
    makedirs(treinoPath)


if not path.exists(testePath):
    makedirs(testePath)

segundos = activateVideoSecunds()
print(segundos)
for arq in faces_list:
    people = arq[0:3]
    idpeople = arq[4:8]
    value = int((segundos[0]*segundos[1])*0.07+1)

    if int(idpeople) <= value:
        shutil.copyfile(imgPath + arq, treinoPath + arq)
    else:
        shutil.copyfile(imgPath + arq, testePath + arq)


# Usando classificador Dlib detectando rostos

def recognitionFace(val):
    image = fr.load_image_file(val)
    faces = fr.face_encodings(image)

    if(len(faces) > 0):
        return True, faces
    return False, []


def get_faces():
    peopleRecognition = []
    labelsRecognition = []

    facesDetec = recognitionFace("imagens/treino/GAB_0000.jpg")
    if(facesDetec[0]):
        peopleRecognition.append(facesDetec[1][0])
        labelsRecognition.append("Gabriel")

    return peopleRecognition, labelsRecognition

# dataTrain, sujectTrain = [], []
# dataTest, sujectTest = [], []

# pegando imagens de treino


# def padronizar_imagem(imagem_caminho):
#     imagem = cv2.imread(imagem_caminho, cv2.IMREAD_GRAYSCALE)
#     imagem = cv2.resize(imagem, (400, 700), interpolation=cv2.INTER_LANCZOS4)
#     return imagem


# faces_train = [i for i in listdir(treinoPath) if isfile(join(treinoPath, i))]
# faces_test = [i for i in listdir(testePath) if isfile(join(testePath, i))]

# for i, arq in enumerate(faces_train):
#     imagem_path = (treinoPath + arq)
#     imagem = padronizar_imagem(imagem_path)
#     dataTrain.append(imagem)
#     sujeitos = arq[4:8]
#     sujectTrain.append(sujeitos)

# # pegando imagens de treino
# for i, arq in enumerate(faces_test):
#     imagem_path = testePath + arq
#     imagem = padronizar_imagem(imagem_path)
#     dataTest.append(faces_test)
#     sujeitos = arq[4:8]
#     sujectTest.append(sujeitos)

# print(type(dataTrain))
# plt.imshow(dataTrain[0], cmap="gray")
# plt.title(sujectTrain[0])
# plt.show()

# transformando em int32 para o classificador
# sujectTrain = np.asarray(sujectTrain, dtype=np.int32)
# sujectTest = np.asarray(sujectTest, dtype=np.int32)


# eingModel = cv2.face.EigenFaceRecognizer_create()
# eingModel.train(dataTrain, sujectTrain)
# print(eingModel)


# plt.figure(figsize=(20, 10))

# plt.subplot(121)
# plt.title("Sujeito " + str(dataTest[21]))
# # plt.imshow(dataTest[21], cmap="gray")
# plt.show()

# plt.subplot(122)
# plt.title("Sujeito " + str(dataTest[27]))
# # plt.imshow(dataTest[27], cmap="gray")

# plt.show()

# predicao = eingModel.predict(dataTest[21])
# print(predicao)
# predicao = eingModel.predict(dataTest[27])
# print(predicao)
