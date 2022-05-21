from cv2 import imshow
import numpy as np
import face_recognition as fr
import cv2
import os.path
import pickle
from tqdm import tqdm

video = "faces/videos/V1-GAB.mp4"

# carrega arquivo binário contendo faces codificadas
data_encoding = pickle.loads(open("face_encodings", "rb").read())
# carrega vídeo do disco
videoCaptureInput = cv2.VideoCapture(0)

# set contendo as possíveis pessoas reconhecidas
unique_names = set(data_encoding["names"])

# gerador de vídeo contendo saída com faces reconhecidas
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# fps = videoCaptureInput.get(cv2.CAP_PROP_FPS)
# videoCaptureOutput = cv2.VideoWriter("output.mp4", fourcc, fps, (1920, 1080))

# gera reconhecimento em vídeo para os 200 primeiros frames
# for i in tqdm(range(0, 500)):
x = 1
# gera frames infinitos
while x:
    # para cada frame
    success, frame = videoCaptureInput.read()

    # # acabou o vídeo?
    # if success == False:
    #     break

# converte frame de formato BGR (OpenCV) para RGB (face_recognition)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = fr.face_locations(frame)
    encodings = fr.face_encodings(frame, boxes)

    names = []

    # para cada codificação de faces encontrada
    for encoding in encodings:
        matches = fr.compare_faces(
            data_encoding["encodings"], encoding)
        facesdistaces = fr.face_distance(data_encoding["encodings"], encoding)
        # print(matches)
        # print("separação")
        # print(facesdistaces)
        # retorna o identificador da lista das faces da base que "batem" com a codificação verificada
        matchesId = [i for i, value in enumerate(
            matches)
            if value == True]
        precisao = sum(facesdistaces)/len(facesdistaces)
        votacao = np.argmin(facesdistaces)

        counts = {}
        for name in unique_names:
            counts[name] = 0
        for i in matchesId:
            name = data_encoding["names"][i]
            counts[name] += 1
            if(precisao <= 0.7):
                name = data_encoding["names"][votacao]
                names.append(name)

        if len(matchesId) == 0:
            name = "Desconhecido"
            names.append(name)

    # desenha o retângulo e escreve o nome da pessoa no frame
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
        cv2.putText(frame, name, (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # converte o frame de volta pro formato do OpenCV (BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # escreve o frame no arquivo de vídeo
    # videoCaptureOutput.write(frame)
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCaptureInput.release()
# videoCaptureOutput.release()
