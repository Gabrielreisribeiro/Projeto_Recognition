import face_recognition as fr
from imageP import get_faces, recognitionFace


desconhecido = recognitionFace("imagens/teste/GAB_0057.jpg")

if(desconhecido[0]):
    rostos_desconhecido = desconhecido[1][0]
    rostos_conhecidos, nomes_conhecidos = get_faces()
    print(len(rostos_conhecidos))
    result = fr.compare_faces(rostos_conhecidos, rostos_desconhecido)
    print(result)

    for i in range(len(rostos_conhecidos)):
        results = results[i]
        if(results):
            print("rosto do", nomes_conhecidos[i], "foi reconhecido")
