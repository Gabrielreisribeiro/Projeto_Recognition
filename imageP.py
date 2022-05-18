from videoCapture import listdir, isfile, join
# padronizando imagens
imgPath = "GAB"
faces_list = [i for i in listdir(imgPath) if isfile(join(imgPath, i))]

print(len(faces_list))
