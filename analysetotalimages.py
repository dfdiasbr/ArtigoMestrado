import os
import pandas as pd

def contar_imagens(diretorio):
    imagens_por_subdiretorio = {}
    for root, dirs, files in os.walk(diretorio):
        total_imagens = 0
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                total_imagens += 1
        imagens_por_subdiretorio[root] = total_imagens
    return imagens_por_subdiretorio

diretorios = ['./images/images-datasetDouglascompleto/train', 
              './images/images-datasetDouglascompleto/val', 
              './images/images-datasetDouglascompleto/test']

data = []
for diretorio in diretorios:
    imagens_por_subdiretorio = contar_imagens(diretorio)
    for subdiretorio, total_imagens in imagens_por_subdiretorio.items():
        data.append({'Diretório': diretorio, 'Subdiretório': subdiretorio, 'Total de imagens': total_imagens})

df = pd.DataFrame(data)
print(df)

df.to_csv('analysetotalimages.csv', index=False)