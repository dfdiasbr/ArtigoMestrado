import os
import random
import shutil


def selecionar_imagens_aleatorias(diretorios_quantidades):
    for diretorio_origem, (quantidade, diretorio_destino) in diretorios_quantidades.items():
        print(f"Selecionando imagens em {diretorio_origem}...")
        for root, dirs, files in os.walk(diretorio_origem):
            # Lista de arquivos de imagens na pasta atual
            imagens = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            
            # Se houver imagens suficientes para atender à quantidade especificada
            if len(imagens) >= quantidade:
                # Diretório relativo à pasta de origem
                pasta_rel_origem = os.path.relpath(root, diretorio_origem)
                #print(f"pasta_rel_origem:{pasta_rel_origem}")
                # Diretório relativo à pasta de destino
                pasta_rel_destino = os.path.join(diretorio_destino, pasta_rel_origem)
                #print(f"pasta_rel_destino:{pasta_rel_destino}")
                
                # Criar subdiretórios intermediários no diretório de destino, se necessário
                if not os.path.exists(pasta_rel_destino):
                    os.makedirs(pasta_rel_destino)
                
                # Selecionar aleatoriamente 'quantidade' de imagens
                imagens_selecionadas = random.sample(imagens, quantidade)
                
                # Copiar as imagens selecionadas para o novo diretório
                for imagem in imagens_selecionadas:
                    origem_imagem = os.path.join(root, imagem)
                    destino_imagem = os.path.join(diretorio_destino, pasta_rel_origem, imagem)
                    shutil.copy(origem_imagem, destino_imagem)
        
        #print(f"{quantidade} imagens selecionadas em {diretorio_origem} e copiadas para {pasta_rel_destino}")

diretorios_quantidades = {
    './images/images-datasetJoaoVictor/train': [210,'./images/images-datasetJoaoVictor/datasetbalanceado/train'],
    './images/images-datasetJoaoVictor/val':  [45,'./images/images-datasetJoaoVictor/datasetbalanceado/val'],
    './images/images-datasetJoaoVictor/test': [45,'./images/images-datasetJoaoVictor/datasetbalanceado/test']
}

selecionar_imagens_aleatorias(diretorios_quantidades)