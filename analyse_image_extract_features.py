import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, graycomatrix, graycoprops
from skimage import exposure

# 1. Carregar a imagem
#img_path = './images/train/granite-blackswan/IMG_6410.JPG'
#img_path = './images/gato.jpg'  # troque se necessário
#img_bgr = cv2.imread('./images/gato.jpg')  # Substitua pelo caminho da sua imagem

# Carrega a imagem
img_bgr = cv2.imread("./images/gato.jpg")  # Substitua pelo caminho da sua imagem
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Extração de HOG
hog_features, hog_image = hog(
    img_gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True
)
hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Extração de GLCM (Haralick)
glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
contrast = graycoprops(glcm, 'contrast')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

# Lista de características
haralick_labels = ['Contraste', 'Correlação', 'Homogeneidade', 'Energia']
haralick_values = [contrast, correlation, homogeneity, energy]

# Visualização
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Imagem original
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title('Imagem Original (RGB)')
axs[0, 0].axis('off')

# HOG
axs[0, 1].imshow(hog_image, cmap='gray')
axs[0, 1].set_title('HOG (Gradiente de Orientação)')
axs[0, 1].axis('off')

# Histograma RGB combinado
axs[1, 0].hist(img_rgb[:, :, 0].ravel(), bins=256, color='red', alpha=0.5, label='R')
axs[1, 0].hist(img_rgb[:, :, 1].ravel(), bins=256, color='green', alpha=0.5, label='G')
axs[1, 0].hist(img_rgb[:, :, 2].ravel(), bins=256, color='blue', alpha=0.5, label='B')
axs[1, 0].set_title('Histograma RGB Combinado')
axs[1, 0].legend()

# Gráfico das características Haralick
axs[1, 1].bar(haralick_labels, haralick_values, color='skyblue')
axs[1, 1].set_title('Características Haralick (GLCM)')
axs[1, 1].set_ylabel('Valor')
axs[1, 1].set_ylim(0, max(haralick_values) + 0.1)

plt.tight_layout()
plt.show()
