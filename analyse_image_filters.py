# Carrega imagem
#img_color = cv2.imread('./images/train/granite-blackswan/IMG_6410.JPG')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.ndimage import median_filter
from skimage.util import random_noise

# Funções para filtros de frequência
def butterworth_lowpass_filter(shape, cutoff, order):
    P, Q = shape
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    return H

def ideal_highpass_filter(shape, cutoff):
    P, Q = shape
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = np.where(D > cutoff, 1, 0)
    return H

def homomorphic_filter(image, gammaL=0.5, gammaH=2.0, c=1, d0=30):
    image_log = np.log1p(np.array(image, dtype="float"))
    fft = fftpack.fftshift(fftpack.fft2(image_log))
    rows, cols = image.shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = (gammaH - gammaL) * (1 - np.exp(-c * (D**2 / d0**2))) + gammaL
    filtered = np.real(fftpack.ifft2(fftpack.ifftshift(fft * H)))
    return np.expm1(filtered)

# Carregamento e pré-processamento da imagem
#img = cv2.imread('./images/train/granite-blackswan/IMG_6410.JPG')
img = cv2.imread('./images/gato.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtros espaciais
blurred = cv2.GaussianBlur(img_rgb, (21, 21), 0)
median = cv2.medianBlur(img_rgb, 21)
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))


# Filtros no domínio da frequência
gray_float = img_gray.astype(float)
fft = fftpack.fftshift(fftpack.fft2(gray_float))
butterworth = butterworth_lowpass_filter(img_gray.shape, cutoff=30, order=2)
ideal_highpass = ideal_highpass_filter(img_gray.shape, cutoff=20)

butterworth_result = np.real(fftpack.ifft2(fftpack.ifftshift(fft * butterworth)))
ideal_highpass_result = np.real(fftpack.ifft2(fftpack.ifftshift(fft * ideal_highpass)))
homo_result = homomorphic_filter(img_gray)

# Adiciona ruído sal e pimenta e aplica filtro de mediana
noisy = random_noise(img_rgb, mode='s&p', amount=0.05)
noisy = (255 * noisy).astype(np.uint8)
median_noisy = cv2.medianBlur(noisy, 5)

# Exibição dos resultados
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()

titles = [
    "Imagem original (RGB)",
    "Filtro de Média (Gaussian Blur)",
    "Filtro de Mediana",
    "Filtro Sobel (cinza)",
    "Filtro Butterworth (passa-baixa)",
    "Filtro Homomórfico",
    "Filtro Ideal (passa-alta)",
    "Imagem com Ruído Sal e Pimenta",
    "Filtro de Mediana no Ruído"
]

images = [
    img_rgb,
    blurred,
    median,
    sobel_magnitude,
    butterworth_result,
    homo_result,
    ideal_highpass_result,
    noisy,
    median_noisy
]

for i in range(9):
    if images[i].ndim == 2:
        axes[i].imshow(images[i], cmap='gray')
    else:
        axes[i].imshow(images[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()
