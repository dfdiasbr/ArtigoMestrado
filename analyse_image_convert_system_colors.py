import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem em BGR
#img_bgr = cv2.imread('./images/train/granite-blackswan/IMG_6410.JPG')
img_bgr = cv2.imread('./images/gato.jpg')
img_bgr = cv2.resize(img_bgr, (400, 400))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Separa canais RGB
r, g, b = cv2.split(img_rgb)

# Cria imagens com os canais em cor real (mantém o canal desejado, zera os outros)
zeros = np.zeros_like(r)
r_rgb = cv2.merge([r, zeros, zeros])
g_rgb = cv2.merge([zeros, g, zeros])
b_rgb = cv2.merge([zeros, zeros, b])

# Função para exibir 3 imagens lado a lado
def show_channels_color(imgs, titles):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# Exibe imagem original
plt.imshow(img_rgb)
plt.title("Imagem Original (RGB)")
plt.axis('off')
plt.show()

# Exibe canais RGB com cores reais
show_channels_color([r_rgb, g_rgb, b_rgb], ['Canal Vermelho (R)', 'Canal Verde (G)', 'Canal Azul (B)'])

# HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)
show_channels_color([h, s, v], ['Hue (H)', 'Saturação (S)', 'Valor (V)'])

# LAB
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l_lab, a_lab, b_lab = cv2.split(img_lab)
show_channels_color([l_lab, a_lab, b_lab], ['L*', 'a*', 'b*'])

# YCrCb
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_ycrcb)
show_channels_color([y, cr, cb], ['Y (Luminância)', 'Cr', 'Cb'])
