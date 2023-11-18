import cv2
import numpy as np
import matplotlib.pyplot as plt

# cargar la imagen
image = cv2.imread('motor2.webp')
# cambio de la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#se aplica el operador canny para detectar bordes en la imagen
edges = cv2.Canny(gray, 80, 90, apertureSize=3)
#esta linea aplica la transf de hough. Fui probando distints thresold.
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

# Secuencia para dibujar las líneas detectadas en la imagen original
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

# visualizacion de los resultados
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Transformada de Hough para Rectas')
plt.show()
