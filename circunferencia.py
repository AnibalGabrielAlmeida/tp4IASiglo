import cv2
import numpy as np
import matplotlib.pyplot as plt

#Se procesa la imagen
image = cv2.imread('motor2.webp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Transformada de Hough
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=60, param2=40, minRadius=15, maxRadius=40
)

# Dibujar las circunferencias
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibujar el círculo en la imagen original
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Visualizar los resultados obtenidos.
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Transformada de Hough para Circunferencias')
plt.show()
