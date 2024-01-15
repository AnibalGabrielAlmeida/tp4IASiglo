# Transformada de Hough para Rectas

Este código en Python utiliza la biblioteca OpenCV para aplicar la Transformada de Hough a una imagen y detectar líneas rectas. Se carga una imagen en escala de grises, se aplica el operador Canny para detectar bordes y luego se aplica la Transformada de Hough para encontrar las líneas en la imagen.

### Código

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('motor2.webp')
# Cambio de la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar el operador Canny para detectar bordes en la imagen
edges = cv2.Canny(gray, 80, 90, apertureSize=3)
# Aplicar la Transformada de Hough
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

# Visualización de los resultados
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Transformada de Hough para Rectas')
plt.show()
```
# Transformada de Hough para Circunferencias

Este código en Python utiliza la biblioteca OpenCV para aplicar la Transformada de Hough a una imagen y detectar circunferencias. Se procesa una imagen en escala de grises, y se aplica la Transformada de Hough para círculos. Luego, se dibujan las circunferencias encontradas en la imagen original.

### Código

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Procesar la imagen
image = cv2.imread('motor2.webp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Transformada de Hough para círculos
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=60, param2=40, minRadius=15, maxRadius=40
)

# Dibujar las circunferencias
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibujar el círculo en la imagen original
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Visualizar los resultados obtenidos
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Transformada de Hough para Circunferencias')
plt.show()

