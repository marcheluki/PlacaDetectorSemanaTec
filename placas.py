import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# 1. Cargar imagen
imagen = cv2.imread("placaJalisco.jpg")
if imagen is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# 2. Convertir la imagen a HSV para aislar el color negro
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Rango para detectar 'negro' en HSV (ajusta según tu imagen)
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([180, 255, 50], dtype=np.uint8)

# Máscara que deja solo las regiones negras
mask_black = cv2.inRange(hsv, lower_black, upper_black)

# 3. Suavizado para difuminar bordes muy pequeños
mask_black = cv2.GaussianBlur(mask_black, (5,5), 0)

# 4. Operaciones morfológicas para cerrar huecos y eliminar ruido
kernel = np.ones((3,3), np.uint8)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=2)

# 5. Eliminar objetos pequeños (por ejemplo, el QR) filtrando contornos por área
contornos, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_filtrada = mask_black.copy()

for cnt in contornos:
    area = cv2.contourArea(cnt)
    if area < 1000:  # Ajusta este umbral según el tamaño de tus letras vs. QR
        cv2.drawContours(mask_filtrada, [cnt], -1, 0, -1)  # Pintamos de negro ese contorno

# 6. Eliminar contornos que estén cerca de los bordes (ej. símbolos en las esquinas)
# Calculamos ancho y alto de la imagen para definir la "zona segura" central
altura, ancho = mask_filtrada.shape
margin = 100  # Ajusta este margen según la posición de las letras en tu placa

# Recalcular contornos después de haber quitado los objetos pequeños
contornos, _ = cv2.findContours(mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contornos:
    x, y, w, h = cv2.boundingRect(cnt)
    # Si el bounding box está cerca de un borde, lo eliminamos
    if (x < margin or (x + w) > (ancho - margin) or
        y < margin or (y + h) > (altura - margin)):
        cv2.drawContours(mask_filtrada, [cnt], -1, 0, -1)

# 7. Usar Sobel sobre la máscara final para detectar los bordes de las letras
mask_float = np.float32(mask_filtrada)
sobelx = cv2.Sobel(mask_float, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(mask_float, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

sobel = np.uint8(sobel)
_, sobel_thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

# 8. Visualizar resultados (el proceso de filtrado y Sobel)
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask_black, cmap="gray")
plt.title("Sólo Negro (Blur + Morfología)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask_filtrada, cmap="gray")
plt.title("Sin Objetos Pequeños y Sin Esquinas")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(sobel_thresh, cmap="gray")
plt.title("Sobel - Letras Centrales")
plt.axis("off")

plt.tight_layout()
plt.show()

# 9. Mejora adicional para OCR (opcional)
# Invertir para tener letras negras sobre fondo blanco (esto mejora el reconocimiento en algunos casos)
sobel_inv = cv2.bitwise_not(sobel_thresh)

# Dilatación para unir componentes de letras (opcional)
kernel = np.ones((2, 2), np.uint8)
sobel_dilated = cv2.dilate(sobel_thresh, kernel, iterations=1)

# --------------- OCR SOBRE LA IMAGEN SOBEL_THRESH COMPLETA ---------------
# Inicializar el lector (se usa español)
reader = easyocr.Reader(['es'])

# Aplicar OCR sobre la imagen completa (sin delimitar regiones)
resultados = reader.readtext(sobel_thresh, detail=0)  # detail=0 devuelve solo el texto detectado

print("\n=== Texto detectado en la imagen completa (Sobel) ===")
print("Textos detectados:", resultados)

# Texto concatenado de todos los resultados
texto_completo = " ".join(resultados)
print(f"Texto completo: {texto_completo}")

# Visualizar solo el resultado final
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(sobel_thresh, cmap="gray")
plt.title("Imagen Sobel Final")
plt.axis("off")

# Crear una imagen para mostrar el texto detectado
texto_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(texto_img, "Texto detectado:", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Mostrar cada texto en una línea separada
y_pos = 80
for i, texto in enumerate(resultados):
    cv2.putText(texto_img, f"{i+1}. {texto}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_pos += 30

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(texto_img, cv2.COLOR_BGR2RGB))
plt.title("Resultados OCR")
plt.axis("off")

plt.tight_layout()
plt.show()

# 10. Prueba alternativa con la imagen invertida (opcional)
resultados_inv = reader.readtext(sobel_inv, detail=0)
print("\n=== Texto detectado en la imagen invertida ===")
print("Textos detectados:", resultados_inv)