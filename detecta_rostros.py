import cv2

def detect(path):
    img = cv2.imread(path)
    # Pasamos la ruta del XML
    cascade = cv2.CascadeClassifier(r"C:\Users\ronny\Documents\Dev_Projects\detector_rostros_imagen\clasificador_haar_viola-jones.xml")
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.CASCADE_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    # Ruta donde guardaremos la imagen con la deteccion de rostros
    cv2.imwrite('resultado.jpg', img);

# Pasamos la imagen que quremos detectar
rects, img = detect("imagen.jpg")
box(rects, img)