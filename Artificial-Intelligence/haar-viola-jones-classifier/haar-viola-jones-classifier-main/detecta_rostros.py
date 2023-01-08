"""
************************************************************************************************************************************************
Código Python 3.x para detectar rostros en una imagen dada.
Desarrollado por: Ing. Ronny Díaz Lopez
Utilizamos la libreria OpenCV y el clasificador de Viola-Jones que incluye un clasificador tipo cascada de Haar.
Martes, 07 Junio de 2022

Para mayor información consulte el articulo "OpenCV Python (Parte 1 y 2)", publicado por mi en el 
blog de la empresa Hepyco Software en el link:  https://www.hepyco.com/blog/programacion/opencv-python-parte-1-de-2/
************************************************************************************************************************************************
"""

# Importamos las 2 librerias con las que vamos a traajar: cv2 de OpenCV y sys que nos permite ejecutar o cargar archivos desde la terminal del sistema.
import cv2
import numpy

"""
Definimos o configuramos los valores por defecto del camino o ruta donde se encuentran los dos archivos importantes para la correcta ejecución del programa, uno es: la ruta suministrada y almacenada en la variable rutaImagen, por la misma terminal del sistema desde donde ejecutaremos nuestro programa que en este caso sera la misma raiz o carpeta donde se encuentra nuestro programa detecta_rostros.py; y la otra ruta a indicar y que se alamacena en la variable rutaClasificador, que es donde se encuentra nuestro archivo con el codigo del clasificador Haas para deteccion de rostros en este caso el clasificadorhaar_viola-jones_rostro.xml.
En este ejemplo la imagen de prueba o fuente se llama imagen.jpg y se encuentra en la misma ubicacion raiz de nuestro programa python.

"""
#rutaImagen = sys.argv[1]
rutaImagen = "imagen.jpeg"
rutaClasificador = "clasificadorhaar_viola-jones_rostro.xml"

# Aqui creamos la cascada tipo haar utilizando el clasificador incorporado en la libreria OpenCV.
cascadaRostro = cv2.CascadeClassifier(rutaClasificador)

# Ahora leemos la imagen y la convertimos a tonos grises para prepararla para el clasificador viola-jones.
imagen = cv2.imread(rutaImagen)
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectamos los rostros en la imagen proporcionada
rostros = cascadaRostro.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    maxSize=(200, 200),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Encontramos {0} rostros!".format(len(rostros)))

# Ahora dibujamos un rectangulo alrededor de los rostros detectados
for (x, y, w, h) in rostros:
    cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Rostros Encontrados", imagen)
cv2.waitKey(0) # El bucle for continua mostrando los rostros detectados en la imagen hasta que se oprima cualquier tecla y se termina el programa.   
cv2.destroyAllWindows()
