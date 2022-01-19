'''
      PROYECTO DE PROCESAMIENTO DE IMAGENES
        CREACIÓN DE EFECTOS INTERACTIVOS

                    Autores:
                Bryan Loor Macias
            Livinsgton Perez Correa
'''

import cv2
import imutils
import tkinter

# Archivo de haarcascade para caracteristicas de detección de rostros
face_cascade = cv2.CascadeClassifier('./modelo/haarcascade_frontalface_alt.xml')

capture = cv2.VideoCapture(0)  # captura señal de video de cámara
cv2.namedWindow("CAPTURE", cv2.WINDOW_AUTOSIZE)

# capture= capturar_video(capture)

eye_op = rostro_op = nariz_op = boca_op = frente_op = barbilla_op = oreja_op = False

while (True):
    # Lectura del frame
    ret, frame = capture.read()

    # Convertir a escala de grises
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostro
    faces = face_cascade.detectMultiScale(frameGris, 1.1, 5)

    for (x, y, w, h) in faces:
        if (rostro_op):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        resolucion = (w + h) / 8

        # Cálculo de posición de ojos usando Ecuaciones
        if (eye_op):
            # Ojo derecho
            centro = (int(x + w * 0.3), int(y + h * 0.4))
            radio = int(resolucion * 0.3)
            # cv2.circle(frame,(centro[0],centro[1]),int(resolucion*0.3),(255,255,255),2)
            # cv2.circle(frame, (centro[0], centro[1]), 1, (255, 0, 255), 2)
            cv2.rectangle(frame,
                          (int(x + w * 0.32 - radio), int(y + h * 0.4 - radio)),
                          (int(x + w * 0.32 + radio), int(y + h * 0.4 + radio)),
                          (255, 255, 255), 2)

            # Ojo izquierdo
            # cv2.circle(frame, (int(x + w * 0.7), int(y + h * 0.4)), int(resolucion * 0.3), (255, 255, 255), 2)
            cv2.rectangle(frame,
                          (int(x + w * 0.7 - radio), int(y + h * 0.4 - radio)),
                          (int(x + w * 0.7 + radio), int(y + h * 0.4 + radio)),
                          (255, 255, 255), 2)

        # Cálculo de posición de las orejas usando Ecuaciones
        if (oreja_op):
            radio = int(resolucion * 0.3)
            # Oreja derecha

            # cv2.circle(frame, (int(x + w * 0.05), int(y + h * 0.45)), int(resolucion * 0.3), (20, 20, 140), 2)
            cv2.rectangle(frame,
                          (int(x + w * 0.05 - radio), int(y + h * 0.47 - radio)),
                          (int(x + w * 0.05 + radio), int(y + h * 0.47 + radio)),
                          (20, 20, 140), 2)
            # Oreja izquierda

            # cv2.circle(frame, (int(x + w * 0.95), int(y + h * 0.45)), int(resolucion * 0.3), (20, 20, 140), 2)
            cv2.rectangle(frame,
                          (int(x + w * 0.95 - radio), int(y + h * 0.47 - radio)),
                          (int(x + w * 0.95 + radio), int(y + h * 0.47 + radio)),
                          (20, 20, 140), 2)

        # Cálculo de posición de boca usando Ecuaciones
        if (boca_op):
            radio = int(resolucion) * 0.4

            cv2.rectangle(frame,
                          (int(x + w * 0.46 - radio), int(y + h * 0.82 - radio)),
                          (int(x + w * 0.56 + radio), int(y + h * 0.75 + radio)),
                          (0, 0, 255), 2)

            '''# arco inferior de boca

            cv2.ellipse(frame, (int(x + w * 0.5), int(y + h * 0.65)), (int(resolucion), int(resolucion)), 45, 10, 80,
                        (0, 0, 255), 0)
            # arco superior de boca
            cv2.ellipse(frame, (int(x + w * 0.5), int(y + h * 1.03)), (int(resolucion), int(resolucion)), 225, 10, 80,
                        (0, 0, 255), 0)'''

        # Cálculo de posición de nariz usando Ecuaciones
        if (nariz_op):
            cv2.rectangle(frame, (int(x + w * 0.45), int(y + h * 0.4)), (int(x + w * 0.55), int(y + h * 0.62)),
                          (0, 0, 0), 2)
            roi = frame[int(y + h * 0.4):int(y + h * 0.62), int(x + w * 0.45):int(x + w * 0.55)]
            frame[int(y + h * 0.4):int(y + h * 0.62), int(x + w * 0.45):int(x + w * 0.55)] = 0
            cv2.imshow("img", roi)
        # Cálculo de posición de frente usando Ecuaciones
        if (frente_op):
            cv2.rectangle(frame, (int(x + w * 0.1), int(y + h * 0.25)), (int(x + w * 0.9), int(y + h * -0.05)),
                          (140, 20, 20), 2)

        # Cálculo de posición de barbilla usando Ecuaciones
        if (barbilla_op):
            cv2.rectangle(frame, (int(x + w * 0.35), int(y + h * 1.1)), (int(x + w * 0.65), int(y + h * 0.93)),
                          (20, 140, 20), 2)

    # Muestra el video resultante con el rostro y sus características detectadas
    cv2.imshow("CAPTURE", frame)

    key = cv2.waitKey(33)  # Retraso en milisegundos para leer el siguiente frame

    if (key == 27):  # Termina presionando la tecla Esc
        break
    elif (key == 49):  # Tecla 1 activa la selección de rostro
        rostro_op = not rostro_op
    elif (key == 50):  # Tecla 2 activa la selección de ojos
        eye_op = not eye_op
    elif (key == 51):  # Tecla 3 activa la selección de boca
        boca_op = not boca_op
    elif (key == 52):  # Tecla 4 activa la selección de nariz
        nariz_op = not nariz_op
    elif (key == 53):  # Tecla 5 activa la selección de frente
        frente_op = not frente_op
    elif (key == 54):  # Tecla 6 activa la selección de barbilla
        barbilla_op = not barbilla_op
    elif (key == 55):  # Tecla 7 activa la selección de orejas
        oreja_op = not oreja_op

capture.release()
cv2.destroyAllWindows()
