'''
      PROYECTO DE PROCESAMIENTO DE IMAGENES
        CREACIÓN DE EFECTOS INTERACTIVOS

                    Autores:
                Bryan Loor Macias
            Livinsgton Perez Correa
'''

# importacion de librerías
import cv2
import imutils
from tkinter import *
from PIL import Image
from PIL import  ImageTk



#Definicion e implementacion de funciones

def empezar():
    global capture
    capture=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    visualizar()

def finalizar():
    global capture
    capture.release()

def setearOjos():
    global eye_op
    eye_op= not eye_op

def setearGorro():
    global cabeza_op
    cabeza_op= not cabeza_op


def setearNariz():
    global nariz_op
    nariz_op= not nariz_op


def setearOrejas():
    global oreja_op
    oreja_op= not oreja_op


def setearBoca():
    global boca_op
    boca_op= not boca_op


def visualizar():
    global capture,rostro_op,eye_op,oreja_op,boca_op,nariz_op,frente_op,barbilla_op
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    if capture is not None:
        ret, frame = capture.read()
        if ret == True:
            frame = imutils.resize(frame, width=640)
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


            # Convertir a escala de grises
            frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detección de rostro
            faces = face_cascade.detectMultiScale(frameGris, 1.1, 5)

            for (x, y, w, h) in faces:
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                resolucion = (w + h) / 8

                # Cálculo de posición de ojos usando Ecuaciones
                if (eye_op):
                    factor_row1, factor_row2, factor_col1, factor_col2 = 0.32, 0.32, 0.4, 0.4
                    factor2_row1, factor2_row2 = 0.7, 0.7

                    radio = int(resolucion * 0.9)
                    row = int(x + w * factor_row2 + radio) - int(x + w * factor_row1 - radio)
                    col = int(y + h * factor_col2 + radio) - int(y + h * factor_col1 - radio)

                    ojo_z = cv2.imread("imagenes/ojo1.png", cv2.IMREAD_UNCHANGED)
                    ojo_d = cv2.imread("imagenes/ojo2.png", cv2.IMREAD_UNCHANGED)


                    # Ojo derecho

                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    resized_image_d = cv2.resize(ojo_d, (row, col))
                    resized_image_d = cv2.cvtColor(resized_image_d, cv2.COLOR_BGRA2RGBA)

                    n_frame_d = frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                                int(x + w * factor2_row1 - radio):int(x + w * factor2_row2 + radio)]

                    # Determinamos la máscara que posee la imagen de entrada
                    # redimensionada y también la invertimos
                    mask = resized_image_d[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)

                    # Creamos una imagen con fondo negro y el gorro/tiara/2021
                    # Luego creamos una imagen en donde en el fondo esté frame

                    bg_black = cv2.bitwise_and(resized_image_d, resized_image_d, mask=mask)
                    bg_black = bg_black[:, :, 0:3]
                    bg_frame = cv2.bitwise_and(n_frame_d, n_frame_d, mask=mask_inv[:, :])
                    bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_OPEN, kernel)
                    bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_CLOSE, kernel)

                    # Sumamos las dos imágenes obtenidas
                    result = cv2.add(bg_black, bg_frame)
                    print(n_frame_d.shape[1], resized_image_d.shape[1])
                    if (n_frame_d.shape[1] == resized_image_d.shape[1]):
                        frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                        int(x + w * factor2_row1 - radio):int(x + w * factor2_row2 + radio)] = result



                    # Ojo izquierdo

                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    resized_image_z = cv2.resize(ojo_z, (row, col))
                    resized_image_z = cv2.cvtColor(resized_image_z, cv2.COLOR_BGRA2RGBA)

                    n_frame_z = frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                                int(x + w * factor_row1 - radio):int(x + w * factor_row2 + radio)]

                    # Determinamos la máscara que posee la imagen de entrada
                    # redimensionada y también la invertimos
                    mask = resized_image_z[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)

                    # Creamos una imagen con fondo negro y el gorro/tiara/2021
                    # Luego creamos una imagen en donde en el fondo esté frame

                    bg_black = cv2.bitwise_and(resized_image_z, resized_image_z, mask=mask)
                    bg_black = bg_black[:, :, 0:3]
                    bg_frame = cv2.bitwise_and(n_frame_z, n_frame_z, mask=mask_inv[:, :])
                    bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_OPEN, kernel)
                    bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_CLOSE, kernel)

                    # Sumamos las dos imágenes obtenidas
                    result = cv2.add(bg_black, bg_frame)
                    print(n_frame_z.shape[1], resized_image_z.shape[1])
                    if (n_frame_z.shape[1] == resized_image_z.shape[1]):
                        frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                        int(x + w * factor_row1 - radio):int(x + w * factor_row2 + radio)] = result


                # Cálculo de posición de las orejas usando Ecuaciones
                if (oreja_op):

                    factor_row1, factor_row2, factor_col1, factor_col2 = 0.01, 0.01, 0.47, 0.47
                    factor2_row1,factor2_row2=0.95,0.95

                    radio = int(resolucion * 2)
                    row=int(x + w * factor_row2 + radio)-int(x + w * factor_row1 - radio)
                    col=int(y + h * factor_col2 + radio)-int(y + h * factor_col1 - radio)

                    oreja_d=cv2.imread("imagenes/oreja.png",cv2.IMREAD_UNCHANGED)
                    oreja_z=cv2.imread("imagenes/oreja2.png",cv2.IMREAD_UNCHANGED)


                    # Oreja izquierda
                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    resized_image = cv2.resize(oreja_z, (row, col))
                    resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)


                    n_frame = frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                        int(x + w * factor_row1 - radio):int(x + w * factor_row2 + radio)]

                    if (n_frame.shape[1] == resized_image.shape[1]):
                        # Determinamos la máscara que posee la imagen de entrada
                        # redimensionada y también la invertimos
                        mask = resized_image[:, :, 3]
                        mask_inv = cv2.bitwise_not(mask)

                        # Creamos una imagen con fondo negro y el gorro/tiara/2021
                        # Luego creamos una imagen en donde en el fondo esté frame

                        bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
                        bg_black = bg_black[:, :, 0:3]
                        bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[:, :])

                        bg_frame = cv2.morphologyEx(bg_frame,cv2.MORPH_OPEN,kernel)
                        bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_CLOSE, kernel)

                        # Sumamos las dos imágenes obtenidas
                        result = cv2.add(bg_black, bg_frame)
                        frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                            int(x + w * factor_row1 - radio):int(x + w * factor_row2 + radio)] = result


                    # Oreja derecha

                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    resized_image_d = cv2.resize(oreja_d, (row, col))
                    resized_image_d = cv2.cvtColor(resized_image_d, cv2.COLOR_BGRA2RGBA)
                    n_frame_d = frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                        int(x + w * factor2_row1 - radio):int(x + w * factor2_row2 + radio)]

                    if (n_frame_d.shape[1] == resized_image_d.shape[1]):

                        # Determinamos la máscara que posee la imagen de entrada
                        # redimensionada y también la invertimos
                        mask = resized_image_d[:, :, 3]
                        mask_inv = cv2.bitwise_not(mask)

                        # Creamos una imagen con fondo negro y el gorro/tiara/2021
                        # Luego creamos una imagen en donde en el fondo esté frame

                        bg_black = cv2.bitwise_and(resized_image_d, resized_image_d, mask=mask)
                        bg_black = bg_black[:, :, 0:3]
                        bg_frame = cv2.bitwise_and(n_frame_d, n_frame_d, mask=mask_inv[:, :])
                        bg_frame = cv2.morphologyEx(bg_frame,cv2.MORPH_OPEN,kernel)
                        bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_CLOSE, kernel)

                        # Sumamos las dos imágenes obtenidas
                        result = cv2.add(bg_black, bg_frame)

                        frame[int(y + h * factor_col1 - radio):int(y + h * factor_col2 + radio),
                        int(x + w * factor2_row1 - radio):int(x + w * factor2_row2 + radio)] = result


                    # Cálculo de posición de boca usando Ecuaciones
                if (boca_op):
                    factor_row1, factor_row2, factor_col1, factor_col2 = 0.33, 0.63, 0.87, 0.80
                    radio = int(resolucion) * 1.5

                    row = int(x + w * factor_row2 + radio) - int(x + w * factor_row1 - radio)
                    col = int(y + h * factor_col2 + radio) - int(y + h * factor_col1 - radio)

                    boca= cv2.imread("imagenes/boca.png",cv2.IMREAD_UNCHANGED)

                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    resized_image = cv2.resize(boca, (row, col))
                    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGBA)

                    n_frame = frame[int(y + h * factor_col1 - radio):int(y + h *factor_col2 + radio),
                                int(x + w * factor_row1 - radio):int(x + w * factor_row2 + radio)]

                    #print(n_frame.shape[1],resized_image.shape[1])
                    #print(n_frame.shape[0], resized_image.shape[0])
                    img_altura=int(y + h * factor_col1 - radio)+resized_image.shape[0]
                    print(img_altura,frame.shape[0])
                    if (img_altura<=frame.shape[0]):
                        # Determinamos la máscara que posee la imagen de entrada
                        # redimensionada y también la invertimos
                        mask = resized_image[:, :, 3]
                        mask_inv = cv2.bitwise_not(mask)


                        # Creamos una imagen con fondo negro y el gorro/tiara/2021
                        # Luego creamos una imagen en donde en el fondo esté frame
                        # y en negro el gorro/tiara/2021
                        bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
                        bg_black = bg_black[:, :, 0:3]
                        bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[:, :])

                        bg_frame = cv2.morphologyEx(bg_frame,cv2.MORPH_OPEN,kernel)


                        # Sumamos las dos imágenes obtenidas
                        result = cv2.add(bg_black, bg_frame)
    
                        frame[int(y + h * factor_col1 - radio):int(y + h *factor_col2 + radio),
                                    int(x + w * factor_row1 - radio):int(x + w * factor_row2 + radio)]= result


                # Cálculo de posición de nariz usando Ecuaciones
                if (nariz_op):
                    factor_row1,factor_row2,factor_col1,factor_col2=0.35,0.70,0.4,0.7
                    row = int(x + w * factor_row2) - int(x + w * factor_row1)
                    col = int(y + h * factor_col2) - int(y + h * factor_col1)

                    nariz=cv2.imread("imagenes/nariz.png",cv2.IMREAD_UNCHANGED)
                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    resized_image = cv2.resize(nariz, (row,col))
                    resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)

                    n_frame = frame[int(y + h * factor_col1):int(y + h * factor_col2),
                                    int(x + w * factor_row1):int(x + w * factor_row2)]

                    # Determinamos la máscara que posee la imagen de entrada
                    # redimensionada y también la invertimos
                    mask = resized_image[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)

                    # Creamos una imagen con fondo negro y el gorro/tiara/2021
                    # Luego creamos una imagen en donde en el fondo esté frame
                    # y en negro el gorro/tiara/2021
                    bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
                    bg_black = bg_black[:, :, 0:3]
                    bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[:, :])
                    bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_OPEN, kernel)

                    # Sumamos las dos imágenes obtenidas
                    result = cv2.add(bg_black, bg_frame)

                    frame[int(y + h * factor_col1):int(y + h * factor_col2), int(x + w * factor_row1):int(x + w * factor_row2)]=result

                # Cálculo de posición de frente usando Ecuaciones
                if (cabeza_op):

                    # Cálculo de posición de frente usando Ecuaciones
                    gorro = cv2.imread('imagenes/cabeza.png', cv2.IMREAD_UNCHANGED)

                    # Redimensionar la imagen de entrada de acuerdo al ancho del
                    # rostro detectado
                    resized_image = imutils.resize(gorro, width=w)
                    resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGRA2RGBA)
                    filas_image = resized_image.shape[0]
                    col_image = w

                    # Determinar una porción del alto de la imagen de entrada
                    # redimensionada
                    porcion_alto = filas_image // 4

                    dif = 0

                    # Si existe suficiente espacio sobre el rostro detectado
                    # para insertar la imagen de entrada resimensionada
                    # se visualizará dicha imagen
                    if y + porcion_alto - filas_image >= 0:
                        # Tomamos la sección de frame, en donde se va a ubicar
                        # el gorro/tiara
                        n_frame = frame[y + porcion_alto - filas_image: y + porcion_alto,
                                  x: x + col_image]
                    else:
                        # Determinamos la sección de la imagen que excede a la del video
                        dif = abs(y + porcion_alto - filas_image)
                        # Tomamos la sección de frame, en donde se va a ubicar
                        # el gorro/tiara
                        n_frame = frame[0: y + porcion_alto,
                                  x: x + col_image]

                    # Determinamos la máscara que posee la imagen de entrada
                    # redimensionada y también la invertimos
                    mask = resized_image[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)

                    # Creamos una imagen con fondo negro y el gorro/tiara/2021
                    # Luego creamos una imagen en donde en el fondo esté frame
                    # y en negro el gorro/tiara/2021
                    bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
                    bg_black = bg_black[dif:, :, 0:3]
                    bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:, :])
                    bg_frame = cv2.morphologyEx(bg_frame, cv2.MORPH_OPEN, kernel)

                    # Sumamos las dos imágenes obtenidas
                    result = cv2.add(bg_black, bg_frame)
                    if y + porcion_alto - filas_image >= 0:
                        frame[y + porcion_alto - filas_image: y + porcion_alto, x: x + col_image] = result

                    else:
                        frame[0: y + porcion_alto, x: x + col_image] = result


            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)

        else:
            lblVideo.image=""
            capture.release()



'''
    Ejecucion del programa
'''

# Archivo de haarcascade para caracteristicas de detección de rostros
face_cascade = cv2.CascadeClassifier('./modelo/haarcascade_frontalface_alt.xml')

eye_op = rostro_op = nariz_op = boca_op = cabeza_op = barbilla_op = oreja_op = False
capture=None

root = Tk()
root.title("GENERACION DE EFECTOS INTERACTIVOS SOBRE EL ROSTRO")
root.geometry('900x640')

btnEmpezar= Button(root,text="Iniciar video",font=("Ravie"),bg="orange", fg="green",width=20,command=empezar)
btnEmpezar.grid(column=0,row=0,padx=5,pady=5,columnspan=2)

btnEmpezar= Button(root,text="Finalizar video",font=("Ravie"),bg="red", fg="blue",width=20,command=finalizar)
btnEmpezar.grid(column=1,row=0,padx=5,pady=5,columnspan=2)


lblVideo =Label(root, bg='green')
lblVideo.grid(column=0,row=1,columnspan=3)
img= cv2.imread("imagenes/fondo.png")
img=cv2.resize(img,(640,500))
im = Image.fromarray(img)
img = ImageTk.PhotoImage(image=im)
lblVideo.configure(image=img)
lblVideo.image = img


btnOjos=Button(root,text="Efectos sobre Ojos",font=("Roboto", 10),bg="green", fg="white",width=40,command=setearOjos)
btnOjos.grid(column=0,row=2,padx=5,pady=5,columnspan=1)

btnGorro=Button(root,text="Efectos sobre Cabeza",font=("Roboto", 10),bg="green", fg="white",width=40,command=setearGorro)
btnGorro.grid(column=1,row=2,padx=5,pady=5,columnspan=1)

btnOrejas=Button(root,text="Efectos sobre Orejas",font=("Roboto", 10),bg="green", fg="white",width=40,command=setearOrejas)
btnOrejas.grid(column=2,row=2,padx=5,pady=5,columnspan=1)

btnBoca=Button(root,text="Efectos sobre Boca",font=("Roboto", 10),bg="green", fg="white",width=40,command=setearBoca)
btnBoca.grid(column=0,row=3,padx=5,pady=5,columnspan=1)

btnNariz=Button(root,text="Efectos sobre Nariz",font=("Roboto", 10),bg="green", fg="white",width=40,command=setearNariz)
btnNariz.grid(column=1,row=3,padx=5,pady=5,columnspan=1)

root.resizable (0,0) #Evitar que los usuarios ajusten el tamaño
root.mainloop()