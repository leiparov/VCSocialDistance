import os
import numpy as np
import cv2
from math import sqrt

def get_domain_boxes(person_boxes):
    """
    Retorna la lista final de bounding boxes según la clase que deseemos, cada bounding box
            contiene las coordenadas (left, top), el ancho y altura (width, height); y el punto central
            del bounding box (cx, cy)
    
    Parámetros:
            DataFrame de Bounding boxes
    """
    person_boxes['width'] = person_boxes['bodyRight'] - person_boxes['bodyLeft']
    person_boxes['height'] = person_boxes['bodyBottom'] - person_boxes['bodyTop']
    person_boxes['cx'] = person_boxes['bodyLeft'] + person_boxes['width']/2
    person_boxes['cy'] = person_boxes['bodyTop'] + person_boxes['height']/2

    cols = ['bodyLeft', 'bodyTop', 'width', 'height', 'cx', 'cy']
    
    boxes = [tuple(x.astype(int)) for x in person_boxes[cols].values]
    
    return boxes

def people_distances_bird_eye_view(boxes, distance_allowed, matrixh = None):
    """
    Esta función detecta si las personas respetan la distancia social.

    Parámetros:
        boxes: Bounding boxes obtenidos luego de aplicar la función get_domain_boxes
        distance_allowed: Distancia mínima permitida para indicar si una persona respeta o no la distancia social

    Salida:
        Retorna una tupla que contiene 2 listas:
        1. La primera lista contiene la información de los puntos (personas) que respetan la distancia social.
        2. La segunda lista contiene la información de los puntos (personas) que no respetan la distancia social.
    """
    people_bad_distances = []
    people_good_distances = []
    # Tomamos los valores center,bottom
    #print(matrixh)
    
#     #points = [[box[4],box[1]+box[3]] for box in boxes]
#     new_points = np.array([[[box[4],box[1]+box[3]] for box in boxes]], dtype=np.float32)
    
#     M = cv2.perspectiveTransform(new_points, matrixh)
#     result = M[0]
    result = __map_points_to_bird_eye_view([[box[4],box[1]+box[3]] for box in boxes], matrixh)[0]
    # Creamos nuevos bounding boxes con valores mapeados de bird eye view (8 elementos por item)
    # left, top, width, height, cx, cy, bev_cy, bev_cy
    new_boxes = [box + tuple(result) for box, result in zip(boxes, result)]

    for i in range(0, len(new_boxes)-1):
        for j in range(i+1, len(new_boxes)):
            cxi,cyi = new_boxes[i][6:]
            cxj,cyj = new_boxes[j][6:]
            distance = euclidean_distance([cxi,cyi], [cxj,cyj])
            if distance < distance_allowed:
                people_bad_distances.append(new_boxes[i])
                people_bad_distances.append(new_boxes[j])

    people_good_distances = list(set(new_boxes) - set(people_bad_distances))
    people_bad_distances = list(set(people_bad_distances))
    
    return (people_good_distances, people_bad_distances)

def draw_new_image_with_boxes(image, people_good_distances, people_bad_distances, distance_allowed, draw_lines=False):
    """
    Esta función se encarga de pintar los bounding boxes y la línea entre instancias del mismo tipo para tener mejor
    entendimiento de a que distancia se encuentran.

    Parámetros:
        image: Imagen original sobre la cual se dibujarán los bounding boxes y líneas
        people_good_distances: Lista de bounding boxes que sí respetan la distancia social
        people_bad_distances: Lista de bounding boxes que no respetan la distancia social
        distance_allowed: Valor de la distancia social mínima permitida
        draw_lines: Flag (True/False) para dibujar la línea entre dos puntos, sólo para las personas que no respetan 
                    la distancia social permitida.

    Salida:
        Retorna la nueva imagen con bounding boxes dibujados y si las líneas en caso se hayan habilitado.
    """
    green = (0, 255, 0)
    red = (255, 0, 0)
    new_image = image.copy()
    
    for person in people_bad_distances:
        left, top, width, height = person[:4]
        cv2.rectangle(new_image, (left, top), (left + width, top + height), red, 2)
    
    for person in people_good_distances:
        left, top, width, height = person[:4]
        cv2.rectangle(new_image, (left, top), (left + width, top + height), green, 2)
    
    if draw_lines:
        for i in range(0, len(people_bad_distances)-1):
            for j in range(i+1, len(people_bad_distances)):
                cxi,cyi,bevxi,bevyi = people_bad_distances[i][4:]
                cxj,cyj,bevxj,bevyj = people_bad_distances[j][4:]
                distance = euclidean_distance([bevxi, bevyi], [bevxj, bevyj])
                if distance < distance_allowed:
                    cv2.line(new_image, (cxi, cyi), (cxj, cyj), red, 2)
            
    return new_image

def __matrix_bird_eye_view():
    """
    Esta función retorna los valores ya obtenidos de la matríz de homografía en pasos previos.
    """
    return np.array([[ 5.59698570e-01,  4.76318646e+00,  1.31800029e+03],
       [-6.78672296e-01,  5.21600140e+00,  1.11148411e+03],
       [-1.17665406e-04,  1.33326098e-03,  1.00000000e+00]])

def __map_points_to_bird_eye_view(points, matrixh=None):
    """
    Esta función realiza el mapeo de puntos de la vista original hacia la vista Bird Eye

    Parámetros:
        points: Lista bidimensional de los puntos que se desean transformar hacia Bird Eye
        
    Salida:
        Retorna los nuevos puntos pertenecientes a la vista Bird Eye
    """
    if not isinstance(points, list):
        raise Exception("poinst must be a list of type [[x1,y1],[x2,y2],...]")
    
    if (matrixh is not None):
        matrix_transformation = __matrix_bird_eye_view()
    else:
        matrix_transformation = matrixh
        
    new_points = np.array([points], dtype=np.float32)
    
    return cv2.perspectiveTransform(new_points, matrix_transformation)
    
def euclidean_distance(point1, point2):
    """
    Esta función realiza el cálculo de la distancia euclidiana entre un par de puntos

    Parámetros:
        point1: Primer punto
        point2: Segundo punto

    Salida:
        Distancia euclidiana entre el par de puntos dados
    """
    x1,y1 = point1
    x2,y2 = point2
    return sqrt((x1-x2)**2 + (y1-y2)**2)


def video2im(src, dst='images'):
    """
    Extracts all frames from a video and saves them as jpgs
    """
    cap = cv2.VideoCapture(src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    frame = 0
    while True:
        check, img = cap.read()
        if check:
            cv2.imwrite(os.path.join(dst,"%d.jpg") %frame, img)
            frame += 1
            print(frame, end = '\r')
        else:
            break
    cap.release()
    
def put_text(frame, text, text_offset_y=50):
    font_scale = 2
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    rectangle_bgr = (0, 0, 0)
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]
    # set the text start position
    text_offset_x = frame.shape[1] - 600
    # make the coords of the box with a small padding of two pixels
    box_coords = (
        (text_offset_x, text_offset_y + 5),
        (text_offset_x + text_width + 2, text_offset_y - text_height - 2),
    )
    frame = cv2.rectangle(
        frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED
    )
    frame = cv2.putText(
        frame,
        text,
        (text_offset_x, text_offset_y),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=3,
    )

    return frame, 2 * text_height + text_offset_y

