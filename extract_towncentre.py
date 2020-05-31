import os
import cv2
import numpy as np

def video2im(src, dst='images', factor=1):
    """
    Extracts all frames from a video and saves them as jpgs
    """
    try:
      os.mkdir(dst)
    except Exception as e:
      print(e)

    frame = 0
    cap = cv2.VideoCapture(src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('Total Frame Count:', length )
            
    """
    Limitamos la cantidad de frames a extraer. Para obtenerlos todos, 
    base con modificar el valor de limitFrame
    """
    limitFrame = 10 

    while True:
        check, img = cap.read()
        if check:
            path = dst

            if frame == limitFrame: 
              break
            
            img = cv2.resize(img, (1920 // factor, 1080 // factor))
            cv2.imwrite(os.path.join(path, str(frame) + ".jpg"), img)

            frame += 1
            print('Processed: ',frame, end = '\r')
        
        else:
            break
    
    cap.release()

if __name__ == '__main__':
    video2im('TownCentreXVID.avi')
